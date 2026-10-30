import hashlib
import importlib.machinery
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_vulkan_async_execution as harness


def fake_build(tmp_path, name, contents):
    build = tmp_path / name
    package = build / "pytorch_vulkan"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "import importlib.machinery, importlib.util\n"
        f"loader = importlib.machinery.SourceFileLoader('pytorch_vulkan._C', {str(package / ('_C' + importlib.machinery.EXTENSION_SUFFIXES[0]))!r})\n"
        "spec = importlib.util.spec_from_loader(loader.name, loader)\n"
        "_C = importlib.util.module_from_spec(spec)\nloader.exec_module(_C)\n"
    )
    extension = package / ("_C" + importlib.machinery.EXTENSION_SUFFIXES[0])
    extension.write_text(contents)
    return build, hashlib.sha256(extension.read_bytes()).hexdigest()


def configured_env(build, sha):
    env = os.environ.copy()
    env["PYTORCH_VULKAN_TEST_BUILD"] = str(build)
    env["PYTORCH_VULKAN_TEST_EXTENSION_SHA256"] = sha
    return env


def test_selected_build_precedes_competing_pythonpath(tmp_path):
    chosen, sha = fake_build(tmp_path, "chosen", "value = 'chosen'\n")
    other, _ = fake_build(tmp_path, "other", "value = 'wrong'\n")
    env = configured_env(chosen, sha)
    env["PYTHONPATH"] = str(other)
    result = harness._run_child("import pytorch_vulkan; print(pytorch_vulkan._C.value)", env)
    assert result.stdout.strip() == "chosen"


def test_child_rejects_binding_hash_mismatch(tmp_path):
    build, _ = fake_build(tmp_path, "build", "value = 1\n")
    with pytest.raises(AssertionError, match="SHA256"):
        harness._run_child("print('must not run')", configured_env(build, "0" * 64))


def test_child_rejects_binding_outside_selected_build(tmp_path):
    build, _ = fake_build(tmp_path, "build", "value = 1\n")
    other, sha = fake_build(tmp_path, "other", "value = 2\n")
    (build / "pytorch_vulkan" / "__init__.py").write_text(
        "import importlib.util\n"
        "import importlib.machinery\n"
        f"loader = importlib.machinery.SourceFileLoader('_C', {str(other / 'pytorch_vulkan' / ('_C' + importlib.machinery.EXTENSION_SUFFIXES[0]))!r})\n"
        "spec = importlib.util.spec_from_loader('_C', loader)\n"
        "_C = importlib.util.module_from_spec(spec)\nloader.exec_module(_C)\n"
    )
    with pytest.raises(AssertionError, match="outside"):
        harness._run_child("pass", configured_env(build, sha))


@pytest.mark.parametrize("marker", ["Validation Error: broken", "VUID-test-broken"])
def test_success_stderr_validation_is_not_hidden(tmp_path, marker):
    build, sha = fake_build(tmp_path, "build", "value = 1\n")
    with pytest.raises(AssertionError, match=marker):
        harness._run_child(f"import sys; print({marker!r}, file=sys.stderr)", configured_env(build, sha))


@pytest.mark.parametrize("termination", ["sys.exit(70)", f"os.kill(os.getpid(), {signal.SIGTERM})"])
def test_terminal_fault_reports_code_and_both_streams(tmp_path, termination):
    build, sha = fake_build(tmp_path, "build", "value = 1\n")
    script = f"import os, sys; print('stdout evidence', flush=True); print('stderr evidence', file=sys.stderr, flush=True); {termination}"
    with pytest.raises(AssertionError) as error:
        harness._run_child(script, configured_env(build, sha))
    message = str(error.value)
    assert "stdout evidence" in message and "stderr evidence" in message
    assert ("70" if "exit" in termination else str(-signal.SIGTERM)) in message


def test_live_output_reaches_outer_pytest_receipt_before_timeout(tmp_path):
    build, sha = fake_build(tmp_path, "build", "value = 1\n")
    gate = tmp_path / "test_gate.py"
    script = "import os,time; os.write(1,b'live-out\\n'); os.write(2,b'live-err\\n'); time.sleep(30)"
    gate.write_text(
        f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_vulkan_async_execution as h\n"
        f"def test_gate():\n h._run_child({script!r}, {configured_env(build, sha)!r})\n"
    )
    out, err = tmp_path / "out", tmp_path / "err"
    with out.open("wb") as stdout, err.open("wb") as stderr:
        process = subprocess.Popen([sys.executable, "-m", "pytest", "-s", "-q", str(gate)],
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and (b"live-out" not in out.read_bytes() or b"live-err" not in err.read_bytes()):
                time.sleep(0.05)
            assert process.poll() is None
            assert b"live-out" in out.read_bytes()
            assert b"live-err" in err.read_bytes()
        finally:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_success_output_visible_in_explicit_uncaptured_pytest_gate(tmp_path):
    build, sha = fake_build(tmp_path, "build", "value = 1\n")
    gate = tmp_path / "test_gate.py"
    script = "import sys; print('success-out'); print('success-err', file=sys.stderr)"
    gate.write_text(
        f"import sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_vulkan_async_execution as h\n"
        f"def test_gate():\n h._run_child({script!r}, {configured_env(build, sha)!r})\n"
    )
    result = subprocess.run([sys.executable, "-m", "pytest", "-s", "-q", str(gate)], capture_output=True, timeout=10)
    assert result.returncode == 0
    assert b"success-out" in result.stdout and b"success-err" in result.stderr


@pytest.mark.parametrize("termination", ["sys.exit(70)", f"os.kill(os.getpid(), {signal.SIGTERM})"])
def test_malformed_bytes_do_not_mask_terminal_fault(tmp_path, termination):
    build, sha = fake_build(tmp_path, "build", "value = 1\n")
    script = f"import os,sys; os.write(1,b'out\\xff'); os.write(2,b'err\\xfe'); {termination}"
    with pytest.raises(AssertionError) as error:
        harness._run_child(script, configured_env(build, sha))
    message = str(error.value)
    assert r"out\xff" in message and r"err\xfe" in message
    assert ("70" if "exit" in termination else str(-signal.SIGTERM)) in message


def test_task4_native_witness_builds_only_allocator_accepted_vulkan_dtypes():
    """The Vulkan allocator rejects Long (outside index/label paths) and Double
    tensors; a witness constructing them as Vulkan operands fails G1 at runtime."""
    import re
    source = (Path(__file__).parents[2] / "tests/cpp/vulkan_transfer_tensor_test.cpp").read_text()
    start = source.index("// N2: non-F32 and two-CPU pairs")
    block = source[start:source.index("// The public entry still rejects", start)]
    for match in re.finditer(r"\.device\(kDevice\)|options\(\)\.device\(kDevice\)", block):
        line = block[:match.start()].splitlines()[-1] + block[match.start():].splitlines()[0]
        assert "kLong" not in line and "kDouble" not in line and "dtype)" not in line, line
    assert "for (const auto dtype : {at::kDouble, at::kLong, at::kBool}) {\n            auto cpu" in block
    assert "at::empty({4}, cpu.options().device(kDevice))" not in block
