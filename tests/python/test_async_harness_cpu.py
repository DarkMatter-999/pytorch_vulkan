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
    block = source[start:source.index("// The public entry now accepts", start)]
    for match in re.finditer(r"\.device\(kDevice\)|options\(\)\.device\(kDevice\)", block):
        line = block[:match.start()].splitlines()[-1] + block[match.start():].splitlines()[0]
        assert "kLong" not in line and "kDouble" not in line and "dtype)" not in line, line
    assert "for (const auto dtype : {at::kDouble, at::kLong, at::kBool}) {\n            auto cpu" in block
    assert "at::empty({4}, cpu.options().device(kDevice))" not in block


def _function_body(source, signature):
    start = source.index(signature)
    brace = source.index("{", source.index(")", start))
    depth = 0
    for index in range(brace, len(source)):
        depth += {"{": 1, "}": -1}.get(source[index], 0)
        if depth == 0:
            return source[brace:index + 1]
    raise AssertionError("unbalanced function body")


def test_async_copy_tensor_blocking_paths_use_internal_helper_not_public_copy_tensor():
    """Source-level guard (fragile to refactors that rename things): the async entry
    must reach blocking behavior through copy_tensor_blocking_impl, so that wiring
    the public copy_tensor to async_copy_tensor later cannot recurse. Structural
    checks would need a linkable Vulkan TU, which CPU-only gates avoid."""
    import re
    source = (Path(__file__).parents[2] / "src/vulkan_transfer.cpp").read_text()
    async_body = _function_body(source, "at::Tensor &async_copy_tensor(at::Tensor &destination,")
    assert not re.search(r"(?<![A-Za-z_])copy_tensor\(", async_body)
    assert async_body.count("copy_tensor_blocking_impl(") == 2
    assert "copy_tensor_blocking_impl(destination, source, non_blocking)" in async_body
    assert "copy_tensor_blocking_impl(destination, source, false)" in async_body
    # Since the Task 5 flip copy_tensor also routes F32 pairs to async_copy_tensor;
    # see test_public_flip_routes_f32_through_async_entry_without_recursion.
    forwarder = _function_body(source, "\nat::Tensor &copy_tensor(at::Tensor &destination, const at::Tensor &source,")
    assert "return copy_tensor_blocking_impl(destination, source, non_blocking);" in forwarder


ROOT_DIR = Path(__file__).parents[2]


def _read(relative):
    return (ROOT_DIR / relative).read_text()


def test_public_flip_routes_f32_through_async_entry_without_recursion():
    source = _read("src/vulkan_transfer.cpp")
    forwarder = _function_body(source, "\nat::Tensor &copy_tensor(at::Tensor &destination, const at::Tensor &source,")
    assert "return async_copy_tensor(destination, source, non_blocking);" in forwarder
    assert "return copy_tensor_blocking_impl(destination, source, non_blocking);" in forwarder
    async_body = _function_body(source, "at::Tensor &async_copy_tensor(at::Tensor &destination,")
    import re
    assert not re.search(r"(?<![A-Za-z_])copy_tensor\(", async_body)


def test_to_copy_has_one_owner_and_f32_non_blocking_gates_are_gone():
    import re
    owners = [path for path in (ROOT_DIR / "src").rglob("*.cpp")
              if re.search(r'm\.impl\("_to_copy"', path.read_text())]
    assert [path.name for path in owners] == ["f32_to_double.cpp"]
    text = _read("src/vulkan/operators/f32_to_double.cpp")
    # Only the Vulkan Double formatter conversion keeps a non_blocking rejection.
    assert text.count("non_blocking=True") == 1
    assert "formatter conversion does not support non_blocking=True" in text
    assert ".pinned_memory(pin_out)," in text
    assert "pin_out || pin_memory" not in text  # N1: explicit pin_memory does not pin (PyTorch 2.4)
    assert "non_blocking && input.is_privateuseone()" in text
    assert "Vulkan _to_copy does not support pinned memory" in text  # H2D pin_memory stays rejected
    assert "does not support pinned CPU output" not in text
    assert "pytorch_vulkan::copy_tensor(output, input, non_blocking);" in text


def test_copy_from_gate_routes_non_blocking_readback_through_copy_tensor():
    text = _read("src/vulkan_allocator.cpp")
    body = _function_body(text, "at::Tensor vulkan_copy_from(const at::Tensor &source,")
    assert "pytorch_vulkan::copy_tensor(result, source, true);" in body
    assert "does not support non_blocking=True" not in body


def test_async_transfer_witness_module_uses_only_existing_native_seams():
    import re
    module = _read("src/python_module.cpp")
    defined = set(re.findall(r'module\.def(?:_python)?\("([A-Za-z_0-9]+)"', module))
    used = set(re.findall(r"\bapi\.([A-Za-z_0-9]+)\(", _read("tests/python/test_vulkan_async_transfer.py")))
    assert used, "witness module no longer uses any native seam"
    assert used <= defined, sorted(used - defined)


def test_non_f32_non_blocking_rejection_tests_are_untouched():
    for name in ("tests/python/test_vulkan_bool_transfer.py", "tests/python/test_vulkan_dtypes.py"):
        text = _read(name)
        assert 'match="non_blocking"' in text or '"non_blocking"' in text, name


# The backend registers empty/empty_strided/fill_/zero_/copy_ but no arange, linspace,
# rand*, eye, ... kernels (b1 failed with NotImplementedError 'aten::arange.start_out'
# on 'vk'). Device-side factories are limited to ones with passing committed uses on
# vk (empty*/zeros/ones/full/tensor reduce to empty + zero_/fill_/copy_); every other
# tensor is CPU-seeded and moved with .to(vk).
DEVICE_SIDE_FACTORY_ALLOWLIST = {"empty", "empty_like", "empty_strided", "zeros", "ones",
                                 "full", "tensor"}


def unsupported_device_factories(source, filename="<module>"):
    """Static (ast, no execution) scan for torch.<factory>(..., device=vk) calls."""
    import ast

    def is_vk(node):
        return (isinstance(node, ast.Name) and node.id in {"vk", "vulkan_backend"}) or (
            isinstance(node, ast.Constant) and node.value == "vk"
        )

    found = []
    for node in ast.walk(ast.parse(source, filename)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
            continue
        if any(keyword.arg == "device" and is_vk(keyword.value) for keyword in node.keywords):
            if node.func.attr not in DEVICE_SIDE_FACTORY_ALLOWLIST:
                found.append((node.lineno, node.func.attr))
    return found


@pytest.mark.parametrize("name", ["test_vulkan_async_transfer.py", "test_vulkan_transfer.py",
                                  "test_vulkan_copy_contracts.py"])
def test_edited_transfer_modules_use_no_unsupported_device_side_factories(name):
    source = (Path(__file__).parent / name).read_text()
    assert unsupported_device_factories(source, name) == []


def test_device_factory_guard_detects_the_b1_pre_fix_module():
    old = Path(__file__).parents[2] / "tmp/task5-build-20261030-02/old_async_transfer_module.py"
    if not old.exists():
        pytest.skip("pre-fix module copy not retained")
    found = unsupported_device_factories(old.read_text(), old.name)
    assert [name for _, name in found] == ["arange"] * 4  # the four b1 failing sites
