"""CPU-only compiler selection and read-only operator verification contracts."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import struct
import io
import tarfile
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest


ROOT = Path(__file__).resolve().parents[2]
OPERATOR = ROOT / "tools/verify_operator_spv.py"
BINARY = struct.pack("<7I", 0x07230203, 0x00010000, 0, 2, 0, 0x00020011, 1)
HOST_FIELDS = """uint32_t mode; uint32_t batch; uint32_t channels; uint32_t spatial;
uint32_t classes; int32_t ignore_index; uint32_t padding0; uint32_t padding1;
float momentum; float eps;"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("failure", ["stage-header", "stage-manifest", "publish-header", "publish-manifest"])
def test_operator_generation_write_failure_preserves_both_products(operator_fixture, monkeypatch, failure):
    from tools import generate_operator_spv as generator
    root, _, _ = operator_fixture
    directory = root / "src/vulkan/shaders/generated"
    before = snapshot(root)
    monkeypatch.setattr(sys, "argv", ["generate", "--output-directory", str(directory)])
    monkeypatch.setattr(generator, "SOURCES", [root / f"src/vulkan/shaders/glsl/{n}.comp" for n in ("normalization", "classification")])
    def compile_fixture(argv, **kwargs):
        Path(argv[argv.index("-o") + 1]).write_bytes(BINARY)
    monkeypatch.setattr(generator.subprocess, "run", compile_fixture)
    write = Path.write_bytes
    replace = os.replace
    name = "operator_spv.h" if failure.endswith("header") else "operator_spv.sha256"
    def fail_write(path, data):
        if path.name == name:
            write(path, data[:3])  # Partial staging write must not reach products.
            raise OSError("injected staging failure")
        return write(path, data)
    tripped = False
    def fail_replace(src, dst):
        nonlocal tripped
        if not tripped and Path(dst) == directory / name:
            tripped = True
            raise OSError("injected publication failure")
        return replace(src, dst)
    if failure.startswith("stage"):
        monkeypatch.setattr(Path, "write_bytes", fail_write)
    else:
        monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        generator.main()
    assert snapshot(root) == before


def tar_member(name, kind=tarfile.REGTYPE, link=""):
    member = tarfile.TarInfo(name)
    member.type = kind
    member.linkname = link
    member.size = 4 if kind == tarfile.REGTYPE else 0
    return member


def hostile_members(case):
    return {
        "absolute": [tar_member("/outside")],
        "traversal": [tar_member("usr/../../outside")],
        "nul": [tar_member("usr/nul\x00outside")],
        "device": [tar_member("usr/device", tarfile.CHRTYPE)],
        "fifo": [tar_member("usr/fifo", tarfile.FIFOTYPE)],
        "symlink-absolute": [tar_member("usr/link", tarfile.SYMTYPE, "/outside")],
        "symlink-traversal": [tar_member("usr/link", tarfile.SYMTYPE, "../../outside")],
        "hardlink-absolute": [tar_member("usr/link", tarfile.LNKTYPE, "/outside")],
        "hardlink-traversal": [tar_member("usr/link", tarfile.LNKTYPE, "../outside")],
        "chain": [tar_member("usr/a", tarfile.SYMTYPE, "b"), tar_member("usr/b", tarfile.SYMTYPE, "../../outside")],
        "cycle": [tar_member("usr/a", tarfile.SYMTYPE, "b"), tar_member("usr/b", tarfile.SYMTYPE, "a")],
        "parent-link-first": [tar_member("usr/a", tarfile.SYMTYPE, "."), tar_member("usr/a/data")],
        "parent-link-last": [tar_member("usr/a/data"), tar_member("usr/a", tarfile.SYMTYPE, ".")],
        "hardlink-chain": [tar_member("usr/a", tarfile.LNKTYPE, "usr/b"), tar_member("usr/b", tarfile.SYMTYPE, "../../outside")],
        "duplicate": [tar_member("usr/data"), tar_member("usr/data", tarfile.SYMTYPE, ".")],
        "extra-member": [tar_member("usr/bin/glslc"), tar_member("../extra")],
    }[case]


@pytest.mark.parametrize("case", ["absolute", "traversal", "nul", "device", "fifo", "symlink-absolute",
    "symlink-traversal", "hardlink-absolute", "hardlink-traversal", "chain", "cycle",
    "parent-link-first", "parent-link-last", "hardlink-chain", "duplicate", "extra-member"])
def test_archive_manifest_rejects_hostile_paths_before_destination_exists(tmp_path, case):
    module = selector()
    preflight = getattr(module, "preflight_archives", None)
    assert preflight is not None, "archive path/link preflight is not implemented"
    members = [tar_member("safe/data")] + hostile_members(case)
    destination = tmp_path / "profile"
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="archive"):
        preflight([members])
    assert not destination.exists() and snapshot(tmp_path) == before


def test_whole_package_set_preflight_precedes_any_profile_creation(tmp_path, monkeypatch):
    module = selector()
    cache = tmp_path / "cache"
    cache.mkdir()
    packages = {}
    for name, members in [("one.tar", [tar_member("safe/data")]), ("two.tar", [tar_member("../outside")])]:
        archive = cache / name
        with tarfile.open(archive, "w") as tar:
            for member in members:
                tar.addfile(member, io.BytesIO(b"data") if member.isfile() else None)
        (cache / (name + ".sig")).write_bytes(b"fixture signature")
        packages[name] = {"sha256": digest(archive.read_bytes()), "signer": "B" * 40}
    monkeypatch.setattr(module, "APPROVED_PACKAGES", packages)
    run = subprocess.run
    def signature_only(argv, **kwargs):
        if argv[0] == "gpgv":
            return subprocess.CompletedProcess(argv, 0, "[GNUPG:] VALIDSIG " + "B" * 40 + " rest\n", "")
        pytest.fail("unsafe archive must be rejected before any external extraction")
    monkeypatch.setattr(module.subprocess, "run", signature_only)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="archive"):
        module.prepare_profile(cache, tmp_path / "profile", tmp_path / "keyring")
    assert snapshot(tmp_path) == before and not (tmp_path / "profile").exists()


@pytest.mark.parametrize("payload", ["manifest-path", "manifest-command", "array-command", "array-path"])
def test_operator_hostile_payload_is_rejected_without_execution_or_escape(operator_fixture, payload):
    root, _, _ = operator_fixture
    generated = root / "src/vulkan/shaders/generated"
    header = generated / "operator_spv.h"
    target = root.parent / "escaped-marker"
    if payload.startswith("manifest"):
        (generated / "operator_spv.sha256").write_text(
            "../../escaped-marker=bad\n" if payload == "manifest-path" else f"source_sha256=__import__('pathlib').Path({str(target)!r}).touch()\n")
    else:
        text = f"__import__('pathlib').Path({str(target)!r}).touch()" if payload == "array-command" else "../../escaped-marker"
        header.write_text(header.read_text().replace("0x07230203U", text, 1))
        manifest(root)
    before = snapshot(root.parent)
    result = verify(operator_fixture)
    assert result.returncode != 0 and not target.exists()
    assert snapshot(root.parent) == before


def executable(path, code):
    path.write_text(f"#!{sys.executable}\n" + code)
    path.chmod(0o755)


def snapshot(root):
    return {str(p.relative_to(root)): (digest(p.read_bytes()), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def manifest(root):
    generated = root / "src/vulkan/shaders/generated"
    sources = root / "src/vulkan/shaders/glsl"
    (generated / "operator_spv.sha256").write_text(
        "source_sha256=" + digest(b"".join((sources / f"{n}.comp").read_bytes()
                                          for n in ("normalization", "classification")))
        + "\nheader_sha256=" + digest((generated / "operator_spv.h").read_bytes()) + "\n")


def assembly(name):
    count = 10 if name == "normalization" else 6
    offsets = [0, 4, 8, 12, 32, 36] if name == "normalization" else [0, 4, 16, 20]
    types = "%u %u %u %u %f %f" if name == "normalization" else "%u %u %u %i"
    return ("OpCapability Shader\nOpExecutionMode %main LocalSize 256 1 1\n"
            + "\n".join(f"OpDecorate %v{i} Binding {i}\nOpDecorate %v{i} DescriptorSet 0"
                        for i in range(count)) + "\n"
            + "\n".join(f"OpMemberDecorate %block {i} Offset {offset}"
                        for i, offset in enumerate(offsets)) + "\n"
            + "%u = OpTypeInt 32 0\n%i = OpTypeInt 32 1\n%f = OpTypeFloat 32\n"
            + f"%block = OpTypeStruct {types}\n"
            + "%ptr = OpTypePointer PushConstant %block\n%push = OpVariable %ptr PushConstant\n")


@pytest.fixture
def operator_fixture(tmp_path):
    root = tmp_path / "root"
    generated = root / "src/vulkan/shaders/generated"
    generated.mkdir(parents=True)
    source = root / "src/vulkan/shaders/glsl"
    source.mkdir()
    for name in ("normalization", "classification"):
        shutil.copy2(ROOT / f"src/vulkan/shaders/glsl/{name}.comp", source)
    host = "struct OperatorParams {" + HOST_FIELDS + "};\n"
    for name, count in [("normalization", 10), ("classification", 6)]:
        host += (f'create_extra("{name}", layout, shader, pipeline_layout, pipeline, '
                 f'vulkan_{name}_shader::kCode, vulkan_{name}_shader::kCodeSize, '
                 f'sizeof(OperatorParams), {count});\n')
    (root / "src/vulkan_compute.cpp").write_text(host)
    words = ", ".join(f"0x{x:08x}U" for x in struct.unpack("<7I", BINARY))
    (generated / "operator_spv.h").write_text("\n".join(
        f"namespace vulkan_{name}_shader {{ inline constexpr uint32_t kCode[] = {{{words}}}; }}"
        for name in ("normalization", "classification")))
    manifest(root)
    tools = root / "tools"
    tools.mkdir()
    for name in ("verify_operator_spv.py", "generate_operator_spv.py", "shader_toolchain.py"):
        original = ROOT / "tools" / name
        if original.exists():
            shutil.copy2(original, tools)
    # The legacy generator has fixed /tmp outputs. Redirect only this COPIED
    # generator so the RED reproduction stays inside the approved fixture.
    generator = tools / "generate_operator_spv.py"
    generator.write_text(generator.read_text().replace('pathlib.Path("/tmp")',
                                                     f'pathlib.Path({str(tmp_path)!r})'))
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "glslc", """import sys,pathlib
if '--version' in sys.argv: print('2026.3'); sys.exit(0)
out=pathlib.Path(sys.argv[sys.argv.index('-o')+1]); out.write_bytes(%r)
with pathlib.Path(%r).open('a') as f: f.write(str(out)+'\\n')
""" % (BINARY, str(tmp_path / "compiler-outputs")))
    executable(bin_dir / "spirv-val", "import sys\nsys.exit(0)\n")
    executable(bin_dir / "spirv-dis", "import sys\nprint(%r['normalization' if 'normalization' in sys.argv[-1] else 'classification'])\n"
               % {n: assembly(n) for n in ("normalization", "classification")})
    env = os.environ.copy()
    # Tests intentionally select fakes, never the user's configured profile.
    for key in ("VULKAN_SHADER_TOOLCHAIN_PROFILE", "VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256"):
        env.pop(key, None)
    env.update(PATH=str(bin_dir) + os.pathsep + env["PATH"], TMPDIR=str(tmp_path),
               PYTHONDONTWRITEBYTECODE="1")
    return root, bin_dir, env


def verify(fixture):
    root, _, env = fixture
    return subprocess.run([sys.executable, str(root / "tools/verify_operator_spv.py"), "--root", str(root)],
                          env=env, capture_output=True, text=True)


def test_legacy_copied_verifier_must_not_mutate_sources(operator_fixture):
    """The original regenerate-then-compare implementation violates purity."""
    root, _, env = operator_fixture
    tools = root / "tools"
    before = snapshot(root)
    result = subprocess.run([sys.executable, str(tools / "verify_operator_spv.py")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert snapshot(root) == before


def test_operator_pass_is_readonly(operator_fixture):
    root, _, _ = operator_fixture
    before = snapshot(root)
    result = verify(operator_fixture)
    assert result.returncode == 0, result.stderr
    assert snapshot(root) == before


@pytest.mark.parametrize("damage, diagnostic", [
    ("source", "source_sha256"), ("header", "header_sha256"),
    ("manifest", "manifest"), ("coherent_stale_binary", "embedded"),
    ("inventory", "inventory"), ("malformed", "SPIR-V"),
    ("compiler_failure", "compiler"), ("compiler_missing", "compiler"),
    ("bindings", "descriptor"), ("push", "push"), ("push_type", "push"),
    ("local", "local"), ("host", "host"), ("validator", "validation"),
])
def test_operator_failure_is_readonly(operator_fixture, damage, diagnostic):
    root, bin_dir, _ = operator_fixture
    generated = root / "src/vulkan/shaders/generated"
    header = generated / "operator_spv.h"
    if damage == "source":
        source = root / "src/vulkan/shaders/glsl/normalization.comp"
        source.write_text(source.read_text() + "\n// drift\n")
    elif damage == "header":
        header.write_text(header.read_text() + "\n// drift\n")
    elif damage == "manifest":
        (generated / "operator_spv.sha256").write_text("source_sha256=invalid\n")
    elif damage == "coherent_stale_binary":
        header.write_text(header.read_text().replace("0x00000002U", "0x00000003U"))
        manifest(root)
    elif damage == "inventory":
        header.write_text(header.read_text().replace("vulkan_classification_shader", "wrong"))
        manifest(root)
    elif damage == "host":
        (root / "src/vulkan_compute.cpp").write_text("struct OperatorParams {float mode;};")
    elif damage == "compiler_missing":
        # Explicit missing executable must not find the system compiler.
        executable(bin_dir / "glslc", "import sys\nprint('compiler missing',file=sys.stderr);sys.exit(127)\n")
    elif damage == "compiler_failure":
        executable(bin_dir / "glslc", "import sys\nprint('compiler failed',file=sys.stderr);sys.exit(2)\n")
    elif damage == "malformed":
        executable(bin_dir / "glslc", "import sys,pathlib\npathlib.Path(sys.argv[sys.argv.index('-o')+1]).write_bytes(b'bad')\n")
    elif damage == "validator":
        executable(bin_dir / "spirv-val", "import sys\nprint('invalid module',file=sys.stderr);sys.exit(1)\n")
    else:
        changes = {"bindings": ("Binding 5", "Binding 6"), "push": ("Offset 20", "Offset 24"),
                   "push_type": ("OpTypeInt 32 1", "OpTypeInt 64 1"),
                   "local": ("LocalSize 256 1 1", "LocalSize 128 1 1")}
        a, b = changes[damage]
        executable(bin_dir / "spirv-dis", "import sys\nprint(%r['normalization' if 'normalization' in sys.argv[-1] else 'classification'])\n"
                   % {n: assembly(n).replace(a, b) for n in ("normalization", "classification")})
    before = snapshot(root)
    result = verify(operator_fixture)
    assert result.returncode != 0
    assert diagnostic.lower() in (result.stdout + result.stderr).lower()
    assert snapshot(root) == before


def test_operator_concurrent_verification_has_unique_temporary_outputs(operator_fixture):
    root, _, env = operator_fixture
    before = snapshot(root)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: verify(operator_fixture), range(2)))
    assert all(r.returncode == 0 for r in results), [r.stderr for r in results]
    outputs = (Path(env["TMPDIR"]) / "compiler-outputs").read_text().splitlines()
    assert len(outputs) == len(set(outputs)) == 4
    assert len({str(Path(p).parent) for p in outputs}) == 2
    assert snapshot(root) == before


def test_coherent_header_with_missing_word_delimiter_is_rejected(operator_fixture):
    root, _, _ = operator_fixture
    header = root / "src/vulkan/shaders/generated/operator_spv.h"
    header.write_text(header.read_text().replace("0x07230203U,", "0x07230203U", 1))
    manifest(root)
    before = snapshot(root)
    result = verify(operator_fixture)
    assert result.returncode != 0 and "array" in result.stderr
    assert snapshot(root) == before


def selector():
    spec = importlib.util.find_spec("tools.shader_toolchain")
    assert spec is not None, "explicit shader toolchain selector is not implemented"
    return importlib.import_module("tools.shader_toolchain")


def test_missing_selected_profile_never_falls_back_to_system(tmp_path):
    with pytest.raises((ValueError, OSError), match="profile"):
        selector().selected_environment(tmp_path / "missing.json")


def test_profile_with_unapproved_executable_and_libraries_is_rejected(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "prefix": "/usr", "packages": {}}))
    with pytest.raises(ValueError, match="profile|package|identity"):
        selector().selected_environment(profile)


def test_selected_profile_hash_is_enforced_before_command(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}\n")
    with pytest.raises(ValueError, match="profile.*hash"):
        selector().selected_environment(profile, expected_sha256="0" * 64)


def test_unsigned_cached_package_is_not_extracted_or_adopted(tmp_path, monkeypatch):
    module = selector()
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / "fixture-package"
    archive.write_bytes(b"untrusted archive")
    (cache / "fixture-package.sig").write_bytes(b"bad signature")
    monkeypatch.setattr(module, "APPROVED_PACKAGES", {
        archive.name: {"sha256": digest(archive.read_bytes()), "signer": "B" * 40}})
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1, "", "invalid signature"))
    target = tmp_path / "compiler"
    with pytest.raises(ValueError, match="signature"):
        module.prepare_profile(cache, target, tmp_path / "keyring")
    assert not target.exists()


def test_cached_package_hash_mismatch_is_not_adopted(tmp_path):
    target = tmp_path / "compiler"
    with pytest.raises(ValueError, match="hash.*missing"):
        selector().prepare_profile(tmp_path, target, tmp_path / "keyring")
    assert not target.exists()


def test_explicit_missing_compiler_never_uses_system(operator_fixture):
    root, _, env = operator_fixture
    before = snapshot(root)
    result = subprocess.run([sys.executable, str(root / "tools/verify_operator_spv.py"),
                             "--glslc", str(root / "missing-glslc")],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "compiler" in result.stderr
    assert snapshot(root) == before


def test_guarded_operator_generation_refuses_unreviewed_bytes_before_write(operator_fixture):
    root, _, env = operator_fixture
    before = snapshot(root)
    result = subprocess.run([sys.executable, str(root / "tools/generate_operator_spv.py"),
                             "--expected-header-sha256", "0" * 64],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "outputs not written" in result.stderr
    assert snapshot(root) == before


def test_cmake_selected_profile_is_validated_before_build(tmp_path):
    result = subprocess.run(["cmake", "-S", str(ROOT), "-B", str(tmp_path / "build"),
                             "-DBUILD_VULKAN_PROBE=ON", "-DPython3_EXECUTABLE=" + sys.executable,
                             "-DVULKAN_SHADER_TOOLCHAIN_PROFILE=" + str(tmp_path / "missing.json")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "profile" in (result.stdout + result.stderr).lower()


@pytest.fixture
def known_profile(tmp_path, monkeypatch):
    module = selector()
    prefix = tmp_path / "compiler"
    (prefix / "usr/bin").mkdir(parents=True)
    (prefix / "usr/lib").mkdir()
    executable(prefix / "usr/bin/glslc", """import sys,os,json
if '--version' in sys.argv: print('2026.3\\n1:1.4.357.0\\n1:1.4.357.0\\n\\nTarget: SPIR-V 1.0');sys.exit(0)
print(json.dumps({'library_path':os.environ.get('LD_LIBRARY_PATH'), 'argv':sys.argv[1:]}))
""")
    lib = prefix / "usr/lib/libglslang.so.16"
    lib.write_bytes(b"reviewed glslang library")
    approved = {"usr/bin/glslc": digest((prefix / "usr/bin/glslc").read_bytes()),
                "usr/lib/libglslang.so.16": digest(lib.read_bytes())}
    packages = {"signed-test-package": {"sha256": "a" * 64, "signer": "B" * 40}}
    monkeypatch.setattr(module, "APPROVED_FILES", approved)
    monkeypatch.setattr(module, "APPROVED_PACKAGES", packages)
    ldd = tmp_path / "fake-ldd"
    executable(ldd, f"print('libglslang.so.16 => {lib} (0x1234)')\n")
    monkeypatch.setattr(module, "LDD", str(ldd))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "schema_version": 1, "prefix": str(prefix), "packages": packages,
        "files": approved, "version": "2026.3\n1:1.4.357.0\n1:1.4.357.0\n\nTarget: SPIR-V 1.0\n",
        "linked_libraries": {"libglslang.so.16": {"path": str(lib), "sha256": approved["usr/lib/libglslang.so.16"]}},
        "flags": ["-Os"], "target_environment": "default Vulkan; SPIR-V 1.0",
    }))
    return module, profile, prefix


def test_known_profile_provenance_and_selection_do_not_pollute_python_libraries(known_profile):
    module, profile, prefix = known_profile
    original = os.environ.copy()
    env = module.selected_environment(profile)
    assert os.environ == original
    assert env.get("LD_LIBRARY_PATH") == original.get("LD_LIBRARY_PATH")
    assert env["VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256"] == digest(profile.read_bytes())
    provenance = module.profile_provenance(profile)
    assert provenance["profile_sha256"] == digest(profile.read_bytes())
    assert provenance["compiler"]["path"] == str(prefix / "usr/bin/glslc")
    assert provenance["linked_libraries"]["libglslang.so.16"]["path"] == str(prefix / "usr/lib/libglslang.so.16")


def test_old_executable_with_changed_library_is_rejected(known_profile):
    module, profile, prefix = known_profile
    (prefix / "usr/lib/libglslang.so.16").write_bytes(b"new glslang")
    with pytest.raises(ValueError, match="identity|hash"):
        module.selected_environment(profile)


def test_changed_host_linked_library_identity_is_rejected(known_profile):
    module, profile, _ = known_profile
    data = json.loads(profile.read_text())
    data["linked_libraries"]["libglslang.so.16"]["path"] = "/usr/lib/libglslang.so.16"
    profile.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="linked|identity"):
        module.selected_environment(profile)


def test_missing_profile_library_never_uses_system_copy(known_profile):
    module, profile, prefix = known_profile
    (prefix / "usr/lib/libglslang.so.16").unlink()
    with pytest.raises((ValueError, OSError), match="identity|library|file"):
        module.selected_environment(profile)


def test_selected_compiler_receives_scoped_library_environment(known_profile, capfd):
    module, profile, prefix = known_profile
    original = os.environ.copy()
    assert module.invoke_compiler(["-Os", "input.comp"], profile) == 0
    output = json.loads(capfd.readouterr().out)
    assert output == {"library_path": str(prefix / "usr/lib"), "argv": ["-Os", "input.comp"]}
    assert os.environ == original


def test_runner_rejects_missing_profile_before_any_device_probe(tmp_path, monkeypatch):
    from tools import run_vulkan_qualification as runner
    monkeypatch.setenv("VULKAN_SHADER_TOOLCHAIN_PROFILE", str(tmp_path / "missing.json"))
    monkeypatch.setenv("VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256", "0" * 64)
    def unexpected_probe(*args, **kwargs):
        pytest.fail("selected profile must be validated before any GPU/device probe")
    monkeypatch.setattr(runner, "device_available", unexpected_probe)
    with pytest.raises(ValueError, match="profile"):
        runner.run_qualification(tmp_path / "report.json")


def test_runner_propagates_known_profile_to_all_subprocesses(known_profile, monkeypatch, tmp_path):
    from tools import run_vulkan_qualification as runner
    module, profile, _ = known_profile
    monkeypatch.setenv("VULKAN_SHADER_TOOLCHAIN_PROFILE", str(profile))
    monkeypatch.setenv("VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256", digest(profile.read_bytes()))
    calls = []
    def fake_run(command, cwd, env=None, **kwargs):
        calls.append((command, env))
        return {"exit_code": 0, "stdout": "fixture CPU command", "stderr": ""}
    monkeypatch.setattr(runner, "run_command", fake_run)
    monkeypatch.setattr(runner, "device_available", lambda *args, **kwargs: {"status": "available", "reason": "fixture"})
    before = os.environ.copy()
    report = runner.run_qualification(tmp_path / "report.json")
    assert report["shader_toolchain"]["profile_sha256"] == digest(profile.read_bytes())
    assert json.loads((tmp_path / ".vulkan-qualification-artifacts/shader-toolchain.json").read_text())["compiler"]["version"].startswith("2026.3\n")
    assert calls and all(env and env["VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256"] == digest(profile.read_bytes()) for _, env in calls)
    assert all(env["PATH"].split(os.pathsep)[0] == str(module.WRAPPERS) for _, env in calls)
    assert all(env.get("LD_LIBRARY_PATH") == before.get("LD_LIBRARY_PATH") for _, env in calls)
    assert os.environ == before


def test_runner_cli_selects_profile_without_python_loader_override(known_profile, monkeypatch, tmp_path):
    from tools import run_vulkan_qualification as runner
    _, profile, _ = known_profile
    original = os.environ.copy()
    monkeypatch.setattr(sys, "argv", [str(ROOT / "tools/run_vulkan_qualification.py"),
                                     "--output", str(tmp_path / "report.json"),
                                     "--shader-toolchain-profile", str(profile)])
    calls = []
    def reenter(command, env):
        calls.append((command, env))
        return 7
    monkeypatch.setattr(runner.subprocess, "call", reenter)
    assert runner.main() == 7
    command, env = calls[0]
    assert "--shader-toolchain-profile" not in command
    assert env["VULKAN_SHADER_TOOLCHAIN_PROFILE"] == str(profile)
    assert env["VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256"] == digest(profile.read_bytes())
    assert env.get("LD_LIBRARY_PATH") == original.get("LD_LIBRARY_PATH")
    assert os.environ == original


def test_cmake_pooling_integrity_does_not_generate_products(tmp_path):
    """Execute the repository's actual target in a copied minimal CMake root."""
    root = tmp_path / "root"
    shutil.copytree(ROOT / "tools", root / "tools")
    shutil.copytree(ROOT / "src", root / "src")
    shutil.copy2(ROOT / "CMakeLists.txt", root / "CMakeLists.txt")
    shutil.copytree(ROOT / "python", root / "python")
    shutil.copytree(ROOT / "tests", root / "tests")
    # Its external compiler cannot change fixture files; the original target's
    # preceding generator writes a visible marker, unlike the corrected target.
    (root / "tools/generate_pooling_spv.py").write_text("from pathlib import Path\nPath(__file__).with_name('unexpected-generation').write_text('mutated')\n")
    (root / "tools/verify_pooling_spv.py").write_text("print('fixture verification')\n")
    build = tmp_path / "build"
    configure = subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                                "-DBUILD_PYTHON_EXTENSION=ON", "-DPython3_EXECUTABLE=" + sys.executable,
                                "-DCMAKE_PREFIX_PATH=" + str(ROOT / ".venv/lib/python3.12/site-packages/torch/share/cmake")],
                               capture_output=True, text=True)
    assert configure.returncode == 0, configure.stderr
    result = subprocess.run(["cmake", "--build", str(build), "--target", "vulkan_pooling_shader_integrity"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not (root / "tools/unexpected-generation").exists()
    generation = subprocess.run(["cmake", "--build", str(build), "--target", "vulkan_pooling_shader_generate"],
                                capture_output=True, text=True)
    assert generation.returncode == 0, generation.stderr
    assert (root / "tools/unexpected-generation").read_text() == "mutated"


@pytest.mark.parametrize("action", ["ordinary-build", "explicit-unselected"])
def test_cmake_operator_regeneration_is_opt_in_and_requires_profile(tmp_path, action):
    root = tmp_path / "root"
    for directory in ("tools", "src", "python", "tests"):
        shutil.copytree(ROOT / directory, root / directory)
    shutil.copy2(ROOT / "CMakeLists.txt", root / "CMakeLists.txt")
    # All13 verifiers are read-only fixture stubs; no GPU/extension code runs.
    for path in (root / "tools").glob("verify_*_spv.py"):
        path.write_text("print('fixture integrity')\n")
    for path in (root / "tools").glob("generate_*_spv.py"):
        path.write_text("from pathlib import Path\nPath(__file__).with_name('unexpected-generation').write_text('mutated')\n")
    for path in (root / "src").rglob("*.cpp"):
        path.write_text("// CPU-only empty compilation fixture\n")
    build = tmp_path / "build"
    result = subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                             "-DBUILD_PYTHON_EXTENSION=ON", "-DVULKAN_SHADER_TOOLCHAIN_PROFILE=",
                             "-DPython3_EXECUTABLE=" + sys.executable,
                             "-DCMAKE_PREFIX_PATH=" + str(ROOT / ".venv/lib/python3.12/site-packages/torch/share/cmake")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    before = snapshot(root)
    target = "vulkan_platform" if action == "ordinary-build" else "vulkan_operator_shader_generate"
    result = subprocess.run(["cmake", "--build", str(build), "--target", target, "-j2"],
                            capture_output=True, text=True)
    if action == "ordinary-build":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "profile" in (result.stdout + result.stderr).lower()
    assert not (root / "tools/unexpected-generation").exists()
    assert snapshot(root) == before


def test_operator_publication_retains_recovery_backups_when_rollback_fails(tmp_path, monkeypatch):
    from tools import generate_operator_spv as generator
    header = tmp_path / "operator_spv.h"
    manifest_path = tmp_path / "operator_spv.sha256"
    header.write_bytes(b"old header")
    manifest_path.write_bytes(b"old manifest")
    replace = os.replace
    def fail_publication_and_recovery(src, dst):
        if Path(dst) == manifest_path or str(src).endswith(".backup"):
            raise OSError("injected persistent filesystem failure")
        return replace(src, dst)
    monkeypatch.setattr(os, "replace", fail_publication_and_recovery)
    with pytest.raises(RuntimeError, match="rollback failed.*retained backups"):
        generator.publish_products(tmp_path, b"new header", b"new manifest")
    stage, = tmp_path.glob(".operator-stage-*")
    assert (stage / "operator_spv.h.backup").read_bytes() == b"old header"
    assert (stage / "operator_spv.sha256.backup").read_bytes() == b"old manifest"
    assert manifest_path.read_bytes() == b"old manifest"


def test_safe_archive_link_chains_are_confined_and_order_independent(tmp_path):
    module = selector()
    members = [tar_member("usr/link0", tarfile.SYMTYPE, "link1"),
               tar_member("usr/link1", tarfile.SYMTYPE, "data"),
               tar_member("usr/hard0", tarfile.LNKTYPE, "usr/hard1"),
               tar_member("usr/hard1", tarfile.LNKTYPE, "usr/link0"), tar_member("usr/data")]
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for member in members:
            archive.addfile(member, io.BytesIO(b"data") if member.isfile() else None)
    data.seek(0)
    root = tmp_path / "profile"
    with tarfile.open(fileobj=data, mode="r:") as archive:
        inventory = archive.getmembers()
        module.preflight_archives([inventory])
        root.mkdir()
        module._extract_preflighted([(archive, inventory)], root)
    assert (root / "usr/link0").read_bytes() == (root / "usr/hard0").read_bytes() == b"data"
    assert (root / "usr/link0").resolve().is_relative_to(root)
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("case", ["symlink-nul", "hardlink-nul", "cross-package-parent", "cross-package-chain"])
def test_archive_cross_package_and_nul_links_reject_before_write(tmp_path, case):
    module = selector()
    manifests = {
        "symlink-nul": [[tar_member("usr/link", tarfile.SYMTYPE, "data\x00../outside")]],
        "hardlink-nul": [[tar_member("usr/link", tarfile.LNKTYPE, "usr/data\x00../outside")]],
        "cross-package-parent": [[tar_member("usr/a", tarfile.SYMTYPE, ".")], [tar_member("usr/a/data")]],
        "cross-package-chain": [[tar_member("usr/a", tarfile.SYMTYPE, "b")],
                                [tar_member("usr/b", tarfile.SYMTYPE, "../../outside")]],
    }[case]
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="archive"):
        module.preflight_archives(manifests)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("payload", ["../escaped", "/escaped", "usr/nul\x00escaped"])
def test_pax_extended_member_paths_are_preflighted_without_extraction(tmp_path, payload):
    module = selector()
    data = io.BytesIO()
    member = tar_member("usr/safe")
    member.pax_headers = {"path": payload}
    with tarfile.open(fileobj=data, mode="w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(member, io.BytesIO(b"data"))
    data.seek(0)
    before = snapshot(tmp_path)
    with tarfile.open(fileobj=data, mode="r:") as archive:
        with pytest.raises(ValueError, match="archive"):
            module.preflight_archives([archive.getmembers()])
    assert snapshot(tmp_path) == before


def test_authenticated_archives_and_cmake_operator_generation_are_protected(tmp_path):
    """Real signed packages and compiler; every write confined to copied roots."""
    module = selector()
    profile = module.prepare_profile(Path("/var/cache/pacman/pkg"), tmp_path / "profile",
                                     Path("/etc/pacman.d/gnupg/pubring.gpg"))
    provenance = module.profile_provenance(profile)
    assert provenance["compiler"]["version"].startswith("2026.3\n")
    root = tmp_path / "root"
    for directory in ("tools", "src", "python", "tests"):
        shutil.copytree(ROOT / directory, root / directory)
    shutil.copy2(ROOT / "CMakeLists.txt", root / "CMakeLists.txt")
    build = tmp_path / "build"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = []
    def run(argv):
        result = subprocess.run(argv, env=env, capture_output=True, text=True)
        commands.append({"argv": argv, "exit": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        (tmp_path / "protected-cmake-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        return result
    result = run(["cmake", "-S", str(root), "-B", str(build),
                             "-DBUILD_PYTHON_EXTENSION=ON", "-DVULKAN_SHADER_TOOLCHAIN_PROFILE=" + str(profile),
                             "-DPython3_EXECUTABLE=" + sys.executable,
                             "-DCMAKE_PREFIX_PATH=" + str(ROOT / ".venv/lib/python3.12/site-packages/torch/share/cmake")])
    assert result.returncode == 0, result.stderr
    generated = root / "src/vulkan/shaders/generated"
    before_bytes = {p.name: p.read_bytes() for p in generated.iterdir() if p.is_file()}
    result = run(["cmake", "--build", str(build), "--target", "vulkan_operator_shader_generate"])
    assert result.returncode == 0, result.stderr
    assert {p.name: p.read_bytes() for p in generated.iterdir() if p.is_file()} == before_bytes
    source = root / "src/vulkan/shaders/glsl/classification.comp"
    before = snapshot(generated)
    source.write_text(source.read_text() + "\n// copied source-only manifest drift\n")
    result = run(["cmake", "--build", str(build), "--target", "vulkan_operator_shader_generate"])
    assert result.returncode != 0 and "expected manifest hash mismatch" in (result.stdout + result.stderr)
    assert snapshot(generated) == before
    # Semantic perturbation in OWNED copy produces genuinely different bytes.
    original_source = source.read_text()
    damaged_source = original_source.replace("p.classes", "(p.classes + 1u)")
    assert damaged_source != original_source
    source.write_text(damaged_source)
    for target in ("vulkan_operator_shader_integrity", "vulkan_operator_shader_generate", "vulkan_platform"):
        before = snapshot(generated)
        result = run(["cmake", "--build", str(build), "--target", target])
        assert result.returncode != 0, result.stdout
        assert "mismatch" in (result.stdout + result.stderr)
        assert snapshot(generated) == before
    library = tmp_path / "profile/usr/lib/libglslang.so.16.4.0"
    library.write_bytes(b"changed library fixture")
    before = snapshot(generated)
    result = run(["cmake", "--build", str(build), "--target", "vulkan_operator_shader_generate"])
    assert result.returncode != 0 and "identity mismatch" in (result.stdout + result.stderr)
    assert snapshot(generated) == before
