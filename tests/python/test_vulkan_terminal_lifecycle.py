"""Opt-in ordinary-API process-exit acceptance (one explicitly cleared gate).

No backend is imported in the pytest process. A gate must supply absolute paths
in PYTORCH_VULKAN_TERMINAL_GATE_CONFIG and a NEW per-selector directory in
PYTORCH_VULKAN_TERMINAL_RECEIPT_ROOT. The JSON config contains package_root,
extension and package_init {path, sha256}, reviewed_build_provenance {path, sha256,
schema}, repository_root, rnn_oracle_source, execution_mode, device, driver and
validation. See docs/vulkan-terminal-lifecycle.md for the strict v1 manifest.
Approval is external: schema/hash validation neither grants approval nor proves
compilation. Rebuilding or executing still requires separate clearance.
Finalizer-return full-quiescence relies on accepted CPU operation-tail coverage:
these GPU scripts synchronize before readback and cannot independently establish
the full pre-drain callback/reservation/owner postcondition at operator return.
Select exactly one test per invocation; never retry a failed gate automatically.
"""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys

import pytest


if os.environ.get("PYTORCH_VULKAN_TERMINAL_GPU") != "1":
    pytest.skip("separate named terminal GPU clearance required", allow_module_level=True)

pytestmark = pytest.mark.vulkan_terminal_gpu
ROOT = Path(__file__).resolve().parents[2]


def _metadata_file(record):
    assert isinstance(record, dict) and record.keys() == {"path", "sha256"}
    path = Path(record["path"])
    assert path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), path
    assert isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    return path


def _metadata_members(records, required):
    assert isinstance(records, dict) and set(records) == {str(p) for p in required}
    for path, digest in records.items():
        _metadata_file(dict(path=path, sha256=digest))


def _metadata_profile(profile_record, provenance_record, toolchain):
    # Read only literal toolchain constants, never execute it, glslc or ldd.
    constants = {}
    for node in ast.parse(toolchain.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"APPROVED_PACKAGES", "APPROVED_FILES", "FLAGS", "TARGET", "VERSION", "SIGNER"}:
                # APPROVED_PACKAGES references the preceding SIGNER literal.
                value = node.value
                if name == "APPROVED_PACKAGES":
                    class SignerLiteral(ast.NodeTransformer):
                        def visit_Name(self, item):
                            assert item.id == "SIGNER"
                            return ast.copy_location(ast.Constant(constants["SIGNER"]), item)
                    value = SignerLiteral().visit(value)
                constants[name] = ast.literal_eval(value)
    profile_path = _metadata_file(profile_record)
    data = json.loads(profile_path.read_bytes())
    assert data.keys() == {"schema_version", "prefix", "packages", "files", "version",
                           "linked_libraries", "flags", "target_environment"}
    assert type(data["schema_version"]) is int and data["schema_version"] == 1
    for field, name in (("packages", "APPROVED_PACKAGES"), ("files", "APPROVED_FILES"),
                        ("version", "VERSION"), ("flags", "FLAGS"), ("target_environment", "TARGET")):
        assert data[field] == constants[name], field
    prefix = Path(data["prefix"])
    assert prefix.is_absolute() and prefix.resolve(strict=True) == prefix
    for name, digest in data["files"].items():
        file = (prefix / name).resolve(strict=True)
        assert file.is_relative_to(prefix)
        _metadata_file(dict(path=str(file), sha256=digest))
    assert isinstance(data["linked_libraries"], dict) and data["linked_libraries"]
    for record in data["linked_libraries"].values():
        _metadata_file(record)
    for name, digest in data["files"].items():
        if name.startswith("usr/lib/"):
            assert data["linked_libraries"][Path(name).name] == dict(
                path=str((prefix / name).resolve(strict=True)), sha256=digest)
    provenance = json.loads(_metadata_file(provenance_record).read_bytes())
    assert provenance == dict(profile=str(profile_path), profile_sha256=profile_record["sha256"],
        compiler=dict(path=str(prefix / "usr/bin/glslc"), sha256=data["files"]["usr/bin/glslc"],
                      version=data["version"]), linked_libraries=data["linked_libraries"],
        packages=data["packages"], flags=data["flags"], target_environment=data["target_environment"])


def _metadata_compiled_sources(build, package_root, extension):
    """Select only recorded extension-link objects and linked archive members.

    Configured commands for unbuilt/unrelated native targets are not compilation
    evidence. Every selected object still needs its own recorded hash and unique
    compile-source join; archive-member hashes must join those exact objects.
    """
    def link_text(target):
        path = package_root / "CMakeFiles" / (target + ".dir") / "link.txt"
        text = build["links"][str(path.relative_to(ROOT))]
        assert path.read_text() == text
        return text

    tokens = shlex.split(link_text("pytorch_vulkan_python"))
    assert tokens.count("-o") == 1
    assert package_root / tokens[tokens.index("-o") + 1] == extension
    objects = {package_root / token for token in tokens if token.endswith(".o")}
    archives = {package_root / token for token in tokens if token.endswith(".a")}
    assert archives
    for archive_path in archives:
        assert archive_path.parent == package_root
        name = archive_path.name
        assert name.startswith("lib") and name.endswith(".a")
        archive = build["archives"][name]
        _metadata_file(dict(path=str(archive_path), sha256=archive["sha256"]))
        archive_command = shlex.split(link_text(name[3:-2]).splitlines()[0])
        assert Path(archive_command[0]).name == "ar" and archive_command[1] == "qc"
        assert package_root / archive_command[2] == archive_path
        archive_objects = {package_root / token for token in archive_command[3:]}
        assert archive_objects and all(p.suffix == ".o" for p in archive_objects)
        assert len({p.name for p in archive_objects}) == len(archive_objects)
        assert set(archive["members"]) == {p.name for p in archive_objects}
        for obj in archive_objects:
            digest = build["objects"][str(obj.relative_to(ROOT))]
            assert digest == archive["members"][obj.name]
        objects.update(archive_objects)
    compiled_sources = set()
    for obj in objects:
        assert obj.is_relative_to(package_root)
        _metadata_file(dict(path=str(obj), sha256=build["objects"][str(obj.relative_to(ROOT))]))
        commands = [c for c in build["commands"]
                    if Path(c["directory"]) / c["output"] == obj]
        assert len(commands) == 1
        command = commands[0]
        assert Path(command["directory"]) == package_root
        source = Path(command["file"])
        assert source.is_relative_to(ROOT / "src") and source.resolve(strict=True) == source
        compiled_sources.add(source)
    return compiled_sources


def _validate_gate_provenance(config):
    """Validate externally approved metadata, not approval or compilation itself.

    The combined review supplies source membership. The accepted build inventory
    supplies the production source/object/archive/ELF evidence reference. Current
    Python gate source is deliberately separate from already compiled bytes.
    """
    assert Path(config["repository_root"]) == ROOT
    package_root = Path(config["package_root"])
    assert package_root.is_absolute() and package_root.resolve(strict=True) == package_root
    assert Path(config["rnn_oracle_source"]) == ROOT / "tests/python/test_vulkan_fused_rnn.py"
    approved = config["reviewed_build_provenance"]
    assert approved.keys() == {"path", "sha256", "schema"}
    assert approved["schema"] == "vulkan-terminal-build-provenance-v1"
    manifest_path = _metadata_file({k: approved[k] for k in ("path", "sha256")})
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest.keys() == {"schema", "repository_root", "package_root", "extension", "package_init",
        "source_review", "build_inventory", "runtime_sources", "gate_sources", "shaders",
        "shader_profile", "shader_provenance", "build_cache"}
    for key in ("repository_root", "package_root", "extension", "package_init"):
        assert manifest[key] == config[key], key
    assert manifest["schema"] == approved["schema"]
    extension = _metadata_file(manifest["extension"])
    assert extension.parent == package_root / "pytorch_vulkan" and extension.name.startswith("_C")
    init = _metadata_file(manifest["package_init"])
    assert init == ROOT / "python/pytorch_vulkan/__init__.py"
    review = json.loads(_metadata_file(manifest["source_review"]).read_bytes())
    reviewed_runtime = {ROOT / name for name in review if name.startswith("src/")}
    minimum = {ROOT / ("src/" + name) for name in (
        "python_module.cpp", "vulkan_lifecycle.cpp", "vulkan_python_lifecycle.cpp",
        "vulkan_platform.cpp", "vulkan_execution.cpp", "vulkan_async_transfer.cpp", "vulkan_compute.cpp")}
    # The unchanged async input is compiled evidence, not a changed lifecycle
    # member of the immutable combined-review source subset.
    assert minimum - {ROOT / "src/vulkan_async_transfer.cpp"} <= reviewed_runtime
    inventory = json.loads(_metadata_file(manifest["build_inventory"]).read_bytes())
    build = inventory[str(package_root.relative_to(ROOT))]
    assert build.keys() == {"commands", "objects", "archives", "links", "binaries"}
    assert build["binaries"][str(extension.relative_to(ROOT))]["sha256"] == manifest["extension"]["sha256"]
    compiled_sources = _metadata_compiled_sources(build, package_root, extension)
    assert minimum <= compiled_sources
    runtime = reviewed_runtime | compiled_sources | {ROOT / "CMakeLists.txt"}
    _metadata_members(manifest["runtime_sources"], runtime)
    # Native CLI/tests/docs changed by Task6 are NOT claimed compiled by Task5.
    for path in reviewed_runtime | {ROOT / "CMakeLists.txt"}:
        assert manifest["runtime_sources"][str(path)] == review[str(path.relative_to(ROOT))]["sha256"]
    cmake = (ROOT / "CMakeLists.txt").read_text()
    package_sources = {ROOT / name for name in re.findall(
        r"python/pytorch_vulkan/[A-Za-z0-9_]+\.py", cmake)}
    assert init in package_sources
    gate_sources = {Path(__file__).resolve(), init, Path(config["rnn_oracle_source"]),
                    ROOT / "tools/qualification_process_supervisor.py"}
    gate_sources.update(package_sources)
    _metadata_members(manifest["gate_sources"], gate_sources)
    assert manifest["gate_sources"][str(init)] == manifest["package_init"]["sha256"]
    for source in package_sources:
        imported = package_root / "pytorch_vulkan" / source.name
        resolved = imported.resolve(strict=True)
        assert resolved == (source if imported.is_symlink() else imported)
        assert hashlib.sha256(imported.read_bytes()).hexdigest() == manifest["gate_sources"][str(source)]
    # Exact checked-in shader inputs/artifacts and generator/verifier dependencies
    # named by the reviewed build rules, including headers embedded by compute.
    shader_names = set(re.findall(r"src/vulkan/shaders/(?:generated|glsl)/[A-Za-z0-9_.]+", cmake))
    shader_names.update(re.findall(r"tools/(?:generate|verify)_[A-Za-z0-9_]+_spv\.py", cmake))
    includes = re.findall(r'"(vulkan/shaders/generated/[A-Za-z0-9_]+_spv\.h)"',
                          (ROOT / "src/vulkan_compute.cpp").read_text())
    shader_names.update("src/" + name for name in includes)
    assert any(name.endswith("_spv.h") for name in shader_names)
    assert any(name.endswith(".comp") for name in shader_names)
    shader_names.add("tools/shader_toolchain.py")
    _metadata_members(manifest["shaders"], {ROOT / name for name in shader_names})
    assert _metadata_file(manifest["shader_provenance"]) == package_root / "shader-toolchain.json"
    cache = _metadata_file(manifest["build_cache"])
    assert cache == package_root / "CMakeCache.txt"
    selected = [line.split("=", 1)[1] for line in cache.read_text().splitlines()
                if line.startswith("VULKAN_SHADER_TOOLCHAIN_PROFILE:FILEPATH=")]
    assert selected == [manifest["shader_profile"]["path"]]
    _metadata_profile(manifest["shader_profile"], manifest["shader_provenance"], ROOT / "tools/shader_toolchain.py")
    return manifest


def _write_receipt(receipt_root, payload):
    with (receipt_root / "result.json").open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    for directory in (receipt_root, receipt_root.parent):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def run_terminal_child(source: str, expected_status: int) -> None:
    """Supervise exactly one child; durably preserve raw failure before asserting."""
    if os.environ.get("PYTORCH_VULKAN_TERMINAL_GPU") != "1":
        raise RuntimeError("terminal GPU gate not enabled")
    # Do not resolve sys.executable: the venv launcher may be a symlink.
    assert Path(sys.executable).absolute() == ROOT / ".venv/bin/python"
    assert sys.flags.optimize == 0, "child assertions must remain enabled"
    config_path = Path(os.environ["PYTORCH_VULKAN_TERMINAL_GATE_CONFIG"])
    receipt_root = Path(os.environ["PYTORCH_VULKAN_TERMINAL_RECEIPT_ROOT"])
    assert config_path.is_absolute() and receipt_root.is_absolute()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    provenance = _validate_gate_provenance(config)  # includes supervisor SHA BEFORE exec_module
    assert config["execution_mode"] in ("async", "sync")
    for key in ("device", "driver", "validation"):
        assert isinstance(config[key], str) and config[key]
    environment = os.environ.copy()
    # No async/validation/driver overrides: retain the approved gate environment.
    configured_mode = "sync" if environment.get("PYTORCH_VULKAN_ASYNC_EXECUTION") == "0" else "async"
    assert config["execution_mode"] == configured_mode
    assert environment.get("PYTHONOPTIMIZE", "0") in ("", "0")
    receipt_root.mkdir()  # create-only, explicit per gate; parent must already exist
    spec = importlib.util.spec_from_file_location(
        "_terminal_qualification_supervisor", ROOT / "tools/qualification_process_supervisor.py"
    )
    supervisor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = supervisor
    spec.loader.exec_module(supervisor)
    result = certificate = error = None
    try:
        with supervisor.cli_cancellation_scope():
            result, certificate = supervisor.run_command_core(
                [sys.executable, "-u", "-c", source], ROOT,
                env=environment, timeout_seconds=300,
            )
    except BaseException as exc:
        error = exc
        result = getattr(exc, "partial_result", result)
        certificate = getattr(exc, "certificate", certificate)
    _write_receipt(receipt_root, dict(
        raw=result,
        certificate=supervisor.certificate_record(certificate) if certificate else None,
        exception=repr(error) if error else None,
        child_source=source,
        child_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        acceptance_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config=config, config_path=str(config_path), reviewed_build_provenance=provenance,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        expected_status=expected_status,
    ))
    if error is not None:
        raise error
    supervisor.require_complete(result, certificate)
    assert certificate.reason == "leader-exit", certificate
    assert result["exit_code"] == expected_status, result
    assert result["stdout"].splitlines().count("TERMINAL_LIFECYCLE_OK") == 1, result
    assert result["stderr"].splitlines().count("TERMINAL_BINARY_BINDING_OK") == 1, result
    assert not any(line.startswith("Vulkan validation:")
                   for line in result["stderr"].splitlines()), result
    assert "Vulkan device lost" not in result["stderr"], result
    # CPython normally swallows atexit/__del__ exceptions. The success marker is
    # printed only after every required callback has checked its own completion.
    assert "Exception ignored" not in result["stderr"], result
    if expected_status == 1:
        assert result["stderr"].rstrip().endswith(
            "RuntimeError: TERMINAL_EXPECTED_ORDINARY_EXCEPTION"
        ), result
    else:
        assert "Traceback (most recent call last)" not in result["stderr"], result


CHILD_COMMON = r'''
import atexit
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

assert os.environ.get("PYTORCH_VULKAN_TERMINAL_GPU") == "1"
config = json.loads(Path(os.environ["PYTORCH_VULKAN_TERMINAL_GATE_CONFIG"]).read_text())
sys.path.insert(0, config["package_root"])

def validate_package_origin(config, package_file, search_paths, extension_file):
    # Resolve the containing directory, NOT the final __init__.py source symlink.
    root = Path(config["package_root"]).resolve(strict=True)
    package = Path(package_file)
    assert package.is_absolute() and package.name == '__init__.py'
    assert package.parent.resolve(strict=True) == root / 'pytorch_vulkan'
    assert list(search_paths) == [str(root / 'pytorch_vulkan')]
    source = Path(config["package_init"]["path"])
    assert source == Path(config["repository_root"]) / 'python/pytorch_vulkan/__init__.py'
    assert source.resolve(strict=True) == source
    resolved = package.resolve(strict=True)
    # CMake CREATE_LINK COPY_ON_ERROR: an actual copy may resolve to itself, but
    # a symlink must target the approved repository source, not an identical decoy.
    assert resolved == (source if package.is_symlink() else package)
    expected_hash = config["package_init"]["sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == expected_hash
    loaded = Path(extension_file).resolve(strict=True)
    expected = Path(config["extension"]["path"]).resolve(strict=True)
    assert loaded == expected, (loaded, expected)
    assert hashlib.sha256(loaded.read_bytes()).hexdigest() == config["extension"]["sha256"]
    return resolved

def validate_rnn_oracle(config):
    approved = config["reviewed_build_provenance"]
    assert approved.keys() == {"path", "sha256", "schema"}
    assert approved["schema"] == "vulkan-terminal-build-provenance-v1"
    manifest_path = Path(approved["path"])
    assert manifest_path.is_absolute() and manifest_path.resolve(strict=True) == manifest_path
    manifest_bytes = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == approved["sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest["schema"] == approved["schema"]
    assert manifest["repository_root"] == config["repository_root"]
    oracle = Path(config["rnn_oracle_source"])
    assert oracle == Path(config["repository_root"]) / 'tests/python/test_vulkan_fused_rnn.py'
    assert oracle.is_absolute() and oracle.resolve(strict=True) == oracle
    assert hashlib.sha256(oracle.read_bytes()).hexdigest() == manifest["gate_sources"][str(oracle)]
    return oracle

def bind_backend():
    global torch, pytorch_vulkan
    import torch
    import pytorch_vulkan
    loaded = Path(pytorch_vulkan._C.__file__).resolve(strict=True)
    expected = Path(config["extension"]["path"]).resolve(strict=True)
    assert loaded == expected, (loaded, expected)
    digest = hashlib.sha256(loaded.read_bytes()).hexdigest()
    assert digest == config["extension"]["sha256"]
    package = validate_package_origin(config, pytorch_vulkan.__file__,
                                      pytorch_vulkan.__path__, pytorch_vulkan._C.__file__)
    print(json.dumps(dict(loaded_extension=str(loaded), sha256=digest,
                          package=str(package), binding=config), sort_keys=True),
          file=sys.stderr, flush=True)
    print("TERMINAL_BINARY_BINDING_OK", file=sys.stderr, flush=True)

def operands():
    return (torch.tensor([[1., 2.]], dtype=torch.float32).to('vk:0'),
            torch.tensor([[3.], [4.]], dtype=torch.float32).to('vk:0'))

def matrix_cat():
    a, b = operands()
    y = torch.cat([torch.mm(a, b), torch.mm(a, b)], dim=0)
    assert y.device.type == 'vk' and y.dtype == torch.float32
    torch.vk.synchronize()
    assert torch.equal(y.cpu(), torch.tensor([[11.], [11.]], dtype=torch.float32))

def assert_finalizing():
    assert pytorch_vulkan._C.execution_mode() == 'sync'
    assert not pytorch_vulkan._C.training_step_active()
    assert pytorch_vulkan._C.pending_compute_count() == 0
    assert pytorch_vulkan._C.completion_hold_expiry_count() == 0

def terminal_ok():
    assert_finalizing()
    assert pytorch_vulkan._C.fallback_count() == 0
    print("TERMINAL_LIFECYCLE_OK", flush=True)

def register_finalizer(action):
    # The holder survives until this before-import atexit handler. Its __del__
    # executes while CPython is alive and in finalizing-sync, not at module clear.
    done = []
    class Finalizer:
        def __del__(self):
            action()
            done.append(True)
    holder = {"owner": Finalizer()}
    def last_required_handler():
        holder.pop("owner")
        gc.collect()
        assert done == [True], "finalizer failed or did not run"
        terminal_ok()
    atexit.register(last_required_handler)
'''

CHILD_CLEAN = r'''
def last_required_handler():
    assert_finalizing()
    terminal_ok()
atexit.register(last_required_handler)
bind_backend()
matrix_cat()
assert pytorch_vulkan._C.execution_mode() == config["execution_mode"]
'''

CHILD_EXCEPTION = r'''
def last_required_handler():
    assert_finalizing()
    terminal_ok()
atexit.register(last_required_handler)
bind_backend()
def ordinary_work():
    matrix_cat()
ordinary_work()  # all local tensor owners have left before the uncaught error
gc.collect()
raise RuntimeError("TERMINAL_EXPECTED_ORDINARY_EXCEPTION")
'''

CHILD_GLOBALS = r'''
def last_required_handler():
    assert_finalizing()
    assert LIVE_TENSOR.device.type == 'vk'
    assert LIVE_PINNED.device.type == 'cpu' and LIVE_PINNED.is_pinned(device='vk')
    torch.vk.synchronize()
    assert torch.equal(LIVE_TENSOR.cpu(), torch.tensor([[11.]], dtype=torch.float32))
    assert torch.equal(LIVE_PINNED, torch.tensor([[1., 2.]], dtype=torch.float32))
    terminal_ok()
atexit.register(last_required_handler)
bind_backend()
LIVE_PINNED = torch.tensor([[1., 2.]], dtype=torch.float32).pin_memory(device='vk')
LIVE_TENSOR = torch.mm(LIVE_PINNED.to('vk:0'),
                       torch.tensor([[3.], [4.]], dtype=torch.float32).to('vk:0'))
# Both actual owners intentionally remain module globals through normal exit.
'''

CHILD_PENDING = r'''
def last_required_handler():
    assert_finalizing()
    torch.vk.synchronize()
    assert torch.equal(PENDING.cpu(), torch.tensor([[11.]], dtype=torch.float32))
    matrix_cat()
    terminal_ok()
atexit.register(last_required_handler)
bind_backend()
a, b = operands()
torch.vk.synchronize()
assert pytorch_vulkan._C.execution_mode() == 'async'
pytorch_vulkan._C.test_hold_completion(True)
try:
    PENDING = torch.mm(a, b)
    assert pytorch_vulkan._C.pending_compute_count() > 0
finally:
    pytorch_vulkan._C.test_hold_completion(False)
# No sync, query, readback, private batch scope or extra backend call here: the
# ordinary eager recording and its retained tensors reach the early transition.
'''

CHILD_HANDLERS = r'''
state = {"after": False}
def before_import():
    assert state["after"], "after-import handler did not complete"
    print("before-import", file=sys.stderr, flush=True)
    assert_finalizing()
    torch.vk.synchronize()
    assert torch.equal(state["pending"].cpu(), torch.tensor([[11.]], dtype=torch.float32))
    matrix_cat()
    terminal_ok()
atexit.register(before_import)
bind_backend()
matrix_cat()
assert pytorch_vulkan._C.execution_mode() == 'async'
def after_import():
    print("after-import", file=sys.stderr, flush=True)
    assert pytorch_vulkan._C.execution_mode() == 'async'
    a, b = operands()
    pytorch_vulkan._C.test_hold_completion(True)
    try:
        first = torch.mm(a, b)
        assert pytorch_vulkan._C.pending_compute_count() > 0
    finally:
        pytorch_vulkan._C.test_hold_completion(False)
    # Self-release precedes every blocking call, never watchdog expiry.
    torch.vk.synchronize()
    assert torch.equal(first.cpu(), torch.tensor([[11.]], dtype=torch.float32))
    assert pytorch_vulkan._C.completion_hold_expiry_count() == 0
    pytorch_vulkan._C.test_hold_completion(True)
    try:
        state["pending"] = torch.mm(a, b)
        assert pytorch_vulkan._C.pending_compute_count() > 0
    finally:
        pytorch_vulkan._C.test_hold_completion(False)
    state["after"] = True
atexit.register(after_import)
'''

CHILD_MATRIX_FINALIZER = r'''
def finalizer_work():
    assert_finalizing()
    matrix_cat()
register_finalizer(finalizer_work)
bind_backend()
matrix_cat()
'''

CHILD_ERROR_FINALIZER = r'''
def finalizer_work():
    assert_finalizing()
    a, b = operands()
    matrix_cat()
    try:
        torch.mm(a, a)
    except RuntimeError as error:
        assert str(error) == "Vulkan mm requires matching 2-D matrices", str(error)
    else:
        raise AssertionError("ordinary malformed mm unexpectedly succeeded")
    # Catch inside __del__, check the exact expected error, then prove healthy
    # ordinary work still completes before the last handler may report success.
    matrix_cat()
register_finalizer(finalizer_work)
bind_backend()
matrix_cat()
'''

CHILD_UNUSED = r'''
assert config["execution_mode"] == 'async'
assert os.environ.get("PYTORCH_VULKAN_ASYNC_EXECUTION") != '0'
def finalizer_first_use():
    # This exported getter is an atomic observation, not python_platform().
    assert pytorch_vulkan._C.platform_construction_count() == 0
    matrix_cat()
    assert pytorch_vulkan._C.platform_construction_count() == 1
    assert_finalizing()
    matrix_cat()
register_finalizer(finalizer_first_use)
bind_backend()
assert pytorch_vulkan._C.platform_construction_count() == 0
# Deliberately no execution_mode/resource getter: those construct the runtime.
'''

CHILD_RNN = r'''
def finalizer_rnn():
    assert_finalizing()
    # Reuse exactly the existing differentiable CPU oracle without importing its
    # test module (which imports the backend and has unrelated fixtures).
    oracle_path = validate_rnn_oracle(config)
    tree = ast.parse(oracle_path.read_text())
    reference = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == '_differentiable_cpu_reference')
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[reference], type_ignores=[]), str(oracle_path), 'exec'), namespace)
    cpu_reference = namespace['_differentiable_cpu_reference']
    generator = torch.Generator(device='cpu').manual_seed(701)
    seeds = [torch.randn(1, 2, 4, generator=generator, dtype=torch.float32),
             torch.randn(5, 4, generator=generator, dtype=torch.float32) * 0.1,
             torch.randn(5, 5, generator=generator, dtype=torch.float32) * 0.1,
             torch.randn(5, generator=generator, dtype=torch.float32) * 0.1]
    gradient = torch.randn(1, 2, 5, generator=generator, dtype=torch.float32)
    # Each selective family exercises the unrequested/scratch gradient owners;
    # the all-requested case additionally checks the normal complete backward.
    for mask in [tuple(i == selected for i in range(4)) for selected in range(4)] + [(True,) * 4]:
        cpu_inputs = [seed.clone().requires_grad_(needed) for seed, needed in zip(seeds, mask)]
        vk_inputs = [seed.clone().to('vk:0').requires_grad_(needed) for seed, needed in zip(seeds, mask)]
        expected = cpu_reference(*cpu_inputs)
        actual = torch.ops.pytorch_vulkan.rnn_sequence(*vk_inputs)
        assert actual.device.type == 'vk'
        expected.backward(gradient)
        actual.backward(gradient.to('vk:0'))
        torch.vk.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)
        for needed, device_input, cpu_input in zip(mask, vk_inputs, cpu_inputs):
            if needed:
                assert device_input.grad is not None and device_input.grad.device.type == 'vk'
                torch.testing.assert_close(device_input.grad.cpu(), cpu_input.grad, rtol=3e-3, atol=3e-3)
            else:
                assert device_input.grad is None and cpu_input.grad is None
        assert_finalizing()
register_finalizer(finalizer_rnn)
bind_backend()
matrix_cat()
'''


def _child_source(body):
    return CHILD_COMMON + "\n" + body


def test_clean_exit():
    run_terminal_child(_child_source(CHILD_CLEAN), 0)


def test_exception_exit():
    run_terminal_child(_child_source(CHILD_EXCEPTION), 1)


def test_live_tensor_and_pinned_globals():
    run_terminal_child(_child_source(CHILD_GLOBALS), 0)


def test_pending_transition():
    run_terminal_child(_child_source(CHILD_PENDING), 0)


def test_handlers_before_after_import():
    run_terminal_child(_child_source(CHILD_HANDLERS), 0)


def test_finalizer_matrix_cat():
    run_terminal_child(_child_source(CHILD_MATRIX_FINALIZER), 0)


def test_finalizer_ordinary_error():
    run_terminal_child(_child_source(CHILD_ERROR_FINALIZER), 0)


def test_unused_then_finalizer_first_use():
    run_terminal_child(_child_source(CHILD_UNUSED), 0)


def test_finalizer_rnn_scratch():
    run_terminal_child(_child_source(CHILD_RNN), 0)
