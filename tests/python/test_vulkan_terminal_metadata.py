"""Stdlib-only metadata tests; no terminal import, collection or child launch.

Fixtures are retained in a new directory selected explicitly by the CPU caller.
AST extraction executes actual product validation definitions, not an oracle.
Absent TERMINAL_METADATA_FIXTURES means discovery skips this retained-fixture
harness; when set, it must name a new absolute fixture root and every test runs.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import unittest

ROOT = Path(__file__).resolve().parents[2]
PRODUCT = ROOT / "tests/python/test_vulkan_terminal_lifecycle.py"


def product_functions(root, acceptance):
    tree = ast.parse(PRODUCT.read_text())
    namespace = dict(Path=Path, hashlib=hashlib, json=json, os=os, ast=ast, re=re, shlex=shlex,
                     ROOT=root, __file__=str(acceptance))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name.startswith(("_metadata_", "_validate_"))]
    assert any(n.name == "_validate_gate_provenance" for n in funcs)
    exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])),
                 str(PRODUCT), "exec"), namespace)
    common = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "CHILD_COMMON" for t in n.targets))
    child = ast.parse(common)
    origin = [n for n in child.body if isinstance(n, ast.FunctionDef)
              and n.name.startswith("validate_")]
    assert any(n.name == "validate_package_origin" for n in origin)
    exec(compile(ast.fix_missing_locations(ast.Module(body=origin, type_ignores=[])),
                 "<actual-child-metadata>", "exec"), namespace)
    return namespace


def live_payload():
    """Populate UNAPPROVED in-memory audit data from existing accepted evidence.

    No approval-manifest file, compiler/loader execution or build-tree write.
    Expected target/count facts are taken from the accepted extension inventory,
    not from the product's closure selector.
    """
    base = ROOT / ".superpowers/sdd/2026-10-26-vulkan-terminal-lifecycle"
    review_path = base / "task-6-integration-review-package-01/manifest.json"
    inv_path = base / "task-5-abi-review-package-01/build-inventory.json"
    build_root = ROOT / "build-terminal-lifecycle-extension-01"
    inventory = json.loads(inv_path.read_text())
    build = inventory[build_root.name]
    review = json.loads(review_path.read_text())
    assert len(build["commands"]) == 74 and len(build["objects"]) == 45
    assert "src/vulkan_async_transfer.cpp" not in review
    targets = {"pytorch_vulkan_python", "vulkan_device_guard", "vulkan_allocator",
               "vulkan_buffer", "vulkan_platform", "vulkan_lifecycle"}
    selected = [c for c in build["commands"] if any(c["output"].startswith(
        "CMakeFiles/" + target + ".dir/") for target in targets)]
    assert len(selected) == 41
    missing = [c for c in build["commands"] if c["file"].startswith(str(ROOT / "src") + "/")
               and str((build_root / c["output"]).relative_to(ROOT)) not in build["objects"]]
    assert len(missing) == 18
    for command in missing:
        assert not (build_root / command["output"]).exists()
    for command in selected:
        assert str((build_root / command["output"]).relative_to(ROOT)) in build["objects"]
    extension_paths = [ROOT / name for name in build["binaries"] if "/pytorch_vulkan/_C" in name]
    assert len(extension_paths) == 1
    record = lambda p: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    mapping = lambda paths: {str(p): record(p)["sha256"] for p in paths}
    init = ROOT / "python/pytorch_vulkan/__init__.py"
    oracle = ROOT / "tests/python/test_vulkan_fused_rnn.py"
    cmake = (ROOT / "CMakeLists.txt").read_text()
    package_sources = {ROOT / p for p in re.findall(r"python/pytorch_vulkan/[A-Za-z0-9_]+\.py", cmake)}
    shaders = set(re.findall(r"src/vulkan/shaders/(?:generated|glsl)/[A-Za-z0-9_.]+", cmake))
    shaders.update(re.findall(r"tools/(?:generate|verify)_[A-Za-z0-9_]+_spv\.py", cmake))
    shaders.update("src/" + p for p in re.findall(r'"(vulkan/shaders/generated/[A-Za-z0-9_]+_spv\.h)"',
                                                (ROOT / "src/vulkan_compute.cpp").read_text()))
    assert len(shaders) == 68 and len(package_sources) == 4
    shaders.add("tools/shader_toolchain.py")
    assert len(shaders) == 69
    profile_provenance = build_root / "shader-toolchain.json"
    profile = Path(json.loads(profile_provenance.read_text())["profile"])
    config = dict(repository_root=str(ROOT), package_root=str(build_root), extension=record(extension_paths[0]),
                  package_init=record(init), rnn_oracle_source=str(oracle), reviewed_build_provenance=dict(
                      path="/UNAPPROVED-IN-MEMORY-AUDIT-NO-FILE", sha256="0" * 64,
                      schema="vulkan-terminal-build-provenance-v1"))
    runtime = {ROOT / name for name in review if name.startswith("src/")}
    runtime.update(Path(c["file"]) for c in selected)
    runtime.add(ROOT / "CMakeLists.txt")
    manifest = dict(schema=config["reviewed_build_provenance"]["schema"], repository_root=str(ROOT),
        package_root=str(build_root), extension=config["extension"], package_init=config["package_init"],
        source_review=record(review_path), build_inventory=record(inv_path), runtime_sources=mapping(runtime),
        gate_sources=mapping(package_sources | {PRODUCT, oracle, ROOT / "tools/qualification_process_supervisor.py"}),
        shaders=mapping({ROOT / name for name in shaders}), shader_profile=record(profile),
        shader_provenance=record(profile_provenance), build_cache=record(build_root / "CMakeCache.txt"))
    return config, manifest, dict(commands=74, recorded_objects=45, selected_objects=41,
                                 unbuilt_src_outputs=18, shader_inputs=68, shader_and_toolchain_inputs=69,
                                 package_sources=4)


def validate_live_payload(config, manifest):
    """Execute actual product payload validation, excluding ONLY manifest file IO.

    The immutable-file loader is covered with fixtures; this read-only audit must
    not mint a real approval manifest. All remaining product statements execute.
    """
    tree = ast.parse(PRODUCT.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_gate_provenance")
    removed = []
    body = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
                "manifest_path", "manifest") for t in node.targets):
            removed.append(node)
        else:
            body.append(node)
    assert len(removed) == 2
    fn.body = body
    fn.args.args.append(ast.arg(arg="manifest"))
    namespace = product_functions(ROOT, PRODUCT)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 "<actual-live-payload-no-manifest-file>", "exec"), namespace)
    return namespace[fn.name](config, manifest)


def run_rnn_metadata_prefix(config):
    tree = ast.parse(PRODUCT.read_text())
    rnn = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "CHILD_RNN" for t in n.targets))
    fn = next(n for n in ast.parse(rnn).body if isinstance(n, ast.FunctionDef) and n.name == "finalizer_rnn")
    assert ast.unparse(fn.body[0]) == "assert_finalizing()"
    body = []
    for node in fn.body[1:]:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "tree" for t in node.targets):
            break
        body.append(node)
    assert body
    namespace = product_functions(ROOT, PRODUCT)
    namespace["config"] = config
    exec(compile(ast.Module(body=body, type_ignores=[]), "<actual-rnn-metadata-prefix-only>", "exec"), namespace)
    return namespace["oracle_path"]


@unittest.skipUnless("TERMINAL_METADATA_FIXTURES" in os.environ,
                     "opt-in retained CPU metadata harness: set TERMINAL_METADATA_FIXTURES to a NEW absolute directory")
class MetadataTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ["TERMINAL_METADATA_FIXTURES"])
        assert base.is_absolute(), "explicit fixture root must be absolute"
        self.root = base / self._testMethodName
        self.root.mkdir(parents=True)
        self.build = self.root / "build"
        (self.build / "pytorch_vulkan").mkdir(parents=True)
        self.acceptance = self.root / "tests/python/test_vulkan_terminal_lifecycle.py"
        self.write(self.acceptance, "acceptance fixture")
        self.oracle = self.root / "tests/python/test_vulkan_fused_rnn.py"
        self.write(self.oracle, "oracle fixture")
        self.init = self.root / "python/pytorch_vulkan/__init__.py"
        self.write(self.init, "approved init")
        self.extension = self.build / "pytorch_vulkan/_C.fixture.so"
        self.write(self.extension, "ELF fixture, never executable")
        self.config = dict(repository_root=str(self.root), package_root=str(self.build),
                           extension=self.record(self.extension), package_init=self.record(self.init),
                           rnn_oracle_source=str(self.oracle), execution_mode="async",
                           device="fixture", driver="fixture", validation="fixture")
        self.ns = product_functions(self.root, self.acceptance)

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(text)

    @staticmethod
    def record(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def origin(self, file, search=None, extension=None):
        c = self.config
        return self.ns["validate_package_origin"](
            c, str(file), search or [str(self.build / "pytorch_vulkan")], str(extension or self.extension))

    def test_symlink_import_preserves_build_root(self):
        link = self.build / "pytorch_vulkan/__init__.py"
        link.symlink_to(self.init)
        self.origin(link)

    def test_cmake_copy_on_error_preserves_source_hash(self):
        target = self.build / "pytorch_vulkan/__init__.py"
        self.write(target, self.init.read_text())
        self.origin(target)

    def test_origin_negative_bindings(self):
        target = self.build / "pytorch_vulkan/__init__.py"
        target.symlink_to(self.init)
        wrong = self.root / "wrong/__init__.py"
        self.write(wrong, self.init.read_text())
        bad_link = self.root / "other-build/pytorch_vulkan/__init__.py"
        bad_link.parent.mkdir(parents=True)
        bad_link.symlink_to(self.init)
        other = self.root / "wrong/_C.so"
        self.write(other, self.extension.read_text())
        for label, file, search, ext in [
            ("unrelated root", bad_link, None, None),
            ("package search", target, [str(wrong.parent)], None),
            ("extension path", target, None, other),
        ]:
            with self.subTest(label=label), self.assertRaises((AssertionError, ValueError)):
                self.origin(file, search, ext)
        self.config["package_init"]["sha256"] = "0" * 64
        with self.assertRaises((AssertionError, ValueError)):
            self.origin(target)
        self.config["package_init"] = self.record(self.init)
        self.config["extension"]["sha256"] = "0" * 64
        with self.assertRaises((AssertionError, ValueError)):
            self.origin(target)

    def test_unrelated_symlink_source_and_wrong_copy_rejected(self):
        decoy = self.root / "unrelated/__init__.py"
        self.write(decoy, self.init.read_text())
        target = self.build / "pytorch_vulkan/__init__.py"
        target.symlink_to(decoy)
        with self.assertRaises((AssertionError, ValueError)):
            self.origin(target)
        # Separate build path avoids modifying/removing the preserved symlink.
        copied = self.root / "copied-build/pytorch_vulkan/__init__.py"
        self.write(copied, "wrong source hash")
        self.config["package_root"] = str(copied.parent.parent)
        with self.assertRaises((AssertionError, ValueError)):
            self.origin(copied, [str(copied.parent)])

    def test_irrelevant_nonempty_lists_are_rejected(self):
        ordinary = self.root / "ordinary.txt"
        self.write(ordinary, "not a shader or profile")
        self.config.update(sources=[self.record(self.acceptance), self.record(self.oracle)],
                           shaders=[self.record(ordinary)], profile=[self.record(ordinary)])
        with self.assertRaises((AssertionError, ValueError, KeyError)):
            self.ns["_validate_gate_provenance"](self.config)

    def provenance_fixture(self, include_package=True):
        # Hand-specified roles, not product-derived expected membership.
        runtime = ["src/python_module.cpp", "src/vulkan_lifecycle.cpp",
                   "src/vulkan_python_lifecycle.cpp", "src/vulkan_platform.cpp",
                   "src/vulkan_execution.cpp", "src/vulkan_async_transfer.cpp",
                   "src/vulkan_compute.cpp", "src/vulkan_lifecycle.h"]
        cmake = self.root / "CMakeLists.txt"
        shader_names = ["src/vulkan/shaders/generated/rnn_sequence_spv.h",
                        "src/vulkan/shaders/generated/rnn_sequence_spv.sha256",
                        "src/vulkan/shaders/glsl/rnn_sequence.comp",
                        "tools/generate_rnn_sequence_spv.py", "tools/verify_rnn_sequence_spv.py"]
        package_names = ["python/pytorch_vulkan/__init__.py", "python/pytorch_vulkan/compiler.py"]
        self.write(cmake, "\n".join(shader_names + package_names))
        self.write(self.root / package_names[1], "package dependency fixture")
        for name in package_names:
            (self.build / "pytorch_vulkan" / Path(name).name).symlink_to(self.root / name)
        for name in runtime + shader_names:
            self.write(self.root / name, "fixture " + name)
        toolchain = self.root / "tools/shader_toolchain.py"
        prefix = self.root / "compiler"
        files = {}
        for name in ("usr/bin/glslc", "usr/lib/libfixture.so"):
            path = prefix / name
            self.write(path, "compiler metadata fixture " + name)
            files[name] = self.record(path)["sha256"]
        constants = dict(SIGNER="fixture-signer", APPROVED_PACKAGES={"fixture-package": dict(sha256="1" * 64, signer="fixture-signer")},
                         APPROVED_FILES=files, FLAGS=["-Os"], TARGET="fixture SPIR-V 1.0", VERSION="fixture version")
        self.write(toolchain, "\n".join(name + " = " + repr(value) for name, value in constants.items()))
        supervisor = self.root / "tools/qualification_process_supervisor.py"
        self.write(supervisor, "supervisor fixture, never executed")
        review = self.root / "combined-review.json"
        review_data = {name: dict(sha256=self.record(self.root / name)["sha256"])
                       for name in runtime + ["CMakeLists.txt"] if name != "src/vulkan_async_transfer.cpp"}
        self.write(review, json.dumps(review_data))
        objects = {}
        commands = []
        for i, name in enumerate(runtime[:-1]):
            output = f"CMakeFiles/fixture.dir/object-{i}.o"
            obj = self.build / output
            self.write(obj, "object fixture " + name)
            objects[str(obj.relative_to(self.root))] = self.record(obj)["sha256"]
            commands.append(dict(directory=str(self.build), file=str(self.root / name),
                                 output=output, command="not executed"))
        archive = self.build / "libfixture.a"
        self.write(archive, "archive fixture")
        link = self.build / "CMakeFiles/pytorch_vulkan_python.dir/link.txt"
        self.write(link, "c++ -o pytorch_vulkan/_C.fixture.so libfixture.a\n")
        archive_link = self.build / "CMakeFiles/fixture.dir/link.txt"
        self.write(archive_link, "ar qc libfixture.a " + " ".join(c["output"] for c in commands) + "\nranlib libfixture.a\n")
        commands.append(dict(directory=str(self.build), file=str(self.root / "src/vulkan_compute.cpp"),
                             output="CMakeFiles/unbuilt_native_test.dir/compute.cpp.o", command="not built"))
        inventory = self.root / "build-inventory.json"
        inv = {self.build.name: dict(commands=commands, objects=objects,
            archives={archive.name: dict(sha256=self.record(archive)["sha256"],
                      members={f"object-{i}.o": digest for i, digest in enumerate(objects.values())})},
            links={str(link.relative_to(self.root)): link.read_text(),
                   str(archive_link.relative_to(self.root)): archive_link.read_text()},
            binaries={str(self.extension.relative_to(self.root)): dict(sha256=self.config["extension"]["sha256"])})}
        self.write(inventory, json.dumps(inv))
        wrong_inventory = self.root / "wrong-binary-inventory.json"
        wrong_inv = copy.deepcopy(inv)
        wrong_inv[self.build.name]["binaries"][str(self.extension.relative_to(self.root))]["sha256"] = "0" * 64
        self.write(wrong_inventory, json.dumps(wrong_inv))
        # Exact production profile schema with hand-pinned tiny compiler metadata.
        real_profile = dict(schema_version=1, prefix=str(prefix), packages=constants["APPROVED_PACKAGES"],
                            files=files, version=constants["VERSION"], flags=constants["FLAGS"],
                            target_environment=constants["TARGET"], linked_libraries={
                                "libfixture.so": self.record(prefix / "usr/lib/libfixture.so")})
        profile = self.root / "profile.json"
        self.write(profile, json.dumps(real_profile))
        shader_provenance = self.build / "shader-toolchain.json"
        provenance = dict(profile=str(profile), profile_sha256=self.record(profile)["sha256"],
                          compiler=dict(path=str(Path(real_profile["prefix"]) / "usr/bin/glslc"),
                                        sha256=constants["APPROVED_FILES"]["usr/bin/glslc"],
                                        version=constants["VERSION"]),
                          linked_libraries=real_profile["linked_libraries"], packages=constants["APPROVED_PACKAGES"],
                          flags=constants["FLAGS"], target_environment=constants["TARGET"])
        self.write(shader_provenance, json.dumps(provenance))
        cache = self.build / "CMakeCache.txt"
        self.write(cache, "VULKAN_SHADER_TOOLCHAIN_PROFILE:FILEPATH=" + str(profile) + "\n")
        mapping = lambda names: {str(self.root / name): self.record(self.root / name)["sha256"] for name in names}
        self.manifest = dict(schema="vulkan-terminal-build-provenance-v1",
            repository_root=str(self.root), package_root=str(self.build), extension=self.config["extension"],
            package_init=self.config["package_init"], source_review=self.record(review),
            build_inventory=self.record(inventory), runtime_sources=mapping(runtime + ["CMakeLists.txt"]),
            gate_sources=mapping(["python/pytorch_vulkan/__init__.py", "tools/qualification_process_supervisor.py",
                                 "tests/python/test_vulkan_terminal_lifecycle.py", "tests/python/test_vulkan_fused_rnn.py"]
                                 + ([package_names[1]] if include_package else [])),
            shaders=mapping(shader_names + ["tools/shader_toolchain.py"]),
            shader_profile=self.record(profile), shader_provenance=self.record(shader_provenance),
            build_cache=self.record(cache))
        self.manifest_path = self.root / "parent-approved-fixture.json"
        self.write(self.manifest_path, json.dumps(self.manifest))
        self.config["reviewed_build_provenance"] = dict(**self.record(self.manifest_path), schema=self.manifest["schema"])

    def test_review_subset_and_configured_unbuilt_commands(self):
        self.provenance_fixture()
        self.validate_manifest()
        inventory = json.loads(Path(self.manifest["build_inventory"]["path"]).read_text())
        selected = self.build / "CMakeFiles/fixture.dir/object-5.o"
        inventory[self.build.name]["objects"].pop(str(selected.relative_to(self.root)))
        missing = self.root / "missing-selected-object-inventory.json"
        self.write(missing, json.dumps(inventory))
        with self.assertRaises((AssertionError, KeyError, ValueError)):
            self.validate_manifest(lambda m: m.update(build_inventory=self.record(missing)))

    def test_live_accepted_inventory_payload_readonly(self):
        config, manifest, facts = live_payload()
        self.assertEqual(validate_live_payload(config, manifest), manifest)
        build_root = Path(config["package_root"])
        build = json.loads(Path(manifest["build_inventory"]["path"]).read_text())[build_root.name]
        for label, mutate in [
            ("direct extension object", lambda b: b["objects"].pop(str((build_root /
                "CMakeFiles/pytorch_vulkan_python.dir/src/python_module.cpp.o").relative_to(ROOT)))),
            ("required dependency archive", lambda b: b["archives"].pop("libvulkan_lifecycle.a")),
            ("archive member", lambda b: b["archives"]["libvulkan_platform.a"]["members"].pop("vulkan_async_transfer.cpp.o")),
        ]:
            broken = copy.deepcopy(build)
            mutate(broken)
            with self.subTest(missing=label), self.assertRaises((AssertionError, KeyError, ValueError)):
                product_functions(ROOT, PRODUCT)["_metadata_compiled_sources"](
                    broken, build_root, Path(config["extension"]["path"]))
        print("LIVE READ-ONLY METADATA (NOT APPROVAL): " + json.dumps(facts, sort_keys=True))

    def test_rnn_prefix_uses_v1_oracle_binding_without_legacy_sources(self):
        self.provenance_fixture()
        self.assertNotIn("sources", self.config)
        try:
            self.assertEqual(run_rnn_metadata_prefix(self.config), self.oracle)
        except KeyError as exc:
            self.fail("valid v1 RNN metadata prefix rejected without legacy config key: " + str(exc))
        bad = copy.deepcopy(self.manifest)
        bad["gate_sources"][str(self.oracle)] = "0" * 64
        bad_path = self.root / "wrong-oracle-manifest.json"
        self.write(bad_path, json.dumps(bad))
        self.config["reviewed_build_provenance"] = dict(**self.record(bad_path), schema=bad["schema"])
        with self.assertRaises((AssertionError, ValueError)):
            run_rnn_metadata_prefix(self.config)
        self.config["reviewed_build_provenance"]["sha256"] = "0" * 64
        with self.assertRaises((AssertionError, ValueError)):
            run_rnn_metadata_prefix(self.config)

    def test_missing_package_dependency_rejected(self):
        self.provenance_fixture(include_package=False)
        with self.assertRaises((AssertionError, ValueError, KeyError)):
            self.validate_manifest()

    def validate_manifest(self, mutate=None):
        if mutate is not None:
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            path = self.root / ("negative-" + str(len(list(self.root.glob("negative-*")))) + ".json")
            self.write(path, json.dumps(changed))
            self.config["reviewed_build_provenance"] = dict(**self.record(path), schema=changed["schema"])
        return self.ns["_validate_gate_provenance"](self.config)

    def test_complete_provenance_and_bounded_negatives(self):
        self.provenance_fixture()
        self.validate_manifest()
        for group, name in [
            ("runtime_sources", "src/vulkan_lifecycle.cpp"),
            ("gate_sources", "tools/qualification_process_supervisor.py"),
            ("gate_sources", "python/pytorch_vulkan/__init__.py"),
        ]:
            with self.subTest(missing=name), self.assertRaises((AssertionError, ValueError, KeyError)):
                self.validate_manifest(lambda m, g=group, n=name: m[g].pop(str(self.root / n)))
        negatives = [
            ("test/oracle only", lambda m: m.update(runtime_sources={})),
            ("stale source", lambda m: m["gate_sources"].update({str(self.acceptance): "0" * 64})),
            ("wrong binary", lambda m: m["extension"].update(sha256="0" * 64)),
            ("wrong identity", lambda m: m.update(package_root=str(self.root))),
            ("wrong schema", lambda m: m.update(schema="arbitrary")),
            ("irrelevant source", lambda m: m["runtime_sources"].update({str(self.oracle): self.record(self.oracle)["sha256"]})),
            ("arbitrary shader", lambda m: m.update(shaders={str(self.oracle): self.record(self.oracle)["sha256"]})),
            ("arbitrary profile", lambda m: m.update(shader_profile=self.record(self.oracle))),
            ("wrong inventory", lambda m: m.update(build_inventory=self.record(self.oracle))),
            ("inventory binary join", lambda m: m.update(build_inventory=self.record(self.root / "wrong-binary-inventory.json"))),
            ("missing generated header", lambda m: m["shaders"].pop(str(self.root / "src/vulkan/shaders/generated/rnn_sequence_spv.h"))),
            ("extra shader record", lambda m: m["shaders"].update({str(self.oracle): self.record(self.oracle)["sha256"]})),
            ("stale manifest", lambda m: None),
        ]
        for label, mutation in negatives:
            with self.subTest(label=label), self.assertRaises((AssertionError, ValueError, KeyError, TypeError)):
                if label == "stale manifest":
                    self.config["reviewed_build_provenance"]["sha256"] = "0" * 64
                    self.validate_manifest()
                else:
                    self.validate_manifest(mutation)

    def test_profile_schema_pins_and_artifact_join(self):
        self.provenance_fixture()
        profile_record = self.manifest["shader_profile"]
        provenance_record = self.manifest["shader_provenance"]
        profile = json.loads(Path(profile_record["path"]).read_text())
        for label, mutation in [
            ("schema", lambda p: p.update(schema_version=2)),
            ("compiler pins", lambda p: p["files"].update({"usr/bin/glslc": "0" * 64})),
            ("library join", lambda p: p.update(linked_libraries={})),
            ("flags", lambda p: p.update(flags=["arbitrary"])),
        ]:
            changed = copy.deepcopy(profile)
            mutation(changed)
            path = self.root / ("profile-negative-" + label.replace(" ", "-") + ".json")
            self.write(path, json.dumps(changed))
            with self.subTest(label=label), self.assertRaises((AssertionError, ValueError, KeyError)):
                self.ns["_metadata_profile"](self.record(path), provenance_record,
                                             self.root / "tools/shader_toolchain.py")


if __name__ == "__main__":
    unittest.main()
