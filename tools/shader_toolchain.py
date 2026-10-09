#!/usr/bin/env python3
"""Explicit, byte-pinned shaderc 2026.3 isolation; never change Python's loader.

Existing generators/verifiers continue to invoke ``glslc``. ``run`` and CMake
prepend our compiler-only wrapper to PATH; only that wrapper's compiler child
receives the isolated library path. A selected missing/changed profile is fatal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import io
import tarfile
import posixpath
import shutil
from contextlib import ExitStack


PROFILE_ENV = "VULKAN_SHADER_TOOLCHAIN_PROFILE"
HASH_ENV = "VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256"
WRAPPERS = Path(__file__).resolve().parent / "shader-toolchain-bin"
LDD = "/usr/bin/ldd"
VERSION = "2026.3\n1:1.4.357.0\n1:1.4.357.0\n\nTarget: SPIR-V 1.0\n"
SIGNER = "262A58EC6C51F7EA395B2E2DFDC3040B92ACA748"
APPROVED_PACKAGES = {
    "shaderc-2026.3-1-x86_64.pkg.tar.zst": {
        "sha256": "4b8c63f7e5074aa3551507d38da77584d017fac8085d6d057af0e08cc0ca1b1b", "signer": SIGNER},
    "glslang-1:1.4.357.0-1-x86_64.pkg.tar.zst": {
        "sha256": "c8417ab41fcccb7ce1f6f9da447733ad612d918c0d530484227e7af95ed735d1", "signer": SIGNER},
    "spirv-tools-1:1.4.357.0-1-x86_64.pkg.tar.zst": {
        "sha256": "aee9b717cfd61aa74ca85d7e5608d7f8914e57e965a2cda982a922728b4772bb", "signer": SIGNER},
}
APPROVED_FILES = {
    "usr/bin/glslc": "4a4743cde357af0949cdbc07668802297327e993e548f80f4e2ee67ba9b6c74d",
    "usr/lib/libglslang.so.16": "0426c31ff9c196910013aef27f192f82a1d4db5d0b49654c409b067d26c06af2",
    "usr/lib/libSPIRV-Tools-opt.so": "1046640e5545652ecfefa828280c99879771c829eb68d0475118859347ad05b2",
    "usr/lib/libSPIRV-Tools.so": "e12f83159f86b969b6fc5c3668a670f15038d3182b510ac04fce68e833b7be2a",
}
FLAGS = ["-Os"]
TARGET = "default Vulkan; SPIR-V 1.0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compiler_environment(prefix):
    env = os.environ.copy()
    if env.get("LD_PRELOAD") or env.get("LD_AUDIT"):
        raise ValueError("shader compiler profile does not permit LD_PRELOAD/LD_AUDIT")
    env["LD_LIBRARY_PATH"] = str(prefix / "usr/lib")
    return env


def _linked_libraries(executable, environment):
    output = subprocess.check_output([LDD, str(executable)], env=environment, text=True)
    if "not found" in output:
        raise ValueError("shader compiler linked library missing: " + output.strip())
    libraries = {}
    for line in output.splitlines():
        match = re.search(r"^\s*(\S+) => (/\S+) ", line)
        loader = re.search(r"^\s*(/\S+) \(", line)
        if match:
            name, path = match.groups()
        elif loader:
            path = loader.group(1)
            name = Path(path).name
        else:
            continue  # Linux vDSO has no on-disk library.
        file = Path(path).resolve(strict=True)
        libraries[name] = {"path": str(file), "sha256": sha256(file)}
    if not libraries:
        raise ValueError("shader compiler linked library inventory is empty")
    return libraries


def _actual_identity(prefix):
    for name, expected in APPROVED_FILES.items():
        file = prefix / name
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"shader compiler profile file identity mismatch: {name}")
        if not file.resolve().is_relative_to(prefix.resolve()):
            raise ValueError(f"shader compiler profile file escapes prefix: {name}")
    executable = prefix / "usr/bin/glslc"
    env = _compiler_environment(prefix)
    version = subprocess.check_output([str(executable), "--version"], env=env, text=True)
    if version != VERSION:
        raise ValueError(f"shader compiler profile version mismatch: {version!r}")
    libraries = _linked_libraries(executable, env)
    # ldd must actually select the approved copies, not merely find them on disk.
    for name in APPROVED_FILES:
        if name.startswith("usr/lib/"):
            library = libraries.get(Path(name).name)
            if library != {"path": str((prefix / name).resolve()), "sha256": APPROVED_FILES[name]}:
                raise ValueError(f"shader compiler linked library identity mismatch: {name}")
    return version, libraries


def profile_provenance(profile, expected_sha256=None):
    profile = Path(profile).resolve()
    if not profile.is_file():
        raise ValueError(f"shader compiler profile missing: {profile}")
    actual_hash = sha256(profile)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError(f"shader compiler profile hash mismatch: {profile}")
    data = json.loads(profile.read_text())
    keys = {"schema_version", "prefix", "packages", "files", "version",
            "linked_libraries", "flags", "target_environment"}
    if (not isinstance(data, dict) or data.keys() != keys or data["schema_version"] != 1
            or data["packages"] != APPROVED_PACKAGES or data["files"] != APPROVED_FILES
            or data["flags"] != FLAGS or data["target_environment"] != TARGET):
        raise ValueError("shader compiler profile schema/package identity mismatch")
    prefix = Path(data["prefix"])
    if not prefix.is_absolute():
        raise ValueError("shader compiler profile prefix must be absolute")
    version, libraries = _actual_identity(prefix)
    if version != data["version"] or libraries != data["linked_libraries"]:
        raise ValueError("shader compiler profile linked library identity changed")
    return {"profile": str(profile), "profile_sha256": actual_hash,
            "compiler": {"path": str(prefix / "usr/bin/glslc"),
                         "sha256": APPROVED_FILES["usr/bin/glslc"], "version": version},
            "linked_libraries": libraries, "packages": data["packages"],
            "flags": FLAGS, "target_environment": TARGET}


def selected_environment(profile, expected_sha256=None):
    provenance = profile_provenance(profile, expected_sha256)
    env = os.environ.copy()
    env[PROFILE_ENV] = provenance["profile"]
    env[HASH_ENV] = provenance["profile_sha256"]
    env["PATH"] = str(WRAPPERS) + os.pathsep + env.get("PATH", os.defpath)
    # Do NOT export the compiler's LD_LIBRARY_PATH to PyTorch or CMake.
    return env


def invoke_compiler(arguments, profile, expected_sha256=None):
    provenance = profile_provenance(profile, expected_sha256)
    compiler = Path(provenance["compiler"]["path"])
    return subprocess.call([str(compiler), *arguments],
                           env=_compiler_environment(compiler.parents[2]))


def compiler_main():
    profile = os.environ.get(PROFILE_ENV)
    if not profile or not os.environ.get(HASH_ENV):
        raise ValueError("compiler wrapper requires an explicit shader profile and hash; no system fallback")
    return invoke_compiler(sys.argv[1:], profile, os.environ[HASH_ENV])


def _archive_path(name):
    if (not name or "\x00" in name or name.startswith("/")
            or ".." in name.split("/")):
        raise ValueError(f"unsafe archive member path: {name!r}")
    normalized = posixpath.normpath(name)
    if normalized == ".":
        raise ValueError(f"unsafe archive member path: {name!r}")
    return normalized


def preflight_archives(manifests):
    """Validate ALL package members/link graphs before any destination exists.

    No data member may have a link/non-directory parent, regardless of archive
    order. Links must resolve to an existing confined member with no cycles;
    hardlinks must terminate at regular data. Package metadata and shared
    directory entries may repeat across packages; other duplicates are rejected.
    """
    inventory = {}
    links = {}
    metadata = {".BUILDINFO", ".MTREE", ".PKGINFO"}
    for members in manifests:
        local = set()
        for member in members:
            name = _archive_path(member.name)
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"unsafe archive member type: {name}")
            if name in local or (name in inventory and not
                    (member.isdir() and inventory[name].isdir() or name in metadata
                     and member.isfile() and inventory[name].isfile())):
                raise ValueError(f"duplicate archive member: {name}")
            local.add(name)
            inventory[name] = member
            if member.issym() or member.islnk():
                target = member.linkname
                if not target or "\x00" in target or target.startswith("/"):
                    raise ValueError(f"unsafe archive link target: {name}")
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), target)
                                           if member.issym() else target)
                if target == ".." or target.startswith("../"):
                    raise ValueError(f"escaping archive link: {name}")
                links[name] = target
    for name in inventory:
        parent = posixpath.dirname(name)
        while parent:
            if parent in inventory and not inventory[parent].isdir():
                raise ValueError(f"archive member has link/non-directory parent: {name}")
            parent = posixpath.dirname(parent)

    def resolve(name, seen):
        if name in seen:
            raise ValueError(f"archive link cycle: {name}")
        parts = name.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            if prefix in links:
                suffix = parts[index:]
                target = posixpath.normpath(posixpath.join(links[prefix], *suffix))
                if target.startswith("../") or target == "..":
                    raise ValueError(f"escaping archive link chain: {name}")
                return resolve(target, seen | {prefix})
        if name not in inventory:
            raise ValueError(f"archive link has missing target: {name}")
        return name

    resolved_links = {}
    for name in links:
        target = resolve(name, set())
        if inventory[name].islnk() and not inventory[target].isfile():
            raise ValueError(f"archive hardlink does not target regular data: {name}")
        resolved_links[name] = target
    return resolved_links


def _extract_preflighted(archives, directory):
    """Write data first, links last; never tar.extract/extractall or tar -xf."""
    resolved_links = preflight_archives([members for _, members in archives])
    for archive, members in archives:
        for member in members:
            path = directory / _archive_path(member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.resolve().is_relative_to(directory):
                    raise ValueError(f"archive destination escaped prefix: {path}")
                with archive.extractfile(member) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output)
                path.chmod(member.mode & 0o777)  # Never adopt owners/setuid/device metadata.
    # Publish links to final regular data, never traverse a just-created link.
    for _, members in archives:
        for member in members:
            if member.islnk():
                name = _archive_path(member.name)
                path = directory / name
                target = directory / resolved_links[name]
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(target, path)
    for _, members in archives:
        for member in members:
            if member.issym():
                path = directory / _archive_path(member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(member.linkname)


def prepare_profile(cache, directory, keyring):
    """Authenticate known cached archives before extracting; no installation."""
    cache, directory, keyring = Path(cache), Path(directory).resolve(), Path(keyring)
    if not directory.is_relative_to(Path("/tmp/opencode").resolve()):
        raise ValueError("shader profile extraction is restricted to /tmp/opencode")
    if directory.exists():
        raise ValueError(f"profile directory already exists; refusing overwrite: {directory}")
    proofs = []
    for name, expected in APPROVED_PACKAGES.items():
        archive = cache / name
        if not archive.is_file() or sha256(archive) != expected["sha256"]:
            raise ValueError(f"cached shader package hash mismatch/missing: {name}")
        command = ["gpgv", "--status-fd", "1", "--keyring", str(keyring),
                   str(archive) + ".sig", str(archive)]
        result = subprocess.run(command, capture_output=True, text=True)
        signers = re.findall(r"^\[GNUPG:\] VALIDSIG ([0-9A-F]+) ", result.stdout, re.M)
        if result.returncode or signers != [expected["signer"]]:
            raise ValueError(f"cached shader package signature failed: {name}\n{result.stderr}")
        proofs.append({"command": command, "sha256": expected["sha256"],
                       "signature_sha256": sha256(str(archive) + ".sig"),
                       "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    with ExitStack() as stack:
        archives = []
        for name, expected in APPROVED_PACKAGES.items():
            payload = (cache / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(f"cached shader package changed after authentication: {name}")
            if payload.startswith(b"\x28\xb5\x2f\xfd"):
                payload = subprocess.run(["zstd", "-dc"], input=payload, capture_output=True, check=True).stdout
            archive = stack.enter_context(tarfile.open(fileobj=io.BytesIO(payload), mode="r:*"))
            archives.append((archive, archive.getmembers()))
        preflight_archives([members for _, members in archives])
        directory.mkdir(mode=0o700, parents=True)
        _extract_preflighted(archives, directory)
    version, libraries = _actual_identity(directory)
    data = {"schema_version": 1, "prefix": str(directory), "packages": APPROVED_PACKAGES,
            "files": APPROVED_FILES, "version": version, "linked_libraries": libraries,
            "flags": FLAGS, "target_environment": TARGET}
    profile = directory / "profile.json"
    profile.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (directory / "signatures.json").write_text(json.dumps(proofs, indent=2) + "\n")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--cache", type=Path, default=Path("/var/cache/pacman/pkg"))
    prepare.add_argument("--directory", type=Path, required=True)
    prepare.add_argument("--keyring", type=Path, default=Path("/etc/pacman.d/gnupg/pubring.gpg"))
    for action in ("inspect", "run"):
        sub = commands.add_parser(action)
        sub.add_argument("--profile", type=Path, required=True)
        sub.add_argument("--expected-profile-sha256")
        sub.add_argument("--provenance", type=Path)
        if action == "run":
            sub.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare_profile(args.cache, args.directory, args.keyring))
        return 0
    provenance = profile_provenance(args.profile, args.expected_profile_sha256)
    if args.provenance:
        args.provenance.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    if args.action == "inspect":
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("shader profile run requires a command after --")
    return subprocess.call(command, env=selected_environment(args.profile, provenance["profile_sha256"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"shader toolchain: {error}")
