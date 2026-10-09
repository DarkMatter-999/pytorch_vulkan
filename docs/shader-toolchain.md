# Reproducible shader verification

Shader verification compares **existing** generated artifacts with source and
fresh exact SPIR-V bytes. It must not regenerate tracked files, normalize result
IDs, or replace historical capture hashes. Explicit generation is a separate
operation; changing compiled bytes requires a rebuild and fresh runtime evidence.

The checked-in pointwise and masked-select artifacts require shaderc 2026.3.
On Arch Linux x86-64, `tools/shader_toolchain.py` supports one authenticated,
complete profile: shaderc2026.3-1 plus glslang/SPIRV-Tools1:1.4.357.0-1. Package
SHA256s and signer fingerprint are pinned in the tool. Both detached signatures
and hashes must validate before extraction; nothing is installed or downgraded.
The profile pins the compiler, shader libraries, full version output, and the
actual linked-library paths/hashes (including host runtime libraries). An update,
missing dependency or edited profile fails visibly. This is not a portable
toolchain claim; other platforms need their own reviewed profile.

Prepare a **new** directory from already cached, signed packages:

```sh
.venv/bin/python tools/shader_toolchain.py prepare --directory /tmp/opencode/my-shader-profile
```

Use the emitted `profile.json` path, not an old executable with current libraries:

```sh
.venv/bin/python tools/shader_toolchain.py inspect --profile /tmp/opencode/my-shader-profile/profile.json
.venv/bin/python tools/shader_toolchain.py run --profile /tmp/opencode/my-shader-profile/profile.json -- .venv/bin/python tools/verify_operator_spv.py
cmake -S . -B build -DVULKAN_SHADER_TOOLCHAIN_PROFILE=/tmp/opencode/my-shader-profile/profile.json
```

Keep the usual build options when configuring. CMake validates the complete
profile and records `build/shader-toolchain.json`; its shader generation,
integrity and CTest commands retain the profile's SHA256. Standalone generators
and verifiers use the same `run` adapter. Without a selected profile, tools use
the system compiler and retain their existing strict mismatch failures—there
is no fallback from a missing or changed selected profile.

The adapter prepends only a `glslc` wrapper to PATH. **Only its compiler child**
receives the isolated library search path; Python/PyTorch, CMake and other
processes do not receive old `LD_LIBRARY_PATH` values. Existing LD_PRELOAD or
LD_AUDIT is incompatible with a pinned compiler profile. Shell configuration
and system libraries are never changed.

For a separately authorized full qualification:

```sh
.venv/bin/python tools/run_vulkan_qualification.py --device vk:0 --build build --output qualification.json --shader-toolchain-profile /tmp/opencode/my-shader-profile/profile.json
```

The runner validates before its device probe, propagates compiler selection to
every gate, and records a separate `shader-toolchain.json` and report provenance.
Environment-based callers must provide both `VULKAN_SHADER_TOOLCHAIN_PROFILE`
and `VULKAN_SHADER_TOOLCHAIN_PROFILE_SHA256`; prefer `run` or the runner CLI to
compute and validate them. This provenance does not qualify GPU execution or
rebind any historical source/binary identity.

Pooling's CMake generation target is explicit (`vulkan_pooling_shader_generate`)
and is not a dependency of its read-only integrity target. Standalone explicit
generation remains available through the compiler adapter.

`generate_operator_spv.py --output-directory DIRECTORY` supports independent
staging; optional `--expected-header-sha256` and `--expected-manifest-sha256`
refuse writes unless both regenerated products match the approved identities.
Never use generation to make a verification failure disappear without an
explicit artifact disposition and evidence-preservation decision.
