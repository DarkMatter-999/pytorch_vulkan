# Wrapper-first qualification and evidence preservation

Import `pytorch_vulkan` before any autograd execution in the process. Upstream
`torch` import without prior backward is allowed. Late wrapper import after CPU
backward remains an unsupported shared-engine initialization history; forwarding
APIs does not fix that engine limitation or disable autograd threading.

Python qualification uses an explicit entry point, without changing test order,
bodies, arguments or pytest exit status:

```sh
PYTHONPATH=.:build:tests/python .venv/bin/python tools/vulkan_wrapper_pytest.py -q -rs tests/python
```

The same driver is registered for the full Python CTest suite, supported-workload
and runtime-foundation tests, and runner conformance/stress/additional-device
tests. Standalone manifest, model, bootstrap-provenance and shader validators stay
CPU-only and import-independent. The runner records `entry_point: wrapper-first`
and retains its separate `bootstrap_provenance` gate, ordinary timeout/cleanup,
skip/unavailability classification and all shader verification gates.

## Historical evidence versus current qualification

Before replacing published evidence, preserve complete published files byte-for-byte
in an immutable excluded archive, with file SHA256/size and canonical record-payload
SHA256s. Historical capture identities are not rewritten. Unaffected operator,
Linear, general-matmul sync, workload and numerical MSE payloads must remain exact.
Source-bound public-MSE and connected-model records require fresh execution when
their dependencies change; never rebind an old receipt to new source hashes.
Raw captures alone do not establish current-source qualification.

## Capture and guarded publication

After independent tooling review and capture authorization, run each capture in
a fresh bounded owned process, async first and sync only on success. Use an
absolute, unique git-excluded `.superpowers` attempt path, never published docs.
The command contract is exact (including argument order):

```sh
PYTORCH_VULKAN_ASYNC_EXECUTION=1 PYTHONPATH=.:build:tests/python .venv/bin/python tools/capture_vulkan_composed_evidence.py --mode async --device vk:0 --output "$ASYNC_ATTEMPT"
PYTORCH_VULKAN_ASYNC_EXECUTION=0 PYTHONPATH=.:build:tests/python .venv/bin/python tools/capture_vulkan_composed_evidence.py --mode sync --device vk:0 --output "$SYNC_ATTEMPT"
```

The recorder bootstraps before importing capture/reference helpers and uses their
existing public MSE/model interfaces. A receipt is written only after both capture
functions succeed. Partial attempts are retained as failures, not adopted. Actual
driver and recorder sources are part of the complete semantic dependency maps.
Receipts must match the actual commands, source inventory and loaded extension;
renaming the recorder invalidates prior receipts for current-source publication.
Do not change record schemas, identities or semantic assertions to admit stale
evidence.

Publication is a separate CPU-only step with exclusive writer ownership:

```sh
PYTHONPATH=.:build:tests/python .venv/bin/python tools/validate_vulkan_bootstrap_provenance.py --publish --async-attempt "$ASYNC_ATTEMPT" --sync-attempt "$SYNC_ATTEMPT" --archive "$IMMUTABLE_ARCHIVE"
.venv/bin/python tools/validate_vulkan_bootstrap_provenance.py
```

CPU-only publication is **not import-independent**: `--publish` uses the existing
capability generator and shared conformance/graph validators, which load upstream
PyTorch and the backend wrapper from `build`. The production generator imports
the wrapper before publication's CPU reference reverse validation. This import
registers the backend but does not execute a GPU workload; publication performs
CPU validation of existing captures, not recapture or GPU qualification.

The standalone provenance command (without `--publish`) remains independent of
upstream/backend imports and needs no `build` entry in `PYTHONPATH`. A missing or
stale sidecar must fail normally, not with a runtime module-import error. Keep
this standalone boundary separate from publication's shared graph-validation
runtime prerequisite; do not change numerical helpers or historical identities
to pretend publication has no runtime dependency.

It verifies receipts, exact inventories, archive hashes/payloads, unchanged
measurements beyond approved provenance, existing semantic validators, and the
complete candidate before publication. Unexpected numerical/counter/route/hardware
changes require investigation. Replacements roll back on Python exceptions;
multi-file replacement is **not crash-atomic**. Keep the immutable archive for
interruption recovery; do not publish concurrently.

`docs/vulkan_bootstrap_provenance.json` is generated only by actual two-mode
capture publication. Schema 1 binds the exact wrapper-first entry point, modes,
executed commands, driver/recorder/init/CMake/runner/validator hashes, loaded
extension identity, record IDs and published output hashes. It does not hash its
own payload. Missing, stale or wrong-entrypoint provenance fails independently of
the strict semantic evidence validators; neither gate replaces the other.

After publication, run CPU evidence checks before fresh composed-generation checks
in one wrapper-first process. The fresh-generation entry point is
`tests/python/test_vulkan_composed_generation.py`; it executes Vulkan captures,
unlike offline CPU replay. Then run complete async qualification and, only on
success, sync qualification. Use the authenticated complete shader profile from
[shader-toolchain.md](shader-toolchain.md); profile identity does not relabel old
capture hashes or establish GPU execution.
