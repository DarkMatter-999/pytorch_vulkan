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

### Explicit transfer artifact transition

Without an explicit transition, publication still rejects a changed extension SHA.
The contiguous same-dtype Bool transfer change can be published only with a
separately approved JSON input, retained under `.superpowers`, passed by adding
`--artifact-transition "$APPROVED_TRANSITION"` to the publication command above.
This is not a default bypass or permission to alter numerical evidence. Obtain
tooling review before fresh captures; old receipts become stale when capture-control
sources change. Construct the transition from the immutable baseline and the
**actual completed** async/sync receipts, never from invented future identities.

The input has exactly these fields:

```json
{
  "schema_version": 1,
  "kind": "contiguous-bool-transfer",
  "baseline_manifest_sha256": "SHA256 of exact archived manifest.json bytes",
  "old_extension_sha256": "extension SHA in all 198 archived MSE/model records",
  "new_extension": {"path": "absolute actual extension path", "sha256": "actual binary SHA256"},
  "source_delta": {"src/vulkan_transfer.cpp": {"before": "archived source SHA256", "after": "current source SHA256"}},
  "receipt_sha256": {"async": "SHA256 of raw async receipt.json", "sync": "SHA256 of raw sync receipt.json"},
  "capture_source_identity": {"each of the six capture-control paths": "current SHA256, matching both receipts"}
}
```

This is a descriptive template, not executable evidence; every hash must be a
64-character lowercase hexadecimal value. Capture-control paths are exactly
`IDENTITY_PATHS` in `tools/validate_vulkan_bootstrap_provenance.py`. The input binds
their current contents, including the reviewed publication policy, separately
from the semantic source maps in the captured records.

The guard authenticates the baseline manifest/files/per-record payloads against
unchanged published bytes, rereads both complete receipts and outputs, checks the
actual current extension, and requires exactly the one approved transfer source
delta in every MSE/model map. All other mapped source hashes must match actual
source. Both modes must use the same artifact and runtime identity except mode.
Only the verified old-to-new extension SHA pair is exempted from measurement
preservation; driver, instance, hardware, device/API/reference/mode, numerical
values, gradients, routes and counters must remain unchanged. Other source changes
or changed measurements require a separate design/investigation, not a broader
allowlist. The guard rechecks the input/baseline/receipts/sources before staging.

The 90 async MSE, 90 sync MSE and 18 connected-model active records are replaced
only by genuine new captures. All historical archive bytes and metadata remain
intact, and unrelated vector/general-matmul/Linear/workload payloads stay exact. No capture helper,
historical source map or binary identity is rehashed to make old execution current.
Temporary CPU test fixtures exercising publication are not GPU capture receipts.

`docs/vulkan_bootstrap_provenance.json` is generated only by actual two-mode
capture publication. Schema 1 binds the exact wrapper-first entry point, modes,
executed commands, driver/recorder/init/CMake/runner/validator hashes, loaded
extension identity, record IDs and published output hashes. It does not hash its
own payload. Missing, stale or wrong-entrypoint provenance fails independently of
the strict semantic evidence validators; neither gate replaces the other.
When an artifact transition is used, schema 1 additionally retains the exact
`artifact_transition` document in the sidecar for audit. Its current controls,
extension and receipt hashes/commands are checked independently; retain the raw
excluded attempts and approval input. Ordinary schema-1 sidecars without a
transition retain their original validation contract.

After publication, run CPU evidence checks before fresh composed-generation checks
in one wrapper-first process. The fresh-generation entry point is
`tests/python/test_vulkan_composed_generation.py`; it executes Vulkan captures,
unlike offline CPU replay. Then run complete async qualification and, only on
success, sync qualification. Use the authenticated complete shader profile from
[shader-toolchain.md](shader-toolchain.md); profile identity does not relabel old
capture hashes or establish GPU execution.
