# PyTorch Vulkan backend

This repository provides an experimental Vulkan backend for PyTorch. The
active, qualified surface is a Linux source build using the `vk:0` device and
the documented F32 operator subset. Unsupported Vulkan operations are rejected
explicitly; they do not silently fall back to CPU.

The reproducible source-build and extension verification procedure is in
[README-build.md](README-build.md).

## Development direction

The goal is CPU/CUDA-comparable functionality through the ordinary PyTorch API.
Backend development follows the supported PyTorch version's schemas,
implementations and autograd rules, reusing its generated/composite machinery
and implementing the required Vulkan primitives. See the
[PyTorch-reference-first development standard](docs/backend-development-standard.md).

This is the project direction, not a claim that the current experimental subset
already provides full compatibility.

## Quick Start

Use the existing `build/` directory or follow the complete source-build
procedure in [README-build.md](README-build.md). The supported configuration is
Linux, PyTorch 2.4.0, Vulkan headers and loader, and the Vulkan validation
layer. The extension is built from source; this project makes no wheel or
package availability claim.

After building, verify the active extension and device with:

```bash
PYTHONPATH=build .venv/bin/python -c \
  'import pytorch_vulkan as torch; print(torch.version.__version__, torch.is_available()); print(torch.device("vk:0"))'
PYTHONPATH=.:build .venv/bin/python tools/vulkan_wrapper_pytest.py -q tests/python
```

Use the wrapper as the application entry point, **before any autograd execution**
(including backward run by another library):

```python
import pytorch_vulkan as torch

model = torch.nn.Linear(4, 2).to("vk:0")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Upstream PyTorch 2.4.0 remains a required dependency, not a replacement runtime.
Public APIs such as `Tensor`, `tensor`, dtypes, `nn`, `optim`, and `autograd` resolve
to upstream objects; existing Vulkan optimizer validation remains installed.
Package-owned device/compiler helpers and `save`/`load` take precedence. In
particular, the latter retain the package's explicit Vulkan serialization format,
not generic upstream `torch.save`/`torch.load` semantics. Use `torch.nn` through
the facade; arbitrary `from pytorch_vulkan.nn import ...` imports are not promised.
Private upstream names are not forwarded, and `sys.modules['torch']` is unchanged.
The dependency version is available as `torch.version.__version__`.

Successful wrapper import registers the extension guard and `vk` device module
before returning, without an artificial backward or Vulkan tensor allocation.
It does not establish device availability: check `torch.is_available()` before
using `vk:0`; unavailable devices do not silently fall back to CPU. CPU APIs
retain upstream behavior, while Vulkan execution is limited to the documented
operator/layout/dtype/autograd contracts.

The legacy `import torch; import pytorch_vulkan` style remains compatible when
no autograd has executed beforehand. Importing upstream torch alone first is
allowed; importing the wrapper after a CPU backward is outside the supported
startup boundary because the shared-engine initialization limitation remains
unresolved. This facade does not repair or reliably detect that late-import case,
and does not qualify arbitrary higher-order AD, forward AD, or transforms.

The full Python CTest suite and qualification runner use that explicit
wrapper-first driver; standalone CPU evidence validators do not import the
backend. [Bootstrap evidence policy](docs/vulkan-bootstrap-evidence.md) separates
immutable historical measurements from current-source startup qualification.
A missing/stale bootstrap provenance sidecar is a failing gate, not permission
to reuse old hashes.

The active capability sources are the authority for what passes this gate:

- [operator capability matrix](docs/vulkan_operator_capability_matrix.md)
- [machine-readable capability manifest](docs/vulkan_capabilities.json)
- [application workload evidence](docs/vulkan_workload_coverage.json)

The six application records qualify only the fixed three-update grouped/depthwise
CrossEntropy classifier using stock momentum SGD with both `zero_grad(set_to_none=True)`
and `zero_grad(set_to_none=False)`, plus a separate two-convolution output-energy
parameter HVP. They apply to the exact PyTorch build, extension build, Renoir
hardware/driver, and asynchronous/synchronized execution modes recorded in the
workload artifact. They do not establish whole-classifier HVP, arbitrary-order
reverse AD, forward AD/JVP or transforms, CUDA, or other unexecuted workloads or
devices. The operator capability manifest remains separate evidence for operator
contracts, not proof of these application results.

The repository's active backend is Vulkan; its build and support contract are
documented above.
