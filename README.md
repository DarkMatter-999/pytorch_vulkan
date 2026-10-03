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
  'import torch, pytorch_vulkan; print(torch.__version__, pytorch_vulkan.is_available()); print(torch.device("vk:0"))'
PYTHONPATH=build .venv/bin/python -m pytest -q tests/python
```

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
