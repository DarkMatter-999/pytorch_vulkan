# PyTorch Vulkan backend

This repository provides an experimental Vulkan backend for PyTorch. The
active, qualified surface is a Linux source build using the `vk:0` device and
the documented F32 operator subset. Unsupported Vulkan operations are rejected
explicitly; they do not silently fall back to CPU.

The reproducible source-build and extension verification procedure is in
[README-build.md](README-build.md).

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

Historical OpenCL and `dlprimitives` material remains in the checkout for
reference only. It is not the active backend, build path, or support contract.
