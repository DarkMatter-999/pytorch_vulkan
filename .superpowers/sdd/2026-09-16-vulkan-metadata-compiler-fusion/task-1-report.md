# Task 1 Report

## Scope

Implemented Task 1 only. The existing Vulkan layout path already validated the
requested tensor metadata, allocation range, positive strides, device/layout,
and view aliasing before Vulkan consumers. The missing contract enforcement was
in explicit serialization: overlapping serialized views were accepted.

## Changes

- Added `tests/python/test_vulkan_metadata.py` covering contiguous F32 metadata,
  positive-stride nonzero-offset aliases, `view`/`reshape`/`as_strided`,
  overlapping and invalid-range rejection, negative strides, and unsupported
  dtypes.
- Updated `python/pytorch_vulkan/serialization.py` to classify serialized
  positive-stride layouts and reject overlapping or unclassifiable large views
  before encoding/decoding storage metadata.
- No changes were made to `docs/vulkan-fusion-compile-poc.md`.

## Validation

- `cmake -S . -B build -DBUILD_PYTHON_EXTENSION=ON`: passed; emitted the
  pre-existing `kineto_LIBRARY-NOTFOUND` warning.
- `cmake --build build --target pytorch_vulkan_python -j2`: passed.
- `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_metadata.py`:
  `7 passed`.
- `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_metadata.py tests/python/test_vulkan_linear.py tests/python/test_vulkan_training.py`:
  `61 passed, 49 warnings`.

The warnings are existing Vulkan `manual_seed` API warnings.
