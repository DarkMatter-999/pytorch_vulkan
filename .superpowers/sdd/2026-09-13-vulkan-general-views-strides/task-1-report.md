# Task 1 Report: Layout And Alias Contracts

## Status

Implemented and committed in `ce57e93` (`feat: lock down Vulkan layout descriptor contract`).

## Files changed

- `src/vulkan_layout.h`: expanded `VulkanTensorLayout` with rank, copied sizes and
  strides, scalar type, element bytes, and internal-overlap classification; added
  checked logical-coordinate and linear-index mapping APIs.
- `src/vulkan_layout.cpp`: added checked address mapping and nonnegative-stride
  overlap classification, while retaining checked byte/range/allocation validation.
- `tests/cpp/vulkan_transfer_tensor_test.cpp`: added native descriptor, transpose,
  zero-stride, mapping-boundary, and invalid-index coverage.
- `tests/python/test_vulkan_view.py`: added Vulkan-resident transpose, slice,
  zero-stride, and empty-view metadata coverage.

## Tests

Commands and results:

```text
cmake --build build --target vulkan_transfer_tensor_test pytorch_vulkan_python -j2
... Built target vulkan_transfer_tensor_test
... Built target pytorch_vulkan_python

ctest --test-dir build --output-on-failure -R vulkan_transfer_tensor_test
1/1 Test #9: vulkan_transfer_tensor_test ......   Passed
100% tests passed out of 1

PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_view.py
......                                                                   [100%]
6 passed in 1.12s

git diff --check
(no output; passed)
```

The native test executable also passed directly with:

```text
./vulkan_transfer_tensor_test
Vulkan tensor transfer tests passed
```

## Concerns

- The first unqualified `python` invocation could not import Torch; the focused
  test was rerun with the project `.venv` and built extension via `PYTHONPATH=build`.
- Non-contiguous Vulkan readback remains intentionally unsupported at this stage;
  Python coverage therefore checks view metadata and storage identity only. Operator
  execution migration is deferred to later tasks.
