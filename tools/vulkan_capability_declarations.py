"""Hand-maintained Vulkan capability declarations.

Every field the generator cannot derive. Keyed by aten schema.

The generator derives `schema`, `test_cases`, `tests` and `shape_constraints`
from the C++ registrations and the conformance registry; everything here is
a human judgement or a semantic fact about an implementation body.

This module must not import vulkan_conformance or the generated manifest:
the generator imports it, and that would close an import cycle.
"""

from __future__ import annotations

DECLARATIONS: dict[str, dict[str, object]] = {
    'aten::_adaptive_avg_pool2d.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 4
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_adaptive_avg_pool2d_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 4
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::concat.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::_copy_from.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "zero_size_noop",
        "execution_contract": "vulkan_copy",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_copy_from_and_resize.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::_local_scalar_dense.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::_log_softmax.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_log_softmax.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_log_softmax_backward_data.out':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_native_multi_head_attention.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::_native_multi_head_attention.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::_reshape_alias.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_view_alias",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "metadata_only",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_softmax.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_softmax.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_softmax_backward_data.out':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_to_copy.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "zero_size_noop",
        "execution_contract": "vulkan_copy",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::_transform_bias_rescale_qkv.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::_upsample_nearest_exact2d.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::_upsample_nearest_exact2d_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::abs.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::abs.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["float32"], "outputs": ["float32"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::abs_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "rejected_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "explicit_source_rejection",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "rejected"
    },
    'aten::add.Scalar':     {
        "aliasing": "no_overlap",
        "autograd": "reverse_second_order_witnessed",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::add.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::add.Tensor':     {
        "aliasing": "no_overlap",
        "autograd": "reverse_second_order_witnessed",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::add.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::add_.Scalar':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::add_.Tensor':     {
        "aliasing": "same_storage_alias",
        "autograd": "optimizer_update",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::addcdiv.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::addcdiv_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "optimizer_update",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::addcmul.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::addcmul_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "optimizer_update",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::addmm.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::addmm.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::amax.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::amax.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::amin.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::amin.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::arange.start_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::argmax.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_differentiable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "int64"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::argmax.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["float32"], "outputs": ["int64"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::as_strided.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_view_alias",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "metadata_only",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::atan.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::avg_pool2d.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 4,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::avg_pool2d_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::bernoulli_.float':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::binary_cross_entropy.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::binary_cross_entropy_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::binary_cross_entropy_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::bitwise_and.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["bool"], "outputs": ["bool"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::bitwise_not.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::bitwise_or.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::bitwise_xor.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::bmm.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "transposed-contiguous",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 3,
            "min": 3
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::bmm.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::cat.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::ceil.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::ceil.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::clamp.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::clamp_min.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    # Scalar restrictions are enforced in src/vulkan/operators/convolution.cpp:172-174
    # and :209-211; the manifest schema cannot express them (scalar_constraints
    # is limited to none/scalar_supported, and shape_constraints is shape-only).
    'aten::convolution.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 4
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::convolution_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 4
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::convolution_backward_overrideable.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::convolution_overrideable.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::copy_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "zero_size_noop",
        "execution_contract": "vulkan_copy",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::div.Tensor':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::div.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::dot.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::empty.memory_format':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "zero_size_noop",
        "execution_contract": "vulkan_copy",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::empty_strided.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::eq.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::eq.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["float32"], "outputs": ["bool"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::exp.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::fill_.Scalar':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::ge.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::ge.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::gelu.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::gelu.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::gelu_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::gt.Scalar':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::gt.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::gt.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::hardsigmoid.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::hardsigmoid_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::hardswish_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::hardswish_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::hardtanh.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::hardtanh_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::hardtanh_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::isfinite.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {"index": 0, "type": "PrivateUse1"},
        "dtypes": {"inputs": ["float32"], "outputs": ["bool"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": ["strided", "non-overlapping"],
        "out": "not_applicable",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": ["none"],
        "status": "supported"
    },
    'aten::isfinite.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "rejected_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "schema_absent_from_pytorch_dispatcher",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "rejected"
    },
    'aten::le.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::le.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::leaky_relu.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::leaky_relu_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::lerp.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::lerp_.Scalar':     {
        "aliasing": "same_storage_alias",
        "autograd": "optimizer_update",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::linear.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 3,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::log.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::log_sigmoid_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::log_sigmoid_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::log_sigmoid_forward.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::log_sigmoid_forward.output':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::logit.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::logit.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::lt.Scalar':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::lt.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::lt.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::masked_select.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_copy_then_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [
            "vulkan_1_2_8bit_storage_int8"
        ],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::max.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::max_pool2d_with_indices.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 4,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::maximum.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::mean.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::mean.dim':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::mean.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_differentiable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::min.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::minimum.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::mm.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::mm.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::mse_loss.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::mse_loss_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::mul.Scalar':     {
        "aliasing": "no_overlap",
        "autograd": "reverse_second_order_witnessed",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::mul.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::mul.Tensor':     {
        "aliasing": "no_overlap",
        "autograd": "reverse_second_order_witnessed",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::mul.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::mul_.Scalar':     {
        "aliasing": "same_storage_alias",
        "autograd": "optimizer_update",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::mul_.Tensor':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::native_batch_norm.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::native_batch_norm_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::native_dropout.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::native_dropout_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::native_layer_norm.default':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::native_layer_norm_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::ne.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["float32"], "outputs": ["bool"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::ne.Tensor':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["float32"], "outputs": ["bool"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::ne.Tensor_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::neg.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::neg.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["float32"], "outputs": ["float32"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::neg_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "rejected_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "explicit_source_rejection",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "rejected"
    },
    'aten::nll_loss_backward.default':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::nll_loss_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32",
                "int64"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::nll_loss_forward.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::nll_loss_forward.output':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32",
                "int64"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping",
            "zero-offset"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::normal_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::pow.Tensor_Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "deferred"
    },
    'aten::prod.dim_int':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::prod.int_out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::reciprocal.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::relu.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::relu.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {"inputs": ["float32"], "outputs": ["float32"]},
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {"max": 2, "min": 2},
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::relu_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "rejected_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "explicit_source_rejection",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "rejected"
    },
    'aten::reshape.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_view_alias",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::resize_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::round.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::rsub.Scalar':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::rsub.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::set_.source_Storage':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::set_.source_Storage_storage_offset':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::sgn.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::sigmoid.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::sigmoid.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::sigmoid_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::sigmoid_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::silu.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::silu_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::sqrt.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::stack.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_rejected",
        "execution_contract": "vulkan_copy",
        "inplace": "not_applicable",
        "layouts": [
            "contiguous",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::sub.Scalar':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::sub.Scalar_out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::sub.Tensor':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::sub.out':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::sub_.Scalar':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "scalar_supported"
        ],
        "status": "supported"
    },
    'aten::sub_.Tensor':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::sum.IntList_out':     {
        "aliasing": "no_overlap",
        "autograd": "not_differentiable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::sum.default':     {
        "aliasing": "no_overlap",
        "autograd": "reverse_second_order_witnessed",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::sum.dim_IntList':     {
        "aliasing": "no_overlap",
        "autograd": "reverse_second_order_witnessed",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "reduction_identity_or_nan",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 3,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::tanh.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_or_none",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::tanh.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::tanh_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::tanh_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "backward_kernel",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::threshold_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::uniform_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::upsample_bilinear2d.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::upsample_bilinear2d_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::upsample_nearest2d.out':     {
        "aliasing": "no_overlap",
        "autograd": "not_applicable",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "contiguous_out_required",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::upsample_nearest2d_backward.grad_input':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_backward",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_deferred",
        "execution_contract": "deferred_before_vulkan",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 8,
            "min": 0
        },
        "reason": "deferred_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "deferred"
    },
    'aten::view.default':     {
        "aliasing": "no_overlap",
        "autograd": "first_order_view_alias",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "metadata_only",
        "inplace": "not_applicable",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 2,
            "min": 2
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
    'aten::zero_.default':     {
        "aliasing": "same_storage_alias",
        "autograd": "optimizer_update",
        "device": {
            "index": 0,
            "type": "PrivateUse1"
        },
        "dtypes": {
            "inputs": [
                "float32"
            ],
            "outputs": [
                "float32"
            ]
        },
        "empty": "empty_output_supported",
        "execution_contract": "vulkan_compute",
        "inplace": "validated_exact_alias_inplace",
        "layouts": [
            "strided",
            "non-overlapping"
        ],
        "out": "not_applicable",
        "ranks": {
            "max": 1,
            "min": 1
        },
        "reason": "supported_contract",
        "required_vulkan_features": [],
        "scalar_constraints": [
            "none"
        ],
        "status": "supported"
    },
}

__all__ = ["DECLARATIONS"]
