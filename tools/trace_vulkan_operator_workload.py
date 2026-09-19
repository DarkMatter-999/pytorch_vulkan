#!/usr/bin/env python3
"""Emit a deterministic CPU reference trace for a small operator workload."""

import argparse
import json
from pathlib import Path


def _tensor_metadata(tensor):
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
    }


def _backward(schema):
    return [{"schema": schema, "overload": "default"}]


def _linear_relu_training():
    import torch

    torch.manual_seed(0)
    inputs = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(3, dtype=torch.float32, requires_grad=True)

    linear = torch.nn.functional.linear(inputs, weight, bias)
    output = torch.relu(linear)
    output.sum().backward()

    return [
        {
            "schema": "aten::linear",
            "overload": "default",
            "inputs": [
                _tensor_metadata(inputs),
                _tensor_metadata(weight),
                _tensor_metadata(bias),
            ],
            "outputs": [_tensor_metadata(linear)],
            "backward": _backward("aten::linear_backward"),
        },
        {
            "schema": "aten::relu",
            "overload": "default",
            "inputs": [_tensor_metadata(linear)],
            "outputs": [_tensor_metadata(output)],
            "backward": _backward("aten::threshold_backward"),
        },
    ]


def _gemm_forward():
    import torch

    torch.manual_seed(1)
    left = torch.randn(2, 4, dtype=torch.float32)
    right = torch.randn(4, 3, dtype=torch.float32)
    value = torch.randn(2, 3, dtype=torch.float32)
    weight = torch.randn(3, 4, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)
    mm = torch.mm(left, right)
    addmm = torch.addmm(value, left, right)
    linear = torch.nn.functional.linear(left, weight, bias)

    def operator(schema, inputs, output):
        return {
            "schema": schema,
            "overload": "default",
            "execution": "cpu_schema_reference",
            "inputs": [_tensor_metadata(item) for item in inputs],
            "outputs": [_tensor_metadata(output)],
            "backward": [],
        }

    return [
        operator("aten::mm", (left, right), mm),
        operator("aten::addmm", (value, left, right), addmm),
        operator("aten::linear", (left, weight, bias), linear),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    if arguments.workload not in {"linear_relu_training", "gemm_forward"}:
        parser.error(f"unsupported workload: {arguments.workload}")

    operators = (
        _linear_relu_training()
        if arguments.workload == "linear_relu_training"
        else _gemm_forward()
    )
    trace = {
        "workload": arguments.workload,
        "operators": operators,
    }
    if arguments.workload == "gemm_forward":
        trace["execution"] = "cpu_schema_reference"
    arguments.output.write_text(json.dumps(trace, indent=2) + "\n")


if __name__ == "__main__":
    main()
