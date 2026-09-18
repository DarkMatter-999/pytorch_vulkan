import json
import subprocess
import sys
from pathlib import Path


def test_linear_relu_training_trace_has_stable_operator_metadata(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    artifact = tmp_path / "linear_relu_training.json"
    result = subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "trace_vulkan_operator_workload.py"),
            "--workload",
            "linear_relu_training",
            "--output",
            str(artifact),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    trace = json.loads(artifact.read_text())
    assert list(trace) == ["workload", "operators"]
    assert trace["workload"] == "linear_relu_training"

    operators = trace["operators"]
    assert [operator["schema"] for operator in operators] == [
        "aten::linear",
        "aten::relu",
    ]
    assert [list(operator) for operator in operators] == [
        ["schema", "overload", "inputs", "outputs", "backward"],
        ["schema", "overload", "inputs", "outputs", "backward"],
    ]

    for operator in operators:
        assert operator["overload"] == "default"
        assert operator["inputs"]
        assert operator["outputs"]
        assert operator["backward"]
        for tensor in [*operator["inputs"], *operator["outputs"]]:
            assert list(tensor) == ["dtype", "device", "shape", "stride"]
            assert tensor["dtype"] == "float32"
            assert tensor["device"] == "cpu"
            assert isinstance(tensor["shape"], list)
            assert isinstance(tensor["stride"], list)
        assert all(
            list(backward) == ["schema", "overload"]
            for backward in operator["backward"]
        )
