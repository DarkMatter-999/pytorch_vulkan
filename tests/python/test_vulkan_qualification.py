import json
from pathlib import Path

from tools import run_vulkan_qualification as qualification


def test_run_command_preserves_nonzero_exit_code(monkeypatch):
    class Completed:
        returncode = 7
        stdout = "out\n"
        stderr = "err\n"

    monkeypatch.setattr(qualification.subprocess, "run", lambda *args, **kwargs: Completed())

    result = qualification.run_command(["example", "--check"], Path("/tmp"))

    assert result["exit_code"] == 7
    assert result["stdout"] == "out\n"
    assert result["stderr"] == "err\n"


def test_classify_unavailable_is_skip_not_pass():
    result = qualification.classify_result(
        {"exit_code": 0, "stdout": "SKIPPED no suitable Vulkan device", "stderr": ""},
        device_dependent=True,
    )

    assert result == "skip"


def test_failed_output_with_unavailable_text_is_not_a_skip():
    result = qualification.classify_result(
        {"exit_code": 1, "stdout": "device unavailable\n", "stderr": ""},
        device_dependent=True,
    )

    assert result == "fail"


def test_qualification_artifact_schema_and_shape_consistency():
    artifact = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/vulkan-runtime-foundation-qualification.json")
        .read_text()
    )
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "qualified"
    assert artifact["performance_claims"] == "timing evidence only; no faster-than-CPU claim"
    assert set(artifact["workloads"]) == {"small", "medium", "large", "skinny", "irregular"}
    required = {
        "shape", "dispatches", "submissions", "completions", "waits", "timing_status",
        "timing_reason", "gpu_timestamp_sample_count", "gpu_timestamp_intervals_ns",
        "host_latency_ns", "cache_metrics", "resource_bounds",
    }
    for name, workload in artifact["workloads"].items():
        assert required <= workload.keys(), name
        assert workload["dispatches"] == workload["submissions"] == workload["completions"]
        assert workload["waits"] == workload["completions"]
        assert workload["gpu_timestamp_sample_count"] == len(workload["gpu_timestamp_intervals_ns"])
        assert len(workload["shape"]) == 3
        assert workload["resource_bounds"]["pipeline_pending_destructions_after"] == 0


def test_run_command_normalizes_bytes_timeout_output(monkeypatch):
    def timeout(*args, **kwargs):
        raise qualification.subprocess.TimeoutExpired(
            args[0], 300, output=b"partial-out", stderr=b"partial-err"
        )

    monkeypatch.setattr(qualification.subprocess, "run", timeout)

    result = qualification.run_command(["example"], Path("/tmp"))

    assert result == {
        "exit_code": 124,
        "stdout": "partial-out",
        "stderr": "partial-err\ncommand timed out after 300 seconds\n",
    }


def test_device_probe_validates_requested_device_and_preserves_evidence(monkeypatch):
    calls = []

    def fake_run(command, cwd, env=None):
        calls.append((command, env))
        return {"exit_code": 77, "stdout": "", "stderr": "device unavailable"}

    monkeypatch.setattr(qualification, "run_command", fake_run)

    probe = qualification.device_available(Path("build"), "vk:3")

    assert probe["status"] == "unavailable"
    assert probe["reason"] == "device unavailable"
    assert probe["exit_code"] == 77
    assert probe["stdout"] == ""
    assert probe["stderr"] == "device unavailable"
    assert calls[0][0][-1] == "vk:3"


def test_invalid_device_is_explicitly_unavailable(monkeypatch):
    probe = qualification.device_available(Path("build"), "cuda:0")

    assert probe["status"] == "unavailable"
    assert "must match vk:<index>" in probe["reason"]
    assert probe["exit_code"] == 2


def test_gate_order_excludes_qualification_runner(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, cwd, env=None):
        calls.append(command)
        return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )

    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)

    names = [gate["name"] for gate in summary["gates"]]
    assert names == [
        "manifest",
        "build",
        "ctest",
        "conformance",
        "shader_verification",
        "validation_layer",
        "stress",
        "timing",
        "benchmark",
        "environment",
        "additional_implementation",
    ]
    assert all("run_vulkan_qualification.py" not in part for command in calls for part in command)


def test_additional_device_runs_qualification_commands_with_device_environment(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(command, cwd, env=None):
        calls.append((command, env))
        return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda build, device, python=None: {
            "status": "available",
            "reason": f"{device} is available",
            "command": ["probe", device],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    summary = qualification.run_qualification(
        tmp_path / "report.json", run_build=False, additional_device="vk:1"
    )

    gate = summary["gates"][-1]
    assert gate["status"] == "pass"
    assert gate["device_probe"]["reason"] == "vk:1 is available"
    assert any("test_vulkan_operator_capabilities.py" in part for command, _ in calls for part in command)
    additional_envs = [env for command, env in calls if env and env.get("VULKAN_DEVICE") == "vk:1"]
    assert additional_envs
    assert all(env["PYTORCH_VULKAN_DEVICE"] == "vk:1" for env in additional_envs)


def test_report_retains_device_probe_evidence(monkeypatch, tmp_path):
    probe = {
        "status": "unavailable",
        "reason": "requested device is unavailable",
        "command": ["probe", "vk:9"],
        "exit_code": 77,
        "stdout": "probe-out",
        "stderr": "probe-err",
    }
    monkeypatch.setattr(qualification, "device_available", lambda *args, **kwargs: probe)
    monkeypatch.setattr(qualification, "run_command", lambda *args, **kwargs: {
        "exit_code": 0,
        "stdout": "ok\n",
        "stderr": "",
    })

    report = tmp_path / "report.json"
    summary = qualification.run_qualification(report, run_build=False)

    assert summary["device_probe"] == probe
    assert json.loads(report.read_text())["device_probe"]["stderr"] == "probe-err"


def test_python_and_artifact_paths_are_configurable(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, cwd, env=None):
        commands.append(command)
        return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )
    artifacts = tmp_path / "evidence"
    qualification.run_qualification(
        tmp_path / "report.json",
        run_build=False,
        python_executable=Path("/custom/python"),
        artifacts=artifacts,
    )

    assert artifacts.is_dir()
    assert all(command[0] != str(qualification.DEFAULT_PYTHON) for command in commands)
    assert any(command[0] == "/custom/python" for command in commands)


def test_validation_ctest_registration_sets_validation_layer():
    cmake = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
    text = cmake.read_text()

    assert "vulkan_supported_workload_validation" in text
    assert "VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation" in text


def test_command_records_effective_environment_deltas(monkeypatch, tmp_path):
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )
    monkeypatch.setattr(
        qualification,
        "run_command",
        lambda *args, **kwargs: {"exit_code": 0, "stdout": "", "stderr": ""},
    )

    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)
    validation = next(gate for gate in summary["gates"] if gate["name"] == "validation_layer")
    environment = validation["commands"][0]["environment"]

    assert environment["VK_INSTANCE_LAYERS"] == "VK_LAYER_KHRONOS_validation"
    assert environment["VULKAN_DEVICE"] == "vk:0"
    assert environment["PYTORCH_VULKAN_DEVICE"] == "vk:0"


def test_report_contains_machine_readable_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(qualification, "run_command", lambda *args, **kwargs: {
        "exit_code": 0,
        "stdout": "ok\n",
        "stderr": "",
    })
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )

    report = tmp_path / "report.json"
    summary = qualification.run_qualification(report, run_build=False)

    assert json.loads(report.read_text()) == summary
