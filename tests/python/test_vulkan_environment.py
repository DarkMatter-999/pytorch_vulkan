import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.record_vulkan_environment import _supported_torch_version


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "record_vulkan_environment.py"


def _record_environment(*extra_args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *extra_args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_environment_recorder_writes_requested_output_file():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "environment.json"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--output", str(output)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert output.exists()
        assert json.loads(output.read_text()) == json.loads(result.stdout)


def test_environment_record_contains_versions_and_repository_provenance():
    record = _record_environment()

    assert record["schema_version"] == 1
    assert record["repository"]["git_root"] == str(ROOT)
    assert record["repository"]["revision"]
    assert isinstance(record["repository"]["submodules"], list)
    assert record["python"]["version"]
    assert record["torch"]["status"] == "available"
    assert record["torch"]["version"].split("+", 1)[0] == "2.4.0"
    assert record["torch"]["supported_2_4"] is True
    assert record["tools"]["cmake"]["status"] in {"available", "unavailable"}
    assert record["tools"]["vulkaninfo"]["status"] in {"available", "unavailable"}


def test_environment_record_reports_unavailable_optional_command():
    record = _record_environment("--probe-command", "command-that-does-not-exist")

    assert record["probes"]["command-that-does-not-exist"]["status"] == "unavailable"
    assert record["probes"]["command-that-does-not-exist"]["error"]


def test_supported_torch_version_requires_exact_2_4_0_base():
    assert _supported_torch_version("2.4.0") is True
    assert _supported_torch_version("2.4.0+cpu") is True
    assert _supported_torch_version("2.4.1") is False
    assert _supported_torch_version("2.4.0a0") is False
