"""Fresh Python processes with bounded output and owned-session cleanup."""

import signal
import subprocess
import sys
from pathlib import Path

from tools.run_vulkan_qualification import _finish_owned_group, _signal_owned_group


ROOT = Path(__file__).resolve().parents[2]


def run_script(script: str, env: dict[str, str], timeout: float = 180):
    command = [sys.executable, "-c", script]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _signal_owned_group(process, signal.SIGTERM)
        try:
            process.communicate(timeout=4)
        except subprocess.TimeoutExpired:
            pass
    finally:
        # The approved runner kills even closed-pipe, TERM-resistant descendants,
        # drains with bounds, reaps the leader and verifies no live group members.
        stdout, stderr = _finish_owned_group(process)
    def text(value):
        return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout, output=text(stdout), stderr=text(stderr))
    return subprocess.CompletedProcess(command, process.returncode, text(stdout), text(stderr))
