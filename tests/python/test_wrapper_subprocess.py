"""CPU fixtures for bounded, owned lifecycle subprocesses (no backend import)."""

import os
from pathlib import Path
import subprocess
import time

import pytest


def _stopped(pid):
    path = Path(f"/proc/{pid}/stat")
    return not path.exists() or path.read_bytes().rsplit(b") ", 1)[1].split()[0] in {b"Z", b"X"}


def test_success_retains_output_and_environment():
    from wrapper_subprocess import run_script

    result = run_script("import os, sys; print(os.environ['WRAPPER_FIXTURE']); print('err', file=sys.stderr)",
                        {**os.environ, "WRAPPER_FIXTURE": "fixture"})
    assert result.returncode == 0
    assert result.stdout == "fixture\n"
    assert result.stderr == "err\n"


def test_failure_retains_exit_and_trace():
    from wrapper_subprocess import run_script

    result = run_script("print('before', flush=True); raise RuntimeError('fixture failure')", dict(os.environ))
    assert result.returncode == 1
    assert result.stdout == "before\n"
    assert "RuntimeError: fixture failure" in result.stderr


@pytest.mark.parametrize("close_pipes", [False, True])
def test_timeout_reaps_leader_and_stops_term_resistant_descendant(close_pipes):
    from wrapper_subprocess import run_script

    child = ("import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "print('ready', flush=True); "
             + ("os.close(1); os.close(2); " if close_pipes else "") + "time.sleep(60)")
    script = f"""
import os, subprocess, sys, time
p = subprocess.Popen([sys.executable, '-c', {child!r}], stdout=subprocess.PIPE, text=True)
assert p.stdout.readline().strip() == 'ready'
print(os.getpid(), p.pid, flush=True)
print('timeout stderr', file=sys.stderr, flush=True)
{'p.stdout.close()' if close_pipes else ''}
time.sleep(60)
"""
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_script(script, dict(os.environ), timeout=0.5)
    assert time.monotonic() - start < 9
    error = caught.value
    leader, child_pid = map(int, error.stdout.strip().split())
    assert error.stderr == "timeout stderr\n"
    assert not Path(f"/proc/{leader}").exists(), "direct child was not reaped"
    assert _stopped(child_pid), "owned descendant still executes"


def test_success_also_stops_descendant_with_closed_output():
    from wrapper_subprocess import run_script

    result = run_script("""
import subprocess, sys
p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(p.pid, flush=True)
""", dict(os.environ))
    assert result.returncode == 0
    assert _stopped(int(result.stdout))
