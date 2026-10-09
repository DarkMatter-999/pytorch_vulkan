"""Real wrapper-driver collection checks; no GPU test bodies are executed."""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = {
    'vulkan_supported_workload_validation': ['-q', '-rs', str(ROOT / 'tests/python/test_vulkan_conformance.py')],
    'vulkan_runtime_foundation_validation': ['-q', str(ROOT / 'tests/python/test_vulkan_reliability.py')
                                            + '::test_mixed_workload_validation_layer_gate'],
    'vulkan_python_suite': ['-q', '-rs', str(ROOT / 'tests/python')],
}


@pytest.mark.parametrize('name', ENTRIES)
def test_ctest_environment_collects_with_real_wrapper_driver(name):
    # Read the production property, not an independently repaired test environment.
    cmake = (ROOT / 'CMakeLists.txt').read_text()
    properties = re.search(r'set_tests_properties\(' + name + r'\s+PROPERTIES(.*?)\)',
                           cmake, re.DOTALL).group(1)
    environment = re.search(r'ENVIRONMENT "([^"]+)"', properties).group(1)
    environment = environment.replace('${CMAKE_BINARY_DIR}', str(ROOT / 'build'))
    environment = environment.replace('${CMAKE_CURRENT_SOURCE_DIR}', str(ROOT))
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)  # Never inherit the outer pytest's source path.
    env.update(item.split('=', 1) for item in environment.split(';'))
    env['TMPDIR'] = '/tmp/opencode'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools/vulkan_wrapper_pytest.py'),
         *ENTRIES[name], '--collect-only'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'collected' in result.stdout
    assert 'ModuleNotFoundError' not in result.stdout + result.stderr
    assert env['PYTHONPATH'].split(os.pathsep) == [
        str(ROOT), str(ROOT / 'build'), str(ROOT / 'tests/python')]
    if name != 'vulkan_python_suite':
        assert env['VK_INSTANCE_LAYERS'] == 'VK_LAYER_KHRONOS_validation'
    if name == 'vulkan_runtime_foundation_validation':
        assert 'SKIP_RETURN_CODE 77' in properties
