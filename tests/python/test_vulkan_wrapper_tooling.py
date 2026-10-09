"""CPU-only tooling fixtures; none are Vulkan execution evidence."""
import copy
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    assert (ROOT / 'tools' / (name + '.py')).exists(), 'approved tool not implemented'
    return importlib.import_module('tools.' + name)


@pytest.mark.parametrize('exit_code', [0, 7])
def test_driver_bootstraps_before_pytest_import_and_forwards_exact_args(tmp_path, exit_code):
    driver = ROOT / 'tools/vulkan_wrapper_pytest.py'
    assert driver.exists(), 'wrapper-first driver missing'
    (tmp_path / 'pytorch_vulkan.py').write_text("import os\nos.environ['CPU_FIXTURE_BOOTSTRAPPED'] = 'yes'\n")
    (tmp_path / 'pytest.py').write_text(
        "import os, json\nassert os.environ.get('CPU_FIXTURE_BOOTSTRAPPED') == 'yes'\n"
        "def main(args):\n print(json.dumps(args))\n return " + str(exit_code) + '\n')
    args = ['-q', '-rs', 'tests/python', '-k', 'a or b', '--maxfail=1']
    result = subprocess.run([sys.executable, str(driver), *args], cwd=tmp_path,
                            env=os.environ | {'PYTHONPATH': str(tmp_path)},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout) == args


def test_recorder_imports_wrapper_before_loading_capture_helpers(monkeypatch, tmp_path):
    recorder = module('capture_vulkan_composed_evidence')
    order = []
    wrapper = SimpleNamespace(_C=SimpleNamespace(execution_mode=lambda: 'async'))
    original = recorder.importlib.import_module

    def load(name):
        order.append(name)
        if name == 'pytorch_vulkan':
            return wrapper
        if name == 'mse_capability_evidence':
            assert order[0] == 'pytorch_vulkan'
            return SimpleNamespace(capture_all_mse_cases=lambda **kw: {'CPU-only': kw})
        if name == 'composed_matmul_evidence':
            assert order[0] == 'pytorch_vulkan'
            return SimpleNamespace(capture_composed_evidence=lambda **kw: {'CPU-only': kw})
        return original(name)

    monkeypatch.setattr(recorder.importlib, 'import_module', load)
    mse, models = recorder.capture('async', 'vk:0')
    assert order[:3] == ['pytorch_vulkan', 'mse_capability_evidence', 'composed_matmul_evidence']
    assert mse == models == {'CPU-only': {'device': 'vk:0'}}
    with pytest.raises(ValueError, match='mode'):
        recorder.capture('sync', 'vk:0')
    with pytest.raises(ValueError, match='device'):
        recorder.capture('async', 'vk:1')


def test_recorder_refuses_existing_or_nonexcluded_output(tmp_path):
    recorder = module('capture_vulkan_composed_evidence')
    with pytest.raises(ValueError, match='excluded'):
        recorder.new_attempt(tmp_path, ROOT)


def cpu_provenance_fixture(tmp_path, validator):
    # Synthetic minimal document used only to test strict provenance mechanics.
    # Never written at the published repository paths or labelled GPU proof.
    for name in validator.IDENTITY_PATHS:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'CPU-only source fixture')
    extension = tmp_path / 'build/CPU-only-extension.so'
    extension.parent.mkdir(exist_ok=True)
    extension.write_bytes(b'CPU-only extension fixture')
    docs = {name: b'CPU-only output fixture' for name in validator.OUTPUT_PATHS}
    for name, data in docs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    document = {
        'schema_version': 1, 'entry_point': 'wrapper-first',
        'source_identity': validator.identity(tmp_path),
        'extension': {'path': str(extension), 'sha256': validator.digest(extension.read_bytes())},
        'outputs': {name: validator.digest(data) for name, data in docs.items()},
        'captures': {mode: {
            'command': ['/CPU-only/python', str(tmp_path / 'tools/capture_vulkan_composed_evidence.py'),
                        '--mode', mode, '--device', 'vk:0', '--output', '/CPU-only/' + mode],
            'mse_record_ids': validator.mse_ids(), 'model_record_ids': validator.model_ids(mode),
        } for mode in ('async', 'sync')},
    }
    return document


def test_provenance_missing_is_failure_and_validator_never_imports_wrapper(tmp_path):
    validator = module('validate_vulkan_bootstrap_provenance')
    with pytest.raises(ValueError, match='missing'):
        validator.validate_file(tmp_path / 'missing.json', tmp_path)
    source = (ROOT / 'tools/validate_vulkan_bootstrap_provenance.py').read_text()
    assert 'import pytorch_vulkan' not in source


@pytest.mark.parametrize('mutation', ['entrypoint', 'source', 'output', 'ids', 'mode', 'command', 'extension', 'extra'])
def test_provenance_rejects_wrong_stale_or_tampered_cpu_fixture(tmp_path, mutation):
    validator = module('validate_vulkan_bootstrap_provenance')
    document = cpu_provenance_fixture(tmp_path, validator)
    validator.validate_document(document, tmp_path)
    if mutation == 'entrypoint': document['entry_point'] = 'cpu-first'
    if mutation == 'source': document['source_identity'][validator.IDENTITY_PATHS[0]] = '0' * 64
    if mutation == 'output': document['outputs'][validator.OUTPUT_PATHS[0]] = '0' * 64
    if mutation == 'ids': document['captures']['async']['mse_record_ids'].pop()
    if mutation == 'mode': document['captures']['other'] = document['captures'].pop('sync')
    if mutation == 'command': document['captures']['async']['command'][1] = 'pytest'
    if mutation == 'extension': document['extension']['sha256'] = '0' * 64
    if mutation == 'extra': document['fake_proof'] = True
    with pytest.raises(ValueError):
        validator.validate_document(document, tmp_path)


def test_publication_validation_failure_preserves_all_original_bytes(tmp_path):
    validator = module('validate_vulkan_bootstrap_provenance')
    files = {'a.json': b'original-a', 'b.json': b'original-b'}
    for name, data in files.items(): (tmp_path / name).write_bytes(data)
    def reject(): raise ValueError('injected validation failure')
    with pytest.raises(ValueError, match='validation'):
        validator.publish_bytes(tmp_path, {name: b'candidate' for name in files}, reject)
    assert {name: (tmp_path / name).read_bytes() for name in files} == files


def test_publication_write_failure_rolls_back_multiple_documents(tmp_path, monkeypatch):
    validator = module('validate_vulkan_bootstrap_provenance')
    files = {'a.json': b'original-a', 'b.json': b'original-b'}
    for name, data in files.items(): (tmp_path / name).write_bytes(data)
    replace = validator.os.replace
    calls = []
    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2: raise OSError('injected write failure')
        return replace(source, destination)
    monkeypatch.setattr(validator.os, 'replace', fail_second)
    with pytest.raises(OSError, match='write failure'):
        validator.publish_bytes(tmp_path, {name: b'candidate' for name in files}, lambda: None)
    assert {name: (tmp_path / name).read_bytes() for name in files} == files


def test_runner_and_ctest_explicit_driver_and_independent_provenance_gate(monkeypatch, tmp_path):
    from tools import run_vulkan_qualification as runner
    calls = []
    monkeypatch.setattr(runner, 'device_available', lambda *a: {'status': 'available', 'reason': 'CPU fixture'})
    def command(argv, *a, **kw):
        calls.append(argv)
        return {'exit_code': 1 if 'tools/validate_vulkan_bootstrap_provenance.py' in argv else 0,
                'stdout': '', 'stderr': '', 'timeout_seconds': kw.get('timeout_seconds', 300)}
    monkeypatch.setattr(runner, 'run_command', command)
    report = runner.run_qualification(tmp_path / 'report.json', run_build=False, additional_device='vk:1')
    assert report['entry_point'] == 'wrapper-first'
    gate = next(g for g in report['gates'] if g['name'] == 'bootstrap_provenance')
    assert gate['status'] == 'fail'
    assert all('-m' not in c or 'pytest' not in c for c in calls)
    python_tests = [c for c in calls if any('test_vulkan_' in arg and '.py' in arg for arg in c)]
    assert python_tests and all(c[1] == 'tools/vulkan_wrapper_pytest.py' for c in python_tests)
    cmake = (ROOT / 'CMakeLists.txt').read_text()
    assert 'vulkan_bootstrap_provenance_validation' in cmake
    assert '-m pytest' not in cmake
    assert cmake.count('${CMAKE_CURRENT_SOURCE_DIR}/tools/vulkan_wrapper_pytest.py') == 3


def test_common_dependencies_bind_actual_driver_and_recorder():
    from composed_evidence_common import DEPENDENCIES
    assert {'tools/vulkan_wrapper_pytest.py', 'tools/capture_vulkan_composed_evidence.py'} <= set(DEPENDENCIES)


@pytest.mark.parametrize('tool', ['capture_vulkan_composed_evidence',
                                  'validate_vulkan_composed_matmul',
                                  'validate_vulkan_bootstrap_provenance'])
def test_public_tool_help_uses_semantic_names(tool):
    import re
    path = ROOT / 'tools' / (tool + '.py')
    assert path.exists(), 'semantic entrypoint missing'
    result = subprocess.run([sys.executable, str(path), '--help'], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not re.search(r'\bG[123]\b|capture_vulkan_g3', result.stdout)


def test_semantic_generation_path_and_durable_operational_docs():
    assert (ROOT / 'tests/python/test_vulkan_composed_generation.py').is_file()
    assert not (ROOT / 'tests/python/test_vulkan_g3_generation.py').exists()
    guide = (ROOT / 'docs/vulkan-bootstrap-evidence.md').read_text()
    commands = [shlex.split(line) for line in guide.splitlines()
                if line.startswith('PYTHONPATH=') and '--publish' in line]
    assert len(commands) == 1
    assert commands[0][0] == 'PYTHONPATH=.:build:tests/python'
    assert guide.count('tools/capture_vulkan_composed_evidence.py --mode') == 2
    assert 'pending' not in guide.lower()
    import re
    for name in ('README.md', 'docs/vulkan_operator_capability_matrix.md',
                 'docs/vulkan-bootstrap-evidence.md'):
        assert not re.search(r'\bG[123]\b', (ROOT / name).read_text()), name


@pytest.mark.parametrize('root_present', [False, True])
@pytest.mark.parametrize('anchored', [True, False])
def test_benchmark_ignore_policy_is_independent_of_output_presence(tmp_path, monkeypatch,
                                                                 root_present, anchored):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True,
                   capture_output=True, text=True, timeout=10)
    policy = (ROOT / '.gitignore').read_text()
    assert '/vulkan_gemm_benchmark.json' in policy.splitlines()
    if not anchored:
        policy = policy.replace('/vulkan_gemm_benchmark.json', 'vulkan_gemm_benchmark.json')
    # Move the rule's line number without changing its meaning.
    (tmp_path / '.gitignore').write_text('# CPU-only policy fixture\n' + policy)
    docs = tmp_path / 'docs'
    docs.mkdir()
    for path in (ROOT / 'docs').glob('*.json'):
        (docs / path.name).write_bytes(b'CPU-only document fixture\n')
    measured = docs / 'vulkan_gemm_benchmark.json'
    measured.write_bytes(b'CPU-only measured benchmark fixture\n')
    generated = tmp_path / 'vulkan_gemm_benchmark.json'
    if root_present:
        generated.write_bytes(b'CPU-only generated benchmark fixture\n')
    before = {path: file_state(path) for path in (measured, generated) if path.exists()}
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)
    if anchored:
        test_only_root_generated_benchmark_is_ignored()
    else:
        result = subprocess.run(['git', 'check-ignore', '--no-index',
                                 'docs/vulkan_gemm_benchmark.json'], cwd=tmp_path,
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert result.stdout.strip() == 'docs/vulkan_gemm_benchmark.json'
        with pytest.raises(AssertionError):
            test_only_root_generated_benchmark_is_ignored()
    assert generated.exists() == root_present
    assert {path: file_state(path) for path in before} == before


def test_only_root_generated_benchmark_is_ignored():
    # --no-index checks policy even before Git records the tracked deletion.
    result = subprocess.run(['git', 'check-ignore', '--no-index', '-v',
                             'vulkan_gemm_benchmark.json',
                             'docs/vulkan_gemm_benchmark.json'], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    matches = result.stdout.splitlines()
    assert len(matches) == 1
    source, path = matches[0].split('\t')
    ignore_file, line, pattern = source.split(':', 2)
    assert ignore_file == '.gitignore' and line.isdigit()
    assert pattern == '/vulkan_gemm_benchmark.json'
    assert path == 'vulkan_gemm_benchmark.json'
    for name in ('vulkan_capabilities.json', 'vulkan_coverage.json',
                 'vulkan_workload_coverage.json', 'vulkan_composed_matmul_coverage.json',
                 'vulkan_mse_sync_coverage.json', 'vulkan_general_matmul_sync_coverage.json',
                 'vulkan-runtime-foundation-qualification.json'):
        assert (ROOT / 'docs' / name).is_file(), name


def test_preservation_rejects_numerical_changes_but_allows_only_new_source_provenance():
    validator = module('validate_vulkan_bootstrap_provenance')
    old = {'cpu': {'value': 1}, 'source_identity': {'old': 'hash'},
           'runtime_identity': {'checkout_head': 'old', 'extension_path': '/old', 'device': 'vk:0'}}
    fresh = copy.deepcopy(old)
    fresh['source_identity'] = {'new': 'hash'}
    fresh['runtime_identity']['checkout_head'] = 'new'
    fresh['runtime_identity']['extension_path'] = '/new'
    validator.preserve_measurements(old, fresh)
    fresh['cpu']['value'] = 2
    with pytest.raises(ValueError, match='measurement'):
        validator.preserve_measurements(old, fresh)


def test_record_inventory_tampering_fails_even_if_output_hash_was_rebound(tmp_path):
    validator = module('validate_vulkan_bootstrap_provenance')
    document = cpu_provenance_fixture(tmp_path, validator)
    for mode, name in [('async', validator.OUTPUT_PATHS[0]), ('sync', validator.OUTPUT_PATHS[1])]:
        data = {key: {'mse_autograd_evidence': {'runtime_identity': {
            'execution_mode': mode, 'extension_sha256': document['extension']['sha256'],
            'extension_path': document['extension']['path']}}} for key in validator.mse_ids()}
        (tmp_path / name).write_text(json.dumps(data))
    models = {'schema_version': 1, 'records': {key: {'runtime_identity': {
        'execution_mode': mode, 'extension_sha256': document['extension']['sha256'],
        'extension_path': document['extension']['path']}}
        for mode in ('async', 'sync') for key in validator.model_ids(mode)}}
    (tmp_path / validator.OUTPUT_PATHS[2]).write_text(json.dumps(models))
    document['outputs'] = {name: validator.digest((tmp_path / name).read_bytes()) for name in validator.OUTPUT_PATHS}
    sidecar = tmp_path / 'CPU-only-sidecar.json'
    sidecar.write_text(json.dumps(document))
    validator.validate_file(sidecar, tmp_path)
    models['records'].pop(next(iter(models['records'])))
    (tmp_path / validator.OUTPUT_PATHS[2]).write_text(json.dumps(models))
    document['outputs'][validator.OUTPUT_PATHS[2]] = validator.digest((tmp_path / validator.OUTPUT_PATHS[2]).read_bytes())
    sidecar.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='inventory'):
        validator.validate_file(sidecar, tmp_path)


def file_state(path):
    data = path.read_bytes()
    return (data, hashlib.sha256(data).hexdigest(), path.stat().st_mtime_ns)


@pytest.mark.parametrize('unsafe', [
    'outside-parent', 'nested-outside-parent', 'inside-parent-alias', 'nested-inside-parent-alias',
    'outside-file-link', 'inside-file-link', 'dangling-file-link',
    'absolute', 'dotdot', 'dot', 'empty', 'double-slash', 'trailing-slash',
    'missing-parent', 'nondirectory-parent', 'directory-leaf',
])
def test_publication_preflights_all_paths_without_staging_or_outside_writes(tmp_path, monkeypatch, unsafe):
    validator = module('validate_vulkan_bootstrap_provenance')
    root, outside = tmp_path / 'trusted-root', tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    sentinel = outside / 'sentinel.json'
    sentinel.write_bytes(b'outside sentinel')
    first = root / 'first.json'
    first.write_bytes(b'first original')
    inside = root / 'inside'
    inside.mkdir()
    (inside / 'sentinel.json').write_bytes(b'inside sentinel')
    name = 'docs/sentinel.json'
    if unsafe == 'outside-parent':
        (root / 'docs').symlink_to(outside, target_is_directory=True)
    elif unsafe == 'nested-outside-parent':
        (root / 'docs').mkdir()
        (root / 'docs' / 'nested').symlink_to(outside, target_is_directory=True)
        name = 'docs/nested/sentinel.json'
    elif unsafe == 'inside-parent-alias':
        (root / 'docs').symlink_to(inside, target_is_directory=True)
    elif unsafe == 'nested-inside-parent-alias':
        (root / 'docs').mkdir()
        (root / 'docs' / 'nested').symlink_to(inside, target_is_directory=True)
        name = 'docs/nested/sentinel.json'
    elif unsafe.endswith('file-link'):
        (root / 'docs').mkdir()
        target = sentinel if unsafe == 'outside-file-link' else inside / 'sentinel.json'
        if unsafe == 'dangling-file-link': target = outside / 'absent.json'
        (root / name).symlink_to(target)
    elif unsafe == 'absolute': name = str(sentinel)
    elif unsafe == 'dotdot': name = '../outside/sentinel.json'
    elif unsafe == 'dot': name = './first.json'
    elif unsafe == 'empty': name = ''
    elif unsafe == 'double-slash': name = 'inside//sentinel.json'
    elif unsafe == 'trailing-slash': name = 'first.json/'
    elif unsafe == 'missing-parent': name = 'missing/sentinel.json'
    elif unsafe == 'nondirectory-parent': name = 'first.json/sentinel.json'
    elif unsafe == 'directory-leaf': name = 'inside'
    before = {p: file_state(p) for p in [sentinel, first, inside / 'sentinel.json']}
    link_before = (root / name).readlink() if not Path(name).is_absolute() and (root / name).is_symlink() else None
    staged, validated = [], []
    original = validator.tempfile.mkdtemp
    def track_staging(*args, **kwargs):
        staged.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(validator.tempfile, 'mkdtemp', track_staging)
    error = None
    try:
        validator.publish_bytes(root, {'first.json': b'new first', name: b'candidate'}, lambda: validated.append(True))
    except Exception as caught:
        error = caught
    # On RED, this proves the actual outside sentinel overwrite, not merely a
    # missing exception. All fixture files are inside the approved pytest root.
    assert {p: file_state(p) for p in before} == before
    assert not (outside / 'absent.json').exists()
    if link_before is not None:
        assert (root / name).is_symlink() and (root / name).readlink() == link_before
    assert isinstance(error, ValueError), repr(error)
    assert not staged and not validated


def test_publication_rechecks_paths_changed_by_validation_before_staging(tmp_path):
    validator = module('validate_vulkan_bootstrap_provenance')
    root, outside = tmp_path / 'root', tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    (root / 'docs').mkdir()
    sentinel = outside / 'target.json'
    sentinel.write_bytes(b'outside sentinel')
    before = file_state(sentinel)
    def changed_parent():
        (root / 'docs').rmdir()
        (root / 'docs').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        validator.publish_bytes(root, {'docs/target.json': b'candidate'}, changed_parent)
    assert file_state(sentinel) == before
    assert not list(root.glob('.composed-publication-*'))


def test_regular_publication_accepts_resolved_supplied_fixture_root_and_new_leaf(tmp_path):
    validator = module('validate_vulkan_bootstrap_provenance')
    root = tmp_path / 'trusted-fixture-root'
    root.mkdir()
    (root / 'docs').mkdir()
    alias = tmp_path / 'root-authority-alias'
    alias.symlink_to(root, target_is_directory=True)
    (root / 'docs/old.json').write_bytes(b'old')
    validator.publish_bytes(alias, {'docs/old.json': b'new', 'docs/new.json': b'created'}, lambda: None)
    assert (root / 'docs/old.json').read_bytes() == b'new'
    assert (root / 'docs/new.json').read_bytes() == b'created'
    assert not list(root.glob('.composed-publication-*'))


def test_failed_rollback_retains_confined_recovery_bytes(tmp_path, monkeypatch):
    validator = module('validate_vulkan_bootstrap_provenance')
    originals = {'a.json': b'original-a', 'b.json': b'original-b'}
    for name, data in originals.items(): (tmp_path / name).write_bytes(data)
    replace = validator.os.replace
    calls = []
    def failure(source, destination):
        calls.append(destination)
        if len(calls) >= 2: raise OSError('injected publication/rollback failure')
        return replace(source, destination)
    monkeypatch.setattr(validator.os, 'replace', failure)
    with pytest.raises(RuntimeError, match='recovery retained'):
        validator.publish_bytes(tmp_path, {name: b'candidate' for name in originals}, lambda: None)
    recovery, = tmp_path.glob('.composed-publication-*')
    manifest = json.loads((recovery / 'recovery.json').read_text())
    assert manifest['root'] == str(tmp_path.resolve())
    for name, backup in manifest['originals'].items():
        assert (recovery / backup).read_bytes() == originals[name]


def test_parent_change_between_replacements_never_redirects_rollback_outside(tmp_path, monkeypatch):
    validator = module('validate_vulkan_bootstrap_provenance')
    root, outside = tmp_path / 'root', tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    (root / 'docs').mkdir()
    originals = {'docs/a.json': b'original-a', 'docs/b.json': b'original-b'}
    for name, data in originals.items():
        (root / name).write_bytes(data)
        (outside / Path(name).name).write_bytes(b'outside sentinel')
    before = {p: file_state(p) for p in outside.iterdir()}
    replace = validator.os.replace
    calls = []
    def change_after_first(source, destination):
        calls.append(destination)
        result = replace(source, destination)
        if len(calls) == 1:
            (root / 'docs').rename(root / 'moved-docs')
            (root / 'docs').symlink_to(outside, target_is_directory=True)
        return result
    monkeypatch.setattr(validator.os, 'replace', change_after_first)
    with pytest.raises(RuntimeError, match='recovery retained'):
        validator.publish_bytes(root, {name: b'candidate' for name in originals}, lambda: None)
    assert len(calls) == 1  # No unsafe second replacement OR rollback attempt.
    assert {p: file_state(p) for p in before} == before
    recovery, = root.glob('.composed-publication-*')
    manifest = json.loads((recovery / 'recovery.json').read_text())
    for name, backup in manifest['originals'].items():
        assert (recovery / backup).read_bytes() == originals[name]


def test_capture_publication_guards_products_before_reading_or_staging_candidate(tmp_path, monkeypatch):
    validator = module('validate_vulkan_bootstrap_provenance')
    root, outside = tmp_path / 'root', tmp_path / 'outside'
    root.mkdir()
    outside.mkdir()
    sentinel = outside / 'vulkan_coverage.json'
    sentinel.write_bytes(b'outside sentinel')
    (root / 'docs').symlink_to(outside, target_is_directory=True)
    before = file_state(sentinel)
    def forbidden(*args, **kwargs):
        pytest.fail('unsafe publication must fail before archive reads/candidate staging')
    monkeypatch.setattr(validator, 'load', forbidden)
    monkeypatch.setattr(validator.tempfile, 'mkdtemp', forbidden)
    with pytest.raises(ValueError, match='unsafe publication'):
        validator.publish_attempts(tmp_path / 'async', tmp_path / 'sync', tmp_path / 'archive', root)
    assert file_state(sentinel) == before


@pytest.mark.parametrize('kind', ['missing', 'file'])
def test_publication_requires_existing_directory_root(tmp_path, kind):
    validator = module('validate_vulkan_bootstrap_provenance')
    root = tmp_path / 'invalid-root'
    if kind == 'file': root.write_bytes(b'not a directory')
    validated = []
    with pytest.raises(ValueError, match='publication root'):
        validator.publish_bytes(root, {'target.json': b'candidate'}, lambda: validated.append(True))
    assert not validated


def test_publication_rollback_removes_only_its_new_regular_leaf(tmp_path, monkeypatch):
    validator = module('validate_vulkan_bootstrap_provenance')
    original = tmp_path / 'old.json'
    original.write_bytes(b'original')
    before = file_state(original)
    replace = validator.os.replace
    def fail_existing(source, destination):
        if destination == original: raise OSError('injected publication failure')
        return replace(source, destination)
    monkeypatch.setattr(validator.os, 'replace', fail_existing)
    with pytest.raises(OSError, match='publication failure'):
        validator.publish_bytes(tmp_path, {'new.json': b'created', 'old.json': b'candidate'}, lambda: None)
    assert not (tmp_path / 'new.json').exists()
    assert file_state(original) == before
    assert not list(tmp_path.glob('.composed-publication-*'))


def test_documented_publication_environment_reaches_archive_after_real_runtime_bootstrap(tmp_path):
    from tools.run_vulkan_qualification import run_command
    documented, = [line for line in (ROOT / 'docs/vulkan-bootstrap-evidence.md').read_text().splitlines()
                   if line.startswith('PYTHONPATH=') and ' --publish ' in line]
    tokens = shlex.split(documented)
    pythonpath = tokens[0].split('=', 1)[1]
    fixture_root = tmp_path / 'trusted-fixture-root'
    fixture_root.mkdir()
    (fixture_root / 'docs').mkdir()
    archive = tmp_path / 'deliberately-missing-archive'
    script = r'''
import json, pathlib, sys
from tools import validate_vulkan_bootstrap_provenance as validator
assert 'pytorch_vulkan' not in sys.modules
events, reverse = [], []
def observe(frame, event, arg):
    name = frame.f_code.co_name
    module = frame.f_globals.get('__name__', '')
    if module == 'pytorch_vulkan' and name == '<module>' and event in ('call', 'return'):
        events.append('wrapper-' + event)
    if module == validator.__name__ and name == 'load' and event == 'call':
        events.append('archive-read')
    if (event == 'call' and module.startswith('torch.autograd')
            and name in ('backward', 'grad', '_engine_run_backward')):
        reverse.append({'function': name, 'wrapper_completed': 'wrapper-return' in events})
    if event == 'c_call' and getattr(arg, '__name__', '') == 'run_backward':
        reverse.append({'function': 'run_backward', 'wrapper_completed': 'wrapper-return' in events})
root, archive = map(pathlib.Path, sys.argv[1:])
sys.setprofile(observe)
try:
    validator.publish_attempts(root / 'unused-async', root / 'unused-sync', archive, root)
except ValueError as error:
    assert str(error) == f'missing/invalid provenance input: {archive / "manifest.json"}', str(error)
else:
    raise AssertionError('missing fixture archive unexpectedly accepted')
finally:
    sys.setprofile(None)
wrapper = sys.modules['pytorch_vulkan']
assert hasattr(wrapper, '_C')
assert events == ['wrapper-call', 'wrapper-return', 'archive-read'], events
assert reverse == [], reverse
assert not list(root.glob('.composed-publication-*'))
assert not list((root / 'docs').iterdir())
print(json.dumps({'boundary': 'runtime-bootstrap-before-missing-archive', 'events': events,
                  'reverse_calls': reverse, 'loaded_extension': wrapper._C.__file__}))
'''
    result = run_command([sys.executable, '-c', script, str(fixture_root), str(archive)], ROOT,
                         os.environ | {'PYTHONPATH': pythonpath, 'TMPDIR': '/tmp/opencode'},
                         timeout_seconds=30)
    assert result['exit_code'] == 0, result['stderr']
    observation = json.loads(result['stdout'])
    assert observation['boundary'] == 'runtime-bootstrap-before-missing-archive'
    assert observation['reverse_calls'] == []
    print(result['stdout'].strip())


def test_standalone_provenance_validation_without_build_stays_import_independent(tmp_path):
    from tools.run_vulkan_qualification import run_command
    missing = tmp_path / 'missing-sidecar.json'
    script = r'''
import json, sys
from tools import validate_vulkan_bootstrap_provenance as validator
assert 'pytorch_vulkan' not in sys.modules and 'torch' not in sys.modules
sys.argv = [validator.__file__, '--provenance', sys.argv[1]]
code = validator.main()
assert 'pytorch_vulkan' not in sys.modules and 'torch' not in sys.modules
print(json.dumps({'exit_code': code, 'wrapper_imported': False, 'upstream_imported': False}))
raise SystemExit(code)
'''
    result = run_command([sys.executable, '-c', script, str(missing)], ROOT,
                         os.environ | {'PYTHONPATH': '.:tests/python', 'TMPDIR': '/tmp/opencode'},
                         timeout_seconds=30)
    assert result['exit_code'] == 1
    assert 'missing/invalid provenance input' in result['stderr']
    assert 'ModuleNotFoundError' not in result['stderr']
    assert json.loads(result['stdout']) == {'exit_code': 1, 'wrapper_imported': False, 'upstream_imported': False}
    print(result['stdout'].strip())
