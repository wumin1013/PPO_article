"""Fresh independent initialization, exact two-phase execution and fair selection."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_target_tracking_formal as formal
from test_preview_full_batch import ShortEpisodes, agent_factory


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def unit_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, 'ROOT', tmp_path)
    monkeypatch.setattr(formal, 'OUT', tmp_path/'formal')
    sources = {p: dict(source_files={}, source_sha256=p+'-test') for p in ('pretraining', 'continuation')}
    protocol = dict(sources=sources, validation_path_ids=['a', 'b', 'c'])
    formal.old.write(formal.OUT/'protocol.json', protocol)
    monkeypatch.setattr(formal, 'prepare', lambda: protocol)
    monkeypatch.setattr(formal, 'source_state', lambda phase: sources[phase])
    return protocol


def fake_own_prior(tmp_path, monkeypatch, method='adaptive', seed=222):
    unit_protocol(tmp_path, monkeypatch)
    prior = formal.run_dir('pretraining', method, seed)
    prior.mkdir(parents=True)
    (prior/'checkpoint.pt').write_bytes(f'fresh {method} {seed}'.encode())
    cfg = formal.configuration('pretraining', method, seed)
    (prior/'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    formal.old.write(prior/'training_summary.json', dict(seed=seed, method=method,
        decisions=4096, checkpoint_sha256=formal.old.sha(prior/'checkpoint.pt')))
    formal.old.write(prior/'provenance.json', dict(source_sha256='pretraining-test',
        config_sha256=formal.old.sha(prior/'config.yaml')))
    formal.old.write(prior/'phase_complete.json', dict(phase='pretraining', seed=seed, method=method,
        artifacts={p.name: formal.artifact(p) for p in prior.iterdir()}))
    return prior


def test_registered_new_seeds_budgets_and_no_old_seed_reuse():
    assert formal.SEEDS == (111, 222, 333, 444, 555)
    assert formal.METHODS == ('fixed3', 'adaptive')
    assert formal.PRETRAIN_DECISIONS == formal.ADDITIONAL_DECISIONS == 4096
    assert formal.MILESTONES == (0, 256, 512, 1024, 2048, 4096)
    for method in formal.METHODS:
        for seed in formal.SEEDS:
            formal.validate_request(method, seed, 'formal')


@pytest.mark.parametrize('method,seed,stage', [
    ('adaptive', 11, 'formal'), ('fixed3', 101, 'formal'), ('adaptive', 901, 'formal'),
    ('fixed4', 111, 'formal'), ('fixed1p2', 222, 'formal'), ('fixed3', 111, 'smoke'),
    ('adaptive', 901, 'holdout'),
])
def test_unregistered_training_rejected(method, seed, stage):
    with pytest.raises(ValueError): formal.validate_request(method, seed, stage)


@pytest.mark.parametrize('method', formal.METHODS)
def test_sensor_starts_without_initializer_or_correction(method):
    cfg = formal.configuration('pretraining', method, 111)
    original = formal.sensor.configuration('preview_time', method, 111)
    assert 'init_checkpoint' not in cfg['revision']
    assert 'feed_tracking' not in cfg['revision']
    for key in original:
        if key != 'revision': assert cfg[key] == original[key]
    ignored = ('formal_study', 'formal_phase', 'data_status')
    assert {k: v for k, v in cfg['revision'].items() if k not in ignored} == {
        k: v for k, v in original['revision'].items() if k not in ignored}


@pytest.mark.parametrize('method', formal.METHODS)
def test_target_config_changes_only_owned_prior_and_attribution(tmp_path, monkeypatch, method):
    prior = fake_own_prior(tmp_path, monkeypatch, method)
    cfg = formal.configuration('continuation', method, 222)
    original = formal.target.configuration('preview_time', method, 101)
    assert cfg['seed'] == 222
    assert cfg['revision']['init_checkpoint'] == (prior/'checkpoint.pt').relative_to(tmp_path).as_posix()
    changed = {'initialization', 'init_checkpoint', 'init_checkpoint_sha256', 'zero_step_role',
               'formal_study', 'formal_phase', 'data_status'}
    for key in original:
        if key not in ('seed', 'revision'): assert cfg[key] == original[key]
    for key, value in original['revision'].items():
        if key not in changed: assert cfg['revision'][key] == value
    assert cfg['revision']['feed_tracking'] == original['revision']['feed_tracking']


@pytest.mark.parametrize('field,value', [('seed', 101), ('method', 'fixed3'), ('decisions', 1024), ('checkpoint_sha256', 'fake')])
def test_mismatched_or_incomplete_prior_rejected(tmp_path, monkeypatch, field, value):
    prior = fake_own_prior(tmp_path, monkeypatch)
    summary = formal.old.read(prior/'training_summary.json')
    summary[field] = value
    formal.old.write(prior/'training_summary.json', summary)
    with pytest.raises(ValueError, match='Own same-method'):
        formal.configuration('continuation', 'adaptive', 222)


def test_prior_bytes_cannot_change_after_completion(tmp_path, monkeypatch):
    prior = fake_own_prior(tmp_path, monkeypatch)
    (prior/'config.yaml').write_text('tampered: true\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Artifact changed'):
        formal.configuration('continuation', 'adaptive', 222)


def test_phase_switch_uses_original_then_corrected_feed_and_restores(tmp_path, monkeypatch):
    unit_protocol(tmp_path, monkeypatch)
    names = ('OUT', 'source_state', 'configuration', 'MacroEnvironment', 'make_agent', 'prepare')
    originals = {key: getattr(formal.core, key) for key in names}
    original_feed = formal.base.cnc_env.apply_kinematic_constraints
    for phase in ('pretraining', 'continuation', 'pretraining'):
        with pytest.raises(RuntimeError, match='deliberate'):
            with formal.phase_installed(phase):
                expected = original_feed if phase == 'pretraining' else formal.target.apply_target_tracking_constraints
                assert formal.base.cnc_env.apply_kinematic_constraints is expected
                assert formal.core.MacroEnvironment is (formal.sensor.SensorEnvironment if phase == 'pretraining'
                    else formal.full.local.LocalCornerEnvironment)
                assert formal.core.make_agent is (formal.BASE_SENSOR_AGENT if phase == 'pretraining'
                    else formal.full.local.make_agent)
                with pytest.raises(RuntimeError, match='nest'):
                    with formal.phase_installed(phase): pass
                raise RuntimeError('deliberate')
        assert all(getattr(formal.core, key) is value for key, value in originals.items())
        assert formal.base.cnc_env.apply_kinematic_constraints is original_feed


def test_unknown_or_prepatched_phase_is_rejected(tmp_path, monkeypatch):
    unit_protocol(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        with formal.phase_installed('holdout'): pass
    monkeypatch.setattr(formal.base.cnc_env, 'apply_kinematic_constraints', lambda *a: None)
    with pytest.raises(RuntimeError, match='fresh process'):
        with formal.phase_installed('continuation'): pass


def test_delegation_restores_old_runner_even_after_exception(tmp_path, monkeypatch):
    unit_protocol(tmp_path, monkeypatch)
    saved = {key: getattr(formal.base, key) for key in ('OUT', 'SEEDS', 'prepare', 'configuration', 'phase_installed')}
    with pytest.raises(RuntimeError, match='deliberate'):
        with formal._delegated():
            assert formal.base.OUT == formal.OUT
            assert formal.base.SEEDS == formal.SEEDS
            assert formal.base.phase_installed is formal.phase_installed
            with pytest.raises(RuntimeError, match='nest'):
                with formal._delegated(): pass
            raise RuntimeError('deliberate')
    assert all(getattr(formal.base, key) is value for key, value in saved.items())


def train_fake_continuation(tmp_path, monkeypatch):
    unit_protocol(tmp_path, monkeypatch)
    prior = tmp_path/'fresh_prior.pt'
    prior.write_bytes(b'unit prior; tiny agent factory does not load this')
    cfg = dict(seed=901, revision=dict(training_path_ids=['three', 'five'], variant='adaptive',
        pretraining_macro_decisions=64, init_checkpoint='fresh_prior.pt'), ppo=dict(epochs=2))
    monkeypatch.setattr(formal, 'configuration', lambda *args: copy.deepcopy(cfg))
    monkeypatch.setattr(formal.full.local, 'LocalCornerEnvironment', ShortEpisodes)
    monkeypatch.setattr(formal.full.local, 'make_agent', agent_factory)
    return cfg


def test_original_full_batch_trains_and_resumes_without_touching_step0(tmp_path, monkeypatch):
    train_fake_continuation(tmp_path, monkeypatch)
    initial = formal.train_continuation('adaptive', 901, 0, 'smoke')
    frozen = (initial/'checkpoint.pt').read_bytes()
    zero = torch.load(initial/'checkpoint.pt', weights_only=False)
    assert zero['actor_optimizer']['state'] == zero['critic_optimizer']['state'] == {}
    final = formal.train_continuation('adaptive', 901, 64, 'smoke')
    state = torch.load(final/'checkpoint.pt', weights_only=False)
    assert state['optimized_decisions'] == 64 and len(state['updates']) == 1
    assert len(state['episodes']) > 1
    assert formal.train_continuation('adaptive', 901, 0, 'smoke') == initial
    assert (initial/'checkpoint.pt').read_bytes() == frozen
    assert formal.train_continuation('adaptive', 901, 64, 'smoke') == final


def test_raw_snapshot_identity_checked_even_when_manifest_is_rewritten(tmp_path, monkeypatch):
    train_fake_continuation(tmp_path, monkeypatch)
    snapshot = formal.train_continuation('adaptive', 901, 0, 'smoke')
    path = snapshot/'training_summary.json'
    bad = formal.old.read(path); bad['optimized_decisions'] = 64
    formal.old.write(path, bad)
    manifest = formal.old.read(snapshot/'snapshot_manifest.json')
    manifest['files']['training_summary.json'] = formal.old.sha(path)
    formal.old.write(snapshot/'snapshot_manifest.json', manifest)
    with pytest.raises(ValueError, match='optimized budget'):
        formal.verify_node('adaptive', 901, 0, 'smoke', require_validation=False)


@pytest.mark.parametrize('field,value', [('seed', 101), ('method', 'fixed3'), ('training_decisions', 64), ('source_sha256', 'wrong')])
def test_validation_identity_cannot_be_relabelled(tmp_path, monkeypatch, field, value):
    train_fake_continuation(tmp_path, monkeypatch)
    snapshot = formal.train_continuation('adaptive', 901, 0, 'smoke')
    directory = snapshot/'validation'; directory.mkdir()
    rows = []
    for path_id in ('a', 'b', 'c'):
        trace = directory/f'{path_id}.npz'; trace.write_bytes(path_id.encode())
        rows.append(dict(path_id=path_id, seed=901, method='adaptive', training_decisions=0,
            source_sha256='continuation-test', config_sha256=formal.old.sha(snapshot/'config.yaml'),
            checkpoint_sha256=formal.old.sha(snapshot/'checkpoint.pt'), trace_sha256=formal.old.sha(trace)))
    rows[1][field] = value
    formal.old.write(directory/'summary.json', dict(rows=rows))
    with pytest.raises(ValueError, match='Validation row'):
        formal.verify_node('adaptive', 901, 0, 'smoke')


def test_complete_coverage_has_priority_over_failure_time_and_step0_is_eligible():
    early = dict(step=0, successes=3, complete_validation_time_s=14.)
    later = dict(step=4096, successes=3, complete_validation_time_s=15.)
    failure = dict(step=0, successes=2, complete_validation_time_s=.001)
    assert min([failure, later, early], key=formal.selection_key) is early
    incomplete = dict(step=256, successes=2, complete_validation_time_s=.0001)
    assert min([incomplete, failure], key=formal.selection_key) is failure


def test_prepare_hashes_holdout_opaquely_and_never_loads_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, 'ROOT', tmp_path)
    monkeypatch.setattr(formal, 'OUT', tmp_path/'new_formal')
    holdout = tmp_path/'holdout_manifest.json'
    holdout.write_bytes(b'opaque holdout descriptor, intentionally not valid json')
    monkeypatch.setattr(formal, 'HOLDOUT_MANIFEST', holdout)
    data = tmp_path/'development'; data.mkdir(); (data/'manifest.json').write_bytes(b'unit development')
    monkeypatch.setattr(formal.old, 'DATA', data)
    monkeypatch.setattr(formal.old, 'check_protocol', lambda: None)
    monkeypatch.setattr(formal.old, 'manifest', lambda: dict(paths=[dict(id='a', split='train'), dict(id='b', split='validation')]))
    monkeypatch.setattr(formal, 'development_gate', lambda: dict(passed=True, descriptors={}))
    monkeypatch.setattr(formal, 'source_state', lambda phase: dict(source_files={}, source_sha256=phase))
    monkeypatch.setattr(formal, 'evaluation_sources', lambda: dict(source_files={}, source_sha256='eval'))
    result = formal.prepare()
    assert result['holdout_manifest']['sha256'] == formal.old.sha(holdout)
    assert result['test_access'] is False
    assert result['prior_reuse'] is False and result['training_budget_total_decisions'] == 81920
    assert result['pretraining_decisions'] == result['additional_decisions'] == 4096
    assert formal.prepare() == result


def test_source_scope_includes_training_eval_statistics_and_two_step():
    all_files = set(formal.DRIVER_FILES) | set(formal.EVALUATION_FILES)
    for suffix in ('formal', 'holdout', 'statistics', 'two_step'):
        assert f'PPO_project/tools/preview_target_tracking_{suffix}.py' in all_files
        assert f'PPO_project/tests/test_preview_target_tracking_{suffix}.py' in all_files
    assert {'PPO_project/tools/preview_formal_two_step.py',
            'PPO_project/tools/preview_two_step_eval.py'} <= all_files


@pytest.mark.parametrize('stage,count', [('formal', 10), ('smoke', 2)])
def test_batch_schedules_each_chain_once_without_freeze_or_test(tmp_path, monkeypatch, stage, count):
    unit_protocol(tmp_path, monkeypatch)
    jobs = []
    def execute(method, seed, stage, directory):
        jobs.append((method, seed, stage))
        return dict(method=method, seed=seed, exit_code=0, wall_s=1.)
    monkeypatch.setattr(formal, 'execute_chain', execute)
    result = formal.batch(3, stage)
    assert len(jobs) == len(set(jobs)) == count
    assert {seed for _, seed, _ in jobs} == (set(formal.SEEDS) if stage == 'formal' else {901})
    assert result['status'] == 'complete'
    assert result['test_access'] is False and result['automatic_freeze'] is False
    assert not (formal.OUT/'freeze.json').exists()


def test_failed_chain_keeps_other_completed_results_and_does_not_select_subset(tmp_path, monkeypatch):
    unit_protocol(tmp_path, monkeypatch)
    def execute(method, seed, stage, directory):
        return dict(method=method, seed=seed, exit_code=int(method == 'fixed3'), wall_s=1.)
    monkeypatch.setattr(formal, 'execute_chain', execute)
    with pytest.raises(RuntimeError, match='chain failure'):
        formal.batch(2, 'smoke')
    report = formal.old.read(next((formal.OUT/'batch_invocations').glob('*/summary.json')))
    assert len(report['commands']) == 2 and report['status'] == 'failed_or_interrupted'
    assert not (formal.OUT/'selections').exists()


@pytest.mark.parametrize('change', ['fixed_distance', 'D_pending', 'not_faster', 'old_comparator', 'incomplete'])
def test_formal_registration_rejects_unpassed_development_gate(tmp_path, monkeypatch, change):
    import preview_target_tracking_extension as extension
    monkeypatch.setattr(formal, 'ROOT', tmp_path)
    monkeypatch.setattr(formal, 'OUT', tmp_path/'formal')
    monkeypatch.setattr(formal, 'EXTENSION_OUT', tmp_path/'extension')
    monkeypatch.setattr(extension, 'OUT', formal.EXTENSION_OUT)
    formal.old.write(formal.EXTENSION_OUT/'delivery_manifest.json', dict(files={}))
    formal.old.write(formal.EXTENSION_OUT/'completion_audit.json', dict(passed=change != 'incomplete',
        coordinator_exit_code=0, all_validation_rows_verified=72))
    report = dict(uniform_fixed_method='fixed1p2' if change == 'fixed_distance' else 'fixed3',
        adaptive_beats_uniform_fixed=change != 'not_faster',
        selected_actor_D_gate=dict(same_actor_D_gate_passed=change != 'D_pending'),
        groups=dict(adaptive=dict(success_count=3, complete_total_time_s=14. if change == 'old_comparator' else 13.8)))
    monkeypatch.setattr(extension, 'report', lambda: report)
    with pytest.raises(ValueError): formal.development_gate()
    assert not (formal.OUT/'protocol.json').exists()
