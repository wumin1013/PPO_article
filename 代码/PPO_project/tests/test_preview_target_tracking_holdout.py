"""Synthetic full-study tests; never open real heldout geometry or models."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_target_tracking_holdout as module
from test_preview_formal_holdout import put_json, artifact, source


@pytest.fixture
def frozen_study(tmp_path, monkeypatch):
    root = tmp_path
    formal = root/'论文项目/revision_20260910_continued/target_tracking_formal_v1'
    pre_source = source(root, ['PPO_project/src/pre_fixture.py'])
    continued_source = source(root, ['PPO_project/src/continued_fixture.py', 'PPO_project/src/algorithms/two_step_baseline.py'])
    evaluation_source = source(root, sorted(module.EVALUATION_FILES))
    holdout = root/module.HOLDOUT_FILE
    paths = [dict(id=f'{family}_{i}', family=family, split='test') for family in module.FAMILIES for i in range(4)]
    put_json(holdout, dict(schema_version='preview-frontier-holdout-v1', paths=paths,
        tolerance_definition=dict(planning_bandwidth_mm=2.5, one_sided_tolerance_mm=1.25,
            source_discretization_bound_mm=.005, source_bound_added_to_planning_tolerance=False)))
    monkeypatch.setattr(module, 'HOLDOUT_SHA256', module.sha(holdout))
    protocol_path = formal/'protocol.json'
    protocol = dict(version=module.VERSION, seeds=list(module.SEEDS), methods=list(module.METHODS),
        sources=dict(pretraining=pre_source, continuation=continued_source), evaluation_sources=evaluation_source,
        holdout_manifest=artifact(root, holdout), validation_path_ids=[f'{f}_validation' for f in module.FAMILIES])
    put_json(protocol_path, protocol)
    configs = {}
    monkeypatch.setattr(module, 'registered_configuration', lambda phase, method, seed: copy.deepcopy(configs[phase, method, seed]))
    monkeypatch.setattr(module, 'registered_source', lambda phase: pre_source if phase == 'pretraining' else continued_source)

    def make_files(directory, cfg, summary, sources, payload):
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'checkpoint.pt').write_bytes(payload)
        (directory/'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
        put_json(directory/'provenance.json', dict(**sources, config_sha256=module.sha(directory/'config.yaml')))
        with zipfile.ZipFile(directory/'source.zip', 'w') as archive:
            for name in sources['source_files']: archive.write(root/name, name)
        put_json(directory/'training_summary.json', dict(**summary, checkpoint_sha256=module.sha(directory/'checkpoint.pt')))

    def logs(directory, step):
        (directory/'updates.jsonl').write_text(''.join(json.dumps(dict(decisions=i+64, transition_count=64))+'\n'
                                                     for i in range(0, step, 64)), encoding='utf-8')
        (directory/'episodes.jsonl').write_text('', encoding='utf-8')
        (directory/'training_final_partial.npz').write_bytes(b'synthetic trace')

    records = []
    for seed in module.SEEDS:
        for method in module.METHODS:
            prior = formal/'pretraining'/f'{method}_seed{seed}'
            pre_cfg = dict(seed=seed, revision=dict(variant=method))
            configs['pretraining', method, seed] = copy.deepcopy(pre_cfg)
            make_files(prior, pre_cfg, dict(seed=seed, method=method, decisions=4096, pretraining_steps=0),
                       pre_source, f'new-prior-{method}-{seed}'.encode())
            logs(prior, 4096)
            put_json(prior/'costs.json', dict(decisions=4096, optimizer_updates=64))
            put_json(prior/'phase_complete.json', dict(phase='pretraining', seed=seed, method=method,
                artifacts={p.name: artifact(root, p) for p in prior.iterdir() if p.is_file()}))
            cfg = dict(seed=seed, environment=dict(epsilon=2.5, interpolation_period=.001, max_steps=60000),
                kinematic_constraints=dict(MAX_VEL=100., MAX_ACC=200., MAX_JERK=2000., MAX_ANG_VEL=2., MAX_ANG_ACC=20., MAX_ANG_JERK=200.),
                revision=dict(adaptive_preview=method == 'adaptive', macro_period_steps=20,
                    fixed_distance_mm=3., initial_preview_mm=3., active_action_dims=[1, 2] if method == 'adaptive' else [1],
                    terminal_definition='Original v2;20ms complete rest', init_checkpoint=(prior/'checkpoint.pt').relative_to(root).as_posix(),
                    init_checkpoint_sha256=module.sha(prior/'checkpoint.pt'),
                    feed_tracking=dict(implementation='preview_target_tracking.project_feed_target', sampling_period_s=.001,
                        not_a_learned_preview_rule=True, terminal_brake='Frozen v2 plan_terminal_brake, unchanged')))
            configs['continuation', method, seed] = copy.deepcopy(cfg)
            candidates = []
            for step in module.STEPS:
                snapshot = formal/'continuation'/f'{method}_seed{seed}'/'checkpoints'/f'step{step}'
                make_files(snapshot, cfg, dict(seed=seed, method=method, decisions=step, optimized_decisions=step,
                    pending_buffer_decisions=0), continued_source, f'actor-{method}-{seed}-{step}'.encode())
                logs(snapshot, step)
                put_json(snapshot/'snapshot_manifest.json', dict(decisions=step,
                    files={p.name: module.sha(p) for p in snapshot.iterdir() if p.is_file()}))
                val = snapshot/'validation'
                val.mkdir()
                binding = dict(checkpoint_sha256=module.sha(snapshot/'checkpoint.pt'), config_sha256=module.sha(snapshot/'config.yaml'),
                    source_sha256=continued_source['source_sha256'])
                put_json(val/'binding.json', binding)
                rows = []
                for path_id in protocol['validation_path_ids']:
                    trace = val/f'{path_id}.npz'; trace.write_bytes(f'{method}-{seed}-{step}-{path_id}'.encode())
                    row = dict(**binding, seed=seed, method=method, training_decisions=step, path_id=path_id,
                        success=True, elapsed_time_s=9. if step == 256 else 10., trace_sha256=module.sha(trace))
                    put_json(val/f'{path_id}.json', row); rows.append(row)
                put_json(val/'summary.json', dict(rows=rows))
                put_json(val/'delivery_manifest.json', dict(files={p.name: module.sha(p) for p in val.iterdir()}))
                candidates.append(dict(step=step, successes=3, count=3, complete_validation_time_s=sum(r['elapsed_time_s'] for r in rows),
                    checkpoint_dir=snapshot.relative_to(root).as_posix(), checkpoint_sha256=module.sha(snapshot/'checkpoint.pt'),
                    validation_sha256=module.sha(val/'summary.json'), training_summary_sha256=module.sha(snapshot/'training_summary.json'),
                    snapshot_manifest_sha256=module.sha(snapshot/'snapshot_manifest.json')))
            selected = candidates[1]
            chosen = root/selected['checkpoint_dir']
            selection_path = formal/'selections'/f'{method}_seed{seed}.json'
            put_json(selection_path, dict(seed=seed, method=method, selected=selected, all_checkpoints=candidates, step0_eligible=True))
            final = root/candidates[-1]['checkpoint_dir']
            names = module.CORE_ARTIFACTS | {'snapshot_manifest.json', 'validation/summary.json', 'validation/delivery_manifest.json'}
            records.append(dict(seed=seed, method=method, selected_step=256,
                available_pretraining_decisions=4096, available_additional_decisions=4096,
                checkpoint_dir=chosen.relative_to(root).as_posix(), artifacts={name: artifact(root, chosen/name) for name in names},
                pretraining=dict(checkpoint_dir=prior.relative_to(root).as_posix(), decisions=4096,
                    artifacts={name: artifact(root, prior/name) for name in module.CORE_ARTIFACTS | {'phase_complete.json', 'costs.json'}}),
                selection=artifact(root, selection_path), additional_budget_completion=artifact(root, final/'training_summary.json'),
                additional_budget_snapshot_manifest=artifact(root, final/'snapshot_manifest.json'), source_sha256=continued_source['source_sha256']))
    freeze = formal/'freeze.json'
    put_json(freeze, dict(version=module.VERSION, seeds=list(module.SEEDS), methods=list(module.METHODS),
        records=records, protocol=artifact(root, protocol_path), source=continued_source, holdout_manifest=artifact(root, holdout),
        all_training_complete=True, training_budget_total_decisions=81920))
    return root, freeze, holdout, configs


def deny_geometry(monkeypatch, holdout):
    read = module.read_json
    digest = module.old.sha
    def guarded_read(path):
        if Path(path) == holdout: raise AssertionError('Holdout manifest opened before complete gate')
        return read(path)
    def guarded_hash(path):
        if Path(path) == holdout: raise AssertionError('Holdout bytes hashed before complete gate')
        return digest(path)
    monkeypatch.setattr(module, 'read_json', guarded_read)
    monkeypatch.setattr(module.old, 'sha', guarded_hash)


def test_gate_checks_ten_chains_and_sixty_nodes_without_manifest_read(frozen_study, monkeypatch):
    root, freeze, holdout, _ = frozen_study
    deny_geometry(monkeypatch, holdout)
    gate = module.validate_freeze(freeze, root)
    assert len(gate['models']) == 10
    assert len(gate['common_configuration']['constraints']) == 6
    assert gate['common_configuration']['feed_tracking']['not_a_learned_preview_rule'] is True


def test_only_complete_gate_can_call_twelve_geometry_loaders(frozen_study):
    root, freeze, _, _ = frozen_study
    seen = []
    gate, manifest, references = module.load_frozen_references(freeze, root,
        lambda path_id, directory: seen.append(path_id) or [[0., 0.], [1., 0.]])
    assert len(gate['models']) == 10 and len(seen) == len(references) == len(manifest['paths']) == 12


@pytest.mark.parametrize('fault', ['missing_freeze', 'missing_tenth', 'duplicate_tenth', 'old_seeds', 'old_schema',
    'incomplete_training', 'wrong_total_budget', 'wrong_holdout', 'evaluator_changed', 'tenth_model_changed',
    'last_final_missing', 'unselected_node_changed', 'validation_trace_changed', 'sensor_logs_changed',
    'candidate_missing', 'wrong_selected', 'wrong_saved_successes', 'wrong_actual_config'])
def test_failed_gate_never_reads_manifest_or_calls_loader(frozen_study, monkeypatch, fault):
    root, freeze, holdout, configs = frozen_study
    data = module.read_json(freeze); record = data['records'][-1]
    if fault == 'missing_freeze': freeze.unlink()
    elif fault == 'missing_tenth': data['records'].pop(); put_json(freeze, data)
    elif fault == 'duplicate_tenth': data['records'][-1] = data['records'][0]; put_json(freeze, data)
    elif fault == 'old_seeds': data['seeds'] = [11, 22, 33, 44, 55]; put_json(freeze, data)
    elif fault == 'old_schema': data['version'] = 'formal-preview-continuation-v1'; put_json(freeze, data)
    elif fault == 'incomplete_training': data['all_training_complete'] = False; put_json(freeze, data)
    elif fault == 'wrong_total_budget': data['training_budget_total_decisions'] = 40960; put_json(freeze, data)
    elif fault == 'wrong_holdout': data['holdout_manifest']['file'] = 'PPO_project/data/preview_continuation_v1/manifest.json'; put_json(freeze, data)
    elif fault == 'evaluator_changed': (root/'PPO_project/tools/preview_target_tracking_statistics.py').write_bytes(b'changed')
    elif fault == 'tenth_model_changed': (root/record['artifacts']['checkpoint.pt']['file']).write_bytes(b'changed')
    elif fault == 'last_final_missing': (root/record['additional_budget_completion']['file']).unlink()
    elif fault == 'unselected_node_changed': (root/record['checkpoint_dir']).with_name('step2048').joinpath('updates.jsonl').write_bytes(b'changed')
    elif fault == 'validation_trace_changed': next((root/record['checkpoint_dir']/'validation').glob('*.npz')).write_bytes(b'changed')
    elif fault == 'sensor_logs_changed': (root/record['pretraining']['checkpoint_dir']/'updates.jsonl').write_bytes(b'changed')
    elif fault == 'wrong_actual_config': configs['continuation', record['method'], record['seed']]['environment']['epsilon'] = 4.
    else:
        path = root/record['selection']['file']; selection = module.read_json(path)
        if fault == 'candidate_missing': selection['all_checkpoints'].pop()
        elif fault == 'wrong_selected': selection['selected'] = selection['all_checkpoints'][-1]
        else: selection['all_checkpoints'][0]['successes'] = 2
        put_json(path, selection); record['selection'] = artifact(root, path); put_json(freeze, data)
    deny_geometry(monkeypatch, holdout)
    def loader(*args): raise AssertionError('Policy test references loaded through failed gate')
    with pytest.raises((module.FreezeError, FileNotFoundError)):
        module.load_frozen_references(freeze, root, loader)


def test_production_holdout_identity_is_fixed_and_has_no_cli_override():
    assert module.HOLDOUT_SHA256 == '21673efe97966cb45e117564df64423014ca4c30b90cb54fc0854bac13e4a927'
    assert module.SEEDS == (111, 222, 333, 444, 555)
    assert module.HOLDOUT_FILE.endswith('preview_frontier_holdout_v1/manifest.json')


def test_old_six_path_schema_remains_rejected_by_new_gate(frozen_study):
    root, freeze, _, _ = frozen_study
    with pytest.raises(module.FreezeError, match='Unknown freeze schema'):
        module.old.validate_freeze(freeze, root)


def test_output_writer_preserves_failed_outcomes(tmp_path):
    path = tmp_path/'failure.json'
    module.write_json_new(path, dict(success=False, failure_reasons=['ordered_path_incomplete']))
    with pytest.raises(FileExistsError): module.write_json_new(path, dict(success=True))
    assert module.read_json(path)['success'] is False


def test_evaluation_recomputes_actions_from_actual_state_and_retains_all_failures(tmp_path, monkeypatch):
    """Only artificial in-memory paths and an artificial two-step environment."""
    from contextlib import contextmanager
    import numpy as np
    import torch
    import preview_full_batch_study as full
    import preview_restart_closed_loop as audit
    import generate_preview_continuation_paths as geometry
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    freeze_path = tmp_path/'formal/freeze.json'; put_json(freeze_path, {})
    checkpoint = tmp_path/'formal/model/checkpoint.pt'
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b'artificial actor')
    cfg = dict(seed=111)
    chosen = dict(record=dict(seed=111, method='adaptive', selected_step=256,
        artifacts={'config.yaml': {'sha256': 'config-sha'}}), cfg=cfg, checkpoint=checkpoint)
    gate = dict(freeze_path=freeze_path, freeze_sha256='freeze-sha', models=[chosen],
        freeze=dict(source={'source_sha256': 'source-sha'}, holdout_manifest={'sha256': 'synthetic-data-sha'}))
    monkeypatch.setattr(module, 'validate_freeze', lambda *args, **kwargs: gate)
    paths = [dict(id=f'{family}_{i}', family=family, reference_sha256='synthetic-reference')
             for family in module.FAMILIES for i in range(4)]
    monkeypatch.setattr(module, 'frozen_manifest', lambda *args: (dict(paths=paths), tmp_path/'fake-data'))
    loaded = []
    monkeypatch.setattr(geometry, 'load_path', lambda path_id, data: loaded.append(path_id) or np.array([[0., 0.], [1., 0.]]))
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: dict(config=cfg, source_sha256='source-sha', actor={}))
    seen = []
    class Actor:
        def load_state_dict(self, state): pass
        def eval(self): pass
    class Agent:
        actor = Actor()
        def take_action(self, state, deterministic):
            assert deterministic is True
            seen.append(float(state[0]))
            return np.array([state[0]])
    class Environment:
        def __init__(self, cfg, path_id, reference):
            assert reference.shape == (2, 2)
            self.state = np.array([0.]); self.done = False
            self.trace = [dict(timestamp_s=0., x_mm=0., y_mm=0.)]
        def step(self, action):
            assert action[0] == self.state[0]
            self.state = self.state + 1
            self.trace.append(dict(timestamp_s=float(self.state[0])*.02, x_mm=float(action[0]), y_mm=0.))
            self.done = self.state[0] == 2
            if self.done:
                self.result = dict(success=False, elapsed_time_s=.04,
                    failure_reasons=['ordered_path_incomplete'], termination_reason='synthetic_failure')
    contexts = []
    @contextmanager
    def installed():
        contexts.append('enter')
        try: yield
        finally: contexts.append('exit')
    monkeypatch.setattr(audit, 'corrected_execution', installed)
    monkeypatch.setattr(full.local, 'LocalCornerEnvironment', Environment)
    monkeypatch.setattr(full.local, 'make_agent', lambda cfg, env: Agent())
    rows = module.evaluate(freeze_path, 111, 'adaptive')
    assert len(rows) == len(loaded) == 12 and seen == [0., 1.]*12
    assert all(row['success'] is False and row['failure_reasons'] == ['ordered_path_incomplete'] for row in rows)
    assert contexts == ['enter', 'exit']
    bytes_before = {p: p.read_bytes() for p in (freeze_path.parent/'holdout').rglob('*') if p.is_file()}
    resumed = module.evaluate(freeze_path, 111, 'adaptive')
    assert resumed == rows and len(loaded) == 12 and seen == [0., 1.]*12
    assert all(p.read_bytes() == content for p, content in bytes_before.items())
