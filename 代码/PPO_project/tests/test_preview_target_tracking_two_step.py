"""Gate ordering and original-planner delegation; no real holdout access."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_target_tracking_two_step as study


def fixtures(monkeypatch, tmp_path):
    cfg = study.original.configuration()
    protocol = dict(configuration=cfg, formal_protocol_sha256='formal', comparison_rule='common complete only')
    checked = dict(common_configuration=dict(environment=dict(interpolation_period=cfg['dt_s'],
            epsilon=cfg['one_sided_tolerance_mm']*2, max_steps=cfg['max_steps']), constraints=cfg['limits']),
        freeze=dict(protocol=dict(sha256='formal'), holdout_manifest=dict(sha256='holdout')), freeze_sha256='freeze')
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(study, 'prepare', lambda: protocol)
    return protocol, checked


@pytest.mark.parametrize('reason', ('missing_model', 'bad_source', 'incomplete4096', 'selection_changed'))
def test_bad_freeze_fails_before_any_geometry(monkeypatch, tmp_path, reason):
    fixtures(monkeypatch, tmp_path)
    calls = []
    def fail(*args): raise ValueError(reason)
    monkeypatch.setattr(study, 'access_gate', lambda: SimpleNamespace(validate_freeze=fail,
        frozen_manifest=lambda *args: calls.append('forbidden')))
    with pytest.raises(ValueError, match=reason): study.evaluate(tmp_path/'freeze.json')
    assert calls == []


@pytest.mark.parametrize('change', ('tolerance', 'limit', 'protocol'))
def test_configuration_mismatch_fails_before_geometry(monkeypatch, tmp_path, change):
    _, checked = fixtures(monkeypatch, tmp_path)
    calls = []
    if change == 'tolerance': checked['common_configuration']['environment']['epsilon'] = 9.
    elif change == 'limit': checked['common_configuration']['constraints'] = {}
    else: checked['freeze']['protocol']['sha256'] = 'other'
    monkeypatch.setattr(study, 'access_gate', lambda: SimpleNamespace(validate_freeze=lambda *args: checked,
        frozen_manifest=lambda *args: calls.append('forbidden')))
    with pytest.raises(ValueError): study.evaluate(tmp_path/'freeze.json')
    assert calls == []


def test_twelve_paths_delegated_once_with_failure_outputs_preserved(monkeypatch, tmp_path):
    _, checked = fixtures(monkeypatch, tmp_path)
    items = [dict(id=f'synthetic_{family}_{i}', family=family)
             for family in ('mixed_lengths', 'dense_corners', 'line_curve') for i in range(4)]
    events = []
    def validate(*args):
        events.append('gate')
        return checked
    def manifest(*args):
        events.append('geometry')
        return dict(paths=items), tmp_path
    monkeypatch.setattr(study, 'access_gate', lambda: SimpleNamespace(validate_freeze=validate, frozen_manifest=manifest))
    import generate_preview_continuation_paths as geometry
    monkeypatch.setattr(geometry, 'load_path', lambda path, data: [[0, 0], [1, 0]])
    calls = []
    def original_planner(path, reference, family, run, protocol, binding, role):
        calls.append(path)
        return dict(path_id=path, success=False, failure_reasons=['full_rest_missing'], elapsed_time_s=.01)
    monkeypatch.setattr(study.original, 'run_one', original_planner)
    (tmp_path/'protocol.json').write_text('{}', encoding='utf-8')
    result = study.evaluate(tmp_path/'freeze.json')
    assert events == ['gate', 'geometry']
    assert len(calls) == len(set(calls)) == 12
    assert result['count'] == 12 and result['successes'] == 0
    assert all(r['failure_reasons'] == ['full_rest_missing'] for r in result['rows'])
    assert result['deterministic_runs_per_path'] == 1
