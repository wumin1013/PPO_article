"""Unchanged baseline delegation, access order and unmodified output auditing."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PPO_project'),str(ROOT/'PPO_project/tools')]
import preview_formal_two_step as baseline


def fake_protocol():
    return dict(configuration=baseline.configuration(),source=dict(source_sha256='unit-source'),
        formal_protocol_sha256='formal-protocol',comparison_rule='Complete common successes only')


def fake_result():
    return SimpleNamespace(smoothed=SimpleNamespace(smoothed_path=np.array([[0.,0.],[1.,0.]])),
        trace_rows=[dict(time=.001,x=.001,y=0.,v=1.,a=17.,j=23.),
                    dict(time=.002,x=.002,y=0.,v=1.,a=19.,j=29.)],
        metrics=dict(termination_status='success',angular_constraints_enforced=False,internal_diagnostic=7.))


def test_default_configuration_is_the_original_algorithm():
    cfg=baseline.configuration()
    assert cfg['dt_s']==.001 and cfg['max_steps']==60000
    assert cfg['one_sided_tolerance_mm']==1.25 and cfg['tolerance_bandwidth_mm']==2.5
    assert cfg['scheduler']==baseline.asdict(baseline.legacy.SchedulerParams())
    assert cfg['smoothing']==baseline.asdict(baseline.legacy.SmoothingParams())
    assert cfg['limits']==dict(MAX_VEL=100.,MAX_ACC=2000.,MAX_JERK=62500.,
        MAX_ANG_VEL=2*np.pi,MAX_ANG_ACC=10*np.pi,MAX_ANG_JERK=1000.)
    assert baseline.legacy.run_two_step_baseline.__module__=='src.algorithms.two_step_baseline'


def test_prepare_does_not_read_any_holdout_coordinates(tmp_path,monkeypatch):
    monkeypatch.setattr(baseline,'OUT',tmp_path/'prepared')
    def forbidden(*args,**kwargs): raise AssertionError('Coordinate or freeze access during metadata prepare')
    monkeypatch.setattr(baseline.gate,'validate_freeze',forbidden)
    monkeypatch.setattr(baseline.gate,'frozen_manifest',forbidden)
    monkeypatch.setattr(baseline.formal.old,'points',forbidden)
    result=baseline.prepare()
    assert result['preparation_reads_holdout_coordinates'] is False
    assert (baseline.OUT/'source.zip').exists()
    assert baseline.prepare()==result


def test_gate_failure_happens_before_manifest_or_loader(tmp_path,monkeypatch):
    events=[]
    monkeypatch.setattr(baseline,'prepare',fake_protocol)
    def reject(*args): events.append('gate');raise ValueError('ten models incomplete')
    monkeypatch.setattr(baseline.gate,'validate_freeze',reject)
    monkeypatch.setattr(baseline.gate,'frozen_manifest',lambda *args:events.append('manifest'))
    with pytest.raises(ValueError,match='incomplete'): baseline.evaluate(tmp_path/'missing.json')
    assert events==['gate']


def test_constraint_mismatch_rejected_before_holdout_manifest(tmp_path,monkeypatch):
    protocol=fake_protocol();cfg=protocol['configuration'];events=[]
    monkeypatch.setattr(baseline,'prepare',lambda:protocol)
    signature=dict(environment=dict(interpolation_period=.002,epsilon=2.5,max_steps=60000),constraints=cfg['limits'])
    monkeypatch.setattr(baseline.gate,'validate_freeze',lambda *args:dict(common_configuration=signature))
    monkeypatch.setattr(baseline.gate,'frozen_manifest',lambda *args:events.append('manifest'))
    with pytest.raises(ValueError,match='share evaluation constraints'): baseline.evaluate(tmp_path/'f.json')
    assert events==[]


def test_initial_point_is_prepended_without_modifying_original_commands():
    result=fake_result();original=copy.deepcopy(result.trace_rows)
    positions,times,heading=baseline.command_arrays(result)
    np.testing.assert_array_equal(positions,[[0.,0.],[.001,0.],[.002,0.]])
    np.testing.assert_array_equal(times,[0.,.001,.002])
    assert heading==0. and result.trace_rows==original


def test_no_invented_rest_or_clipping_and_raw_internal_metrics_preserved(tmp_path,monkeypatch):
    result=fake_result();received={}
    def original(**kwargs):received.update(kwargs);return result
    monkeypatch.setattr(baseline.legacy,'run_two_step_baseline',original)
    record=baseline.run_one('unit',[[0.,0.],[1.,0.]],'unit',tmp_path/'run',fake_protocol(),{'unit':True},'unit fixture')
    assert received['max_steps']==60000 and received['half_epsilon']==1.25
    assert received['smoothing_params']==baseline.legacy.SmoothingParams()
    assert received['scheduler_params']==baseline.legacy.SchedulerParams()
    assert record['success'] is False and record['terminal_rest_satisfied'] is False
    assert record['appended_rest_samples']==record['altered_original_command_samples']==0
    assert record['output_sample_count']==3 and record['elapsed_time_s']==.002
    assert 'rest_window_incomplete' in record['failure_reasons']
    with np.load(tmp_path/'run/commands.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['raw_trace__a'],[17.,19.])
        np.testing.assert_array_equal(data['raw_trace__j'],[23.,29.])
        assert len(data['timestamp_s'])==3
    assert baseline.read(tmp_path/'run/internal_metrics.json')==result.metrics


def test_planner_failure_keeps_failure_record_and_empty_output_files(tmp_path,monkeypatch):
    def fail(**kwargs):raise RuntimeError('original planner failure')
    monkeypatch.setattr(baseline.legacy,'run_two_step_baseline',fail)
    row=baseline.run_one('unit',[[0.,0.],[1.,0.]],'unit',tmp_path/'run',fake_protocol(),{},'unit fixture')
    assert row['success'] is False and row['failure_reasons']==['planner_exception']
    assert row['elapsed_time_s'] is None
    assert all((tmp_path/'run'/name).exists() for name in ['commands.npz','commands.csv','raw_trace.csv','result.json'])
    with np.load(tmp_path/'run/commands.npz',allow_pickle=False) as data:
        assert data['command_positions'].shape==(0,2)


def test_cached_result_is_verified_and_never_overwritten(tmp_path,monkeypatch):
    monkeypatch.setattr(baseline.legacy,'run_two_step_baseline',lambda **kwargs:fake_result())
    args=('unit',[[0.,0.],[1.,0.]],'unit',tmp_path/'run',fake_protocol(),{'unit':True},'unit fixture')
    first=baseline.run_one(*args)
    assert baseline.run_one(*args)==first
    (tmp_path/'run/raw_trace.csv').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='result changed'):baseline.run_one(*args)


def test_all_six_synthetic_fixture_loads_follow_complete_gate(tmp_path,monkeypatch):
    import generate_preview_continuation_paths as paths
    protocol=fake_protocol();cfg=protocol['configuration'];events=[]
    monkeypatch.setattr(baseline,'OUT',tmp_path)
    baseline.write_new(tmp_path/'protocol.json',protocol)
    monkeypatch.setattr(baseline,'prepare',lambda:protocol)
    checked=dict(common_configuration=dict(environment=dict(interpolation_period=.001,epsilon=2.5,max_steps=60000),
        constraints=cfg['limits']),freeze=dict(protocol=dict(sha256='formal-protocol'),holdout_manifest=dict(sha256='fixture-data')),
        freeze_sha256='fixture-freeze')
    def validate(*args):events.append('ten_model_gate');return checked
    monkeypatch.setattr(baseline.gate,'validate_freeze',validate)
    def manifest(*args):
        events.append('manifest')
        return dict(paths=[dict(id=f'unit_fixture_{i}',family='fixture') for i in range(6)]),tmp_path
    monkeypatch.setattr(baseline.gate,'frozen_manifest',manifest)
    def load(name,data):events.append(name);return np.array([[0.,0.],[1.,0.]])
    monkeypatch.setattr(paths,'load_path',load)
    monkeypatch.setattr(baseline,'run_one',lambda path_id,*args:dict(path_id=path_id,success=False,failure_reasons=['fixture']))
    result=baseline.evaluate(tmp_path/'freeze.json')
    assert events[:2]==['ten_model_gate','manifest'] and events[2:]==[f'unit_fixture_{i}' for i in range(6)]
    assert result['count']==6 and result['successes']==0 and len(result['rows'])==6


def test_smoke_rejects_test_identifier_without_loading_points(monkeypatch):
    monkeypatch.setattr(baseline,'prepare',fake_protocol)
    monkeypatch.setattr(baseline.formal.old,'points',lambda *args:pytest.fail('No test coordinates allowed in smoke'))
    with pytest.raises(ValueError,match='existing validation'):baseline.smoke('unit_fake_test')


def test_nonfinite_internal_diagnostics_remain_explicit():
    assert baseline._json_value(dict(x=np.nan,y=np.inf,z=-np.inf))==dict(x='NaN',y='+Infinity',z='-Infinity')
