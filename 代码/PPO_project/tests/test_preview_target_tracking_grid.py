"""Grid scope and pooled selection tests; no training or external geometry."""
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
import preview_target_tracking_grid as study


def node(step,time=5.,success=3):
    return dict(step=step,checkpoint=f'model{step}',checkpoint_sha256=str(step),
        rows=[dict(path_id=p,success=i<success,elapsed_time_s=time) for i,p in enumerate(study.audit.PATH_IDS)])


def groups():
    return {method:[node(step) for step in study.STEPS] for method in study.FIXED_DISTANCES}


def test_uniform_distance_uses_all_three_paths_and_all_nodes():
    data=groups();data['fixed4'][2]=node(512,time=4.)
    winner,_=study.select_uniform_fixed(data)
    assert winner['method']=='fixed4' and winner['selected_step']==512


def test_per_path_fixed_distance_or_failed_times_cannot_win():
    data=groups();data['fixed1p2']=[node(step,time=.01,success=2) for step in study.STEPS]
    winner,_=study.select_uniform_fixed(data)
    assert winner['method']=='fixed3' and winner['success_count']==3


def test_eligible_step0_cannot_be_discarded():
    data=groups();data['fixed3'][0]=node(0,time=4.)
    winner,_=study.select_uniform_fixed(data)
    assert winner['method']=='fixed3' and winner['selected_step']==0


def test_missing_grid_member_or_node_rejected():
    data=groups();del data['fixed4']
    with pytest.raises(ValueError):study.select_uniform_fixed(data)
    data=groups();data['fixed4'].pop()
    with pytest.raises(ValueError):study.select_uniform_fixed(data)


@pytest.mark.parametrize('method,seed',[('adaptive',101),('fixed3',101),('fixed4',11)])
def test_cached_methods_and_new_seeds_cannot_train(method,seed):
    with pytest.raises(ValueError):study.configuration('preview_time',method,seed)


def test_nested_configuration_differences_remain_visible():
    assert study.differences({'revision':{'D':3.},'reward':{'w':1.}},
        {'revision':{'D':4.},'reward':{'w':2.}})=={'revision.D':[3.,4.],'reward.w':[1.,2.]}


@pytest.mark.parametrize('a,f,original,dpassed,expected',[
    (13.,14.,13.931,True,True),(13.,12.,13.931,True,False),
    (13.94,15.,13.931,True,False),(13.,14.,13.931,False,False)])
def test_4096_requires_every_registered_gate(a,f,original,dpassed,expected):
    report=study.eligibility({'success_count':3,'complete_total_time_s':a},
        {'success_count':3,'complete_total_time_s':f},original,dpassed)
    assert report['eligible4096']==expected
    assert not report['adopted'] and not report['executed4096']


def test_incomplete_actor_never_opens_larger_budget():
    report=study.eligibility({'success_count':2,'complete_total_time_s':None},
        {'success_count':3,'complete_total_time_s':14.},13.931,True)
    assert not report['eligible4096']


def test_binding_checks_the_verified_parent(monkeypatch,tmp_path):
    rows=node(256)['rows']
    monkeypatch.setattr(study.audit,'sha',lambda _: 'parent')
    monkeypatch.setattr(study.audit,'read',lambda p: {'source_sha256':'parent'} if p.name=='provenance.json'
        else {'checkpoint_sha256':'wrong','config_sha256':'parent','source_sha256':'parent'})
    with pytest.raises(ValueError,match='verified parent'):
        study.validate_cached_binding(tmp_path,rows,256,'fixed3')


def test_scoped_grid_bindings_restore_on_error():
    original=study.full.OUT;fn=study.audit.cnc_env.apply_kinematic_constraints
    with pytest.raises(RuntimeError,match='test failure'):
        with study.installed():
            assert study.full.OUT==study.OUT
            raise RuntimeError('test failure')
    assert study.full.OUT==original and study.audit.cnc_env.apply_kinematic_constraints is fn
