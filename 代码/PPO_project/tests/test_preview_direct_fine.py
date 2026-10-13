"""The new certificate is shared, ordered, and isolated from frozen versions."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'PPO_project/tools'))
import preview_direct_fine as fine
from src.utils import revision_metrics as metrics


@pytest.fixture
def installed(monkeypatch):
    for obj,names in [(fine.study,['OUT','config_for','source_state','protocol_definition']),
                      (fine.study.v2,['make_env']),(metrics,['ordered_projection'])]:
        for name in names:monkeypatch.setattr(obj,name,getattr(obj,name))
    yield fine.install()


def load_trace(relative):
    z=np.load(ROOT/'论文项目/revision_20260908'/relative)
    d=dict(zip(z['columns'].tolist(),z['data'].T))
    return np.c_[d['x_mm'],d['y_mm']]


def test_import_does_not_activate_adapters():
    code="""import sys
sys.path.insert(0,'PPO_project/tools')
import preview_direct_research as direct
from src.utils import revision_metrics as metrics
state=(direct.OUT,direct.config_for,direct.source_state,direct.protocol_definition,direct.v2.make_env,metrics.ordered_projection)
import preview_direct_fine
import preview_direct_fine_batch
assert state==(direct.OUT,direct.config_for,direct.source_state,direct.protocol_definition,direct.v2.make_env,metrics.ordered_projection)
"""
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


@pytest.mark.parametrize('variant,family,expected',[
    ('fixed_mid','dense_corners',True),('fixed_center','mixed_lengths',True),('fixed_long','dense_corners',False)])
def test_online_offline_progress_matches_every_sample(installed,variant,family,expected):
    reference=np.loadtxt(ROOT/f'PPO_project/data/preview_v2/paths/{family}_validation.csv',delimiter=',',skiprows=1)
    xy=load_trace(f'preview_direct_gain/pilot/{variant}_seed101/evaluation_validation_nominal/{family}_validation.npz')
    objective=fine.FineTimeObjective(SimpleNamespace(half_epsilon=1.25,current_position=xy[0]),
        {'revision':{'ordered_witness_spacing_mm':.01}},reference)
    online=[objective.coverage.snapshot()['ordered_progress']]
    online.extend(objective.coverage.update(p)['ordered_progress'] for p in xy[1:])
    offline=metrics.ordered_projection(reference,xy,tolerance_mm=1.25)
    np.testing.assert_allclose(online,offline['ordered_progress'],rtol=0,atol=1e-12)
    assert objective.coverage.snapshot()['ordered_coverage_complete'] is expected
    assert offline['ordered_coverage_complete'] is expected
    assert not fine.BASE_ORDERED_PROJECTION(reference,xy,tolerance_mm=1.25)['ordered_coverage_complete']


def test_motion_constraints_are_identical_and_not_relaxed(installed):
    base=fine.gain.config_for('direct',11)
    new=fine.config_for('direct',11)
    for key in ['environment','kinematic_constraints','reward_weights','ppo']:
        assert base[key]==new[key]
    reference=np.array([[0.,0.],[1.,0.]])
    xy=np.array([[0.,0.],[.1,0.],[.5,0.]])
    result=metrics.evaluate_trajectory(reference,xy,[0.,.001,.002],tolerance_mm=1.25,
        limits=new['kinematic_constraints'],initial_velocity=[0.,0.],initial_acceleration=[0.,0.])
    assert not result['success']
    for key in ['feed_mm_s','feed_acceleration_mm_s2','feed_jerk_mm_s3']:
        assert result['constraints'][key]['satisfied'] is False
    assert result['witness_spacing_mm']==.01
    assert result['witness_match_radius_mm']==1.245
    assert new['environment']['epsilon']==2.5


def test_make_env_and_install_are_consistent(installed):
    assert fine.install() is installed
    cfg=fine.config_for('direct',101)
    reference=np.array([[0.,0.],[1.,0.],[1.,1.]])
    env,observation=installed.make_env(cfg,'smoke_fixture',reference=reference)
    assert isinstance(env.env,fine.FineTimeObjective)
    assert env.coverage.spacing_mm==.01
    assert env.coverage.radius_mm==1.245
    assert observation.shape==(20,)
    env.reset(random_start=False)
    assert env.coverage.spacing_mm==.01


def test_registered_budget_and_all_methods_share_fine_spacing(installed):
    for variant in installed.PILOT_VARIANTS:
        cfg=installed.config_for(variant,101)
        assert cfg['revision']['ordered_witness_spacing_mm']==.01
        assert cfg['reward_weights']['control_authority']['tangent_blend']==.015
    protocol=installed.protocol_definition()
    assert protocol['pilot_steps']==32768
    assert protocol['formal_steps']==131072
    assert protocol['formal_seeds']==[11,22,33,44,55]
    assert len(protocol['pilot_variants'])==6
    assert len(protocol['formal_variants'])==3
    assert protocol['geometry_distance_rule_used'] is False
