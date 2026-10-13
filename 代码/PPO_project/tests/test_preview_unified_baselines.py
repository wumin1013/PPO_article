"""Unified legacy baselines must retain controls and fail closed before test access."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'PPO_project/tools'))
import preview_unified_baselines as tool


def frozen_fixture(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}',encoding='utf-8')
    monkeypatch.setattr(tool.v2,'DATA',tmp_path)
    models={};configs={}
    for key in ['run_a','run_b']:
        path=tmp_path/key;path.mkdir()
        (path/'checkpoint.pt').write_bytes(key.encode())
        (path/'config.yaml').write_text('epsilon: 2.5\n',encoding='utf-8')
        models[key]=tool.sha(path/'checkpoint.pt');configs[key]=tool.sha(path/'config.yaml')
    value={'models':models,'configs':configs,'path_manifest_sha256':tool.sha(manifest),'source_sha256':'source'}
    (tmp_path/'freeze.json').write_text(json.dumps(value),encoding='utf-8')
    return value


def test_import_does_not_activate_legacy_or_final_code():
    code="""import sys
sys.path.insert(0,'PPO_project/tools')
import preview_research_v2 as v2
from src.utils import revision_metrics as metrics
before=(v2.OUT,v2.make_env,v2.backend.OUT,v2.backend.make_env,metrics.ordered_projection)
import preview_unified_baselines
assert before==(v2.OUT,v2.make_env,v2.backend.OUT,v2.backend.make_env,metrics.ordered_projection)
"""
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_every_frozen_model_is_checked(tmp_path,monkeypatch):
    frozen_fixture(tmp_path,monkeypatch)
    tool.verify_freeze(tmp_path,{'run_a','run_b'},source_sha256='source')
    (tmp_path/'run_b/checkpoint.pt').write_bytes(b'changed second model')
    with pytest.raises(ValueError,match='run_b'):
        tool.verify_freeze(tmp_path,{'run_a','run_b'},source_sha256='source')


def test_missing_registered_model_cannot_pass(tmp_path,monkeypatch):
    frozen_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='complete registered'):
        tool.verify_freeze(tmp_path,{'run_a','run_b','run_c'},source_sha256='source')


@pytest.mark.parametrize('count',[3,4])
def test_final_roster_is_derived_from_its_protocol(tmp_path,monkeypatch,count):
    variants=['selected_fixed','direct','direct_no_anchor','selected_fixed_no_anchor'][:count]
    seeds=[11,22,33,44,55]
    (tmp_path/'freeze.json').write_text('{}',encoding='utf-8')
    (tmp_path/'protocol.json').write_text(json.dumps({'formal_variants':variants,'formal_seeds':seeds}),encoding='utf-8')
    candidate=SimpleNamespace(OUT=tmp_path,source_state=lambda:{'source_sha256':'source'})
    verifier=Mock(return_value={})
    monkeypatch.setattr(tool,'verify_freeze',verifier)
    tool.validate_final(candidate)
    required={f'formal/{variant}_seed{seed}' for variant in variants for seed in seeds}
    verifier.assert_called_once_with(tmp_path,required,source_sha256='source',verify_protocol=True)
    assert len(required)==count*5


@pytest.mark.parametrize('study',['direct-fine','direct-explore'])
def test_test_cli_refuses_before_any_path_or_policy_access(tmp_path,monkeypatch,study):
    candidate=SimpleNamespace(OUT=tmp_path,source_state=lambda:{'source_sha256':'unused'})
    monkeypatch.setattr(tool,'final_study',lambda name:candidate)
    sentinel=Mock(side_effect=AssertionError('No path/policy access before complete freeze'))
    monkeypatch.setattr(tool.v2,'manifest',sentinel)
    monkeypatch.setattr(tool,'baseline_runs',sentinel)
    monkeypatch.setattr(tool,'transferred_rollout',sentinel)
    monkeypatch.setattr(sys,'argv',['preview_unified_baselines.py','--study',study,'evaluate','--split','test'])
    with pytest.raises(ValueError,match='freeze is required'):
        tool.main()
    sentinel.assert_not_called()
    assert not (tmp_path/'unified_baselines').exists()


def test_missing_explore_module_has_explicit_error(monkeypatch):
    def missing(name):
        error=ModuleNotFoundError(name);error.name=name;raise error
    monkeypatch.setattr(tool.importlib,'import_module',missing)
    with pytest.raises(ValueError,match='preview_direct_explore.py'):
        tool.final_study('direct-explore')


@pytest.mark.parametrize('baseline,relative',[
    ('absolute','preview_v2/finetune/absolute_seed11/config.yaml'),
    ('nnc','preview_v2/nnc_auxiliary/runs/nnc_seed11/config.yaml')])
def test_original_guidance_policy_and_kcm_are_preserved(baseline,relative):
    original=yaml.safe_load((ROOT/'论文项目/revision_20260908'/relative).read_text(encoding='utf-8'))
    before=copy.deepcopy(original)
    cfg=tool.effective_config(original,tool.fine,'nominal')
    assert cfg['reward_weights']==original['reward_weights']
    assert cfg['ppo']==original['ppo']
    assert cfg['experiment']==original['experiment']
    assert cfg['environment']['lookahead_obs_enabled']==original['environment']['lookahead_obs_enabled']
    assert cfg['environment']['lookahead_points']==original['environment']['lookahead_points']
    assert cfg['revision']['ordered_witness_spacing_mm']==.01
    assert cfg['environment']['epsilon']==2.5
    assert cfg['kinematic_constraints']==original['kinematic_constraints']
    assert cfg['reward_weights']['control_authority']['tangent_blend']==.08
    assert original==before


def test_evaluation_keeps_physical_failures_and_restores_scope():
    candidate=tool.fine.config_for('direct',101)
    rows=[dict(x_mm=x,y_mm=0.,timestamp_s=t,heading_rad=0.) for x,t in [(0.,0.),(.1,.001),(.5,.002)]]
    original_projection=tool.metrics.ordered_projection
    result=tool.evaluate_trace(np.array([[0.,0.],[1.,0.]]),rows,candidate,'success',True)
    assert tool.metrics.ordered_projection is original_projection
    assert result['witness_spacing_mm']==.01
    assert not result['success']
    assert not result['constraints']['feed_mm_s']['satisfied']
    assert not result['constraints']['feed_jerk_mm_s3']['satisfied']


def test_regression_specs_reuse_original_three_paths(monkeypatch):
    sentinel=Mock(side_effect=AssertionError('Regression must not read held-out paths'))
    monkeypatch.setattr(tool.v2,'manifest',sentinel)
    specs=tool.reference_specs('regression')
    assert len(specs)==3
    assert len({spec['id'] for spec in specs})==3
    assert all('historical_spec' in spec for spec in specs)
    sentinel.assert_not_called()
