"""The candidate changes initializer identity only and still learns preview."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PPO_project'),str(ROOT/'PPO_project/tools')]
import preview_fixed_prior_study as candidate
from test_preview_full_batch import ShortEpisodes, assert_tree_equal


@pytest.fixture(autouse=True)
def one_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_only_prior_identity_and_descriptive_metadata_change():
    before=candidate.BASE_CONFIG('preview_time','adaptive',101)
    after=candidate.configuration('preview_time','adaptive',101)
    allowed={'study','init_checkpoint','init_checkpoint_sha256','initialization','candidate_change',
             'validation_milestones','zero_step_role','registered_additional_budget'}
    for key in before:
        if key!='revision':assert before[key]==after[key]
    for key in before['revision'].keys()|after['revision'].keys():
        if key not in allowed:assert before['revision'][key]==after['revision'][key]
    assert before['revision']['init_checkpoint']!=after['revision']['init_checkpoint']
    assert after['revision']['init_checkpoint'].endswith('preview_time_fixed3_seed101/checkpoint.pt')
    assert after['revision']['active_action_dims']==[1,2]
    assert after['revision']['adaptive_preview'] is True


def test_actor_critic_exactly_loaded_from_fixed_prior_with_fresh_adam():
    cfg=candidate.configuration('preview_time','adaptive',101)
    env=type('Env',(),{'observation_dim':26})()
    agent=candidate.full.local.make_agent(cfg,env)
    prior=torch.load(candidate.PRIOR/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert_tree_equal(agent.actor.state_dict(),prior['actor'])
    assert_tree_equal(agent.critic.state_dict(),prior['critic'])
    assert agent.actor_optimizer.state_dict()['state']=={}
    assert agent.critic_optimizer.state_dict()['state']=={}


def test_original_physical_feed_mapping_retained_but_preview_is_active():
    cfg=candidate.configuration('preview_time','adaptive',101)
    env=type('Env',(),{'observation_dim':26})()
    adaptive=candidate.full.local.make_agent(cfg,env)
    fixed_cfg=candidate.full.local.sensor.configuration('preview_time','fixed3',101)
    fixed=candidate.full.core.make_agent(fixed_cfg,env)
    prior=torch.load(candidate.PRIOR/'checkpoint.pt',map_location='cpu',weights_only=False)
    fixed.actor.load_state_dict(prior['actor'])
    for state in np.random.default_rng(19).normal(0,.1,(16,26)).astype('float32'):
        before=fixed.take_action(state,deterministic=True)
        after=candidate.full.local.positive.execute_action(adaptive.take_action(state,deterministic=True))
        np.testing.assert_allclose(after,before,atol=2e-8,rtol=1e-6)
    samples=[adaptive.take_action(np.zeros(26,dtype='float32'),return_info=True) for _ in range(64)]
    assert np.std([action[2] for action,_ in samples])>.05
    assert all(np.array_equal(info['action_mask'],[0.,1.,1.]) for _,info in samples)


def test_installer_restores_bindings_and_refuses_nested_calls(tmp_path,monkeypatch):
    monkeypatch.setattr(candidate,'OUT',tmp_path/'study')
    names=('OUT','configuration','source_state','prepare')
    before={name:getattr(candidate.full,name) for name in names}
    core_before={name:getattr(candidate.full.core,name) for name in (*names,'MacroEnvironment','make_agent')}
    with pytest.raises(RuntimeError,match='deliberate'):
        with candidate.installed():
            assert candidate.full.configuration is candidate.configuration
            assert candidate.full.core.MacroEnvironment is candidate.full.local.LocalCornerEnvironment
            with pytest.raises(RuntimeError,match='nest'):
                with candidate.installed():pass
            raise RuntimeError('deliberate')
    assert all(getattr(candidate.full,k) is v for k,v in before.items())
    assert all(getattr(candidate.full.core,k) is v for k,v in core_before.items())


def test_installer_rejects_foreign_feed_wrapper(monkeypatch):
    monkeypatch.setattr(candidate.cnc_env,'apply_kinematic_constraints',lambda *args:None)
    with pytest.raises(RuntimeError,match='fresh process'):
        with candidate.installed():pass


@pytest.mark.parametrize('decisions,stage',[(2048,'development'),(4096,'development'),(16,'development'),(17,'smoke'),(0,'formal')])
def test_budget_and_stage_gate_precedes_training(decisions,stage):
    with pytest.raises(ValueError,match='Registered'):candidate.train(decisions,stage)


def test_prepare_rejects_changed_original_source(tmp_path,monkeypatch):
    monkeypatch.setattr(candidate,'OUT',tmp_path/'study')
    monkeypatch.setattr(candidate,'BASE_SOURCE',lambda:{'source_sha256':'changed'})
    with pytest.raises(ValueError,match='full_batch source changed'):candidate.prepare()
    assert not (candidate.OUT/'protocol.json').exists()


def test_real_loaded_initializer_resumes_exactly_and_rejects_wrong_source(tmp_path):
    cfg=candidate.configuration('preview_time','adaptive',901)
    cfg['revision']['training_path_ids']=['three','five']
    factory=candidate.full.local.make_agent
    first=candidate.full.FullBatchTrainer(cfg,tmp_path/'complete',ShortEpisodes,factory,'own-prior-source')
    first.advance(128)
    expected_rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
    partial=candidate.full.FullBatchTrainer(cfg,tmp_path/'resume',ShortEpisodes,factory,'own-prior-source')
    partial.advance(17)
    resumed=candidate.full.FullBatchTrainer(cfg,tmp_path/'resume',ShortEpisodes,factory,'own-prior-source')
    resumed.advance(128)
    for name in ('actor','critic','actor_optimizer','critic_optimizer'):
        assert_tree_equal(getattr(first.agent,name).state_dict(),getattr(resumed.agent,name).state_dict())
    assert_tree_equal(first.updates,resumed.updates)
    assert_tree_equal(first.episodes,resumed.episodes)
    assert_tree_equal(expected_rng,(torch.get_rng_state(),np.random.get_state(),random.getstate()))
    with pytest.raises(ValueError,match='source mismatch'):
        candidate.full.FullBatchTrainer(cfg,tmp_path/'resume',ShortEpisodes,factory,'changed')


def test_selection_preserves_step0_and_never_ranks_incomplete_times():
    initial=dict(step=0,successes=3,complete_time_s=20.)
    incomplete=dict(step=1024,successes=2,complete_time_s=None)
    assert min([initial,incomplete],key=candidate.selection_key) is initial
    faster=dict(step=512,successes=3,complete_time_s=19.)
    assert min([initial,faster],key=candidate.selection_key) is faster
