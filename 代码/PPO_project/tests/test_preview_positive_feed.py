"""Check transformed execution, PPO densities and saved-model continuity."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PPO_project'),str(ROOT/'PPO_project/tools')]
from preview_macro_ppo import PreviewMacroPPO
from preview_positive_feed_ppo import PositiveFeedPPO
from preview_positive_feed_study import execute_action, PositiveFeedEnvironment, configuration


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def agent_options():
    return dict(state_dim=26,hidden_dim=32,action_dim=3,epochs=2,eps=.1,
                ent_coef=.0003,device=torch.device('cpu'),active_action_dims=(1,2))


def test_loaded_policy_has_identical_deterministic_physical_feed_and_preview():
    old=PreviewMacroPPO(**agent_options())
    new=PositiveFeedPPO(**agent_options())
    new.actor.load_state_dict(old.actor.state_dict())
    for state in np.random.default_rng(12).normal(size=(20,26)).astype('float32'):
        expected=old.take_action(state,deterministic=True)
        actual=execute_action(new.take_action(state,deterministic=True))
        np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=1e-6)


def test_samples_execute_strictly_positive_without_changing_scored_latents():
    torch.manual_seed(23)
    agent=PositiveFeedPPO(**agent_options())
    state=np.zeros(26,dtype='float32')
    with torch.no_grad():
        mu,std=agent.actor(torch.from_numpy(state[None]))
        latent=torch.distributions.Normal(mu,std).sample((10000,))[:,0,:]
    executed=np.array([execute_action(row.numpy()) for row in latent])
    assert np.all((executed[:,1]>0)&(executed[:,1]<1))
    assert np.any(latent[:,1].numpy()<0)
    action,info=agent.take_action(state,return_info=True)
    expected=torch.distributions.Normal(mu,std).log_prob(torch.tensor(action[None]))[0,1:].sum()
    assert info['old_log_prob']==pytest.approx(float(expected),abs=1e-6)


def test_rollout_update_checks_original_latents_and_rejects_executed_substitution():
    torch.manual_seed(28)
    agent=PositiveFeedPPO(**agent_options())
    states=np.random.default_rng(41).normal(0,.1,(12,26)).astype('float32')
    actions=[];probabilities=[];masks=[]
    for state in states:
        action,info=agent.take_action(state,return_info=True)
        actions.append(action);probabilities.append(info['old_log_prob']);masks.append(info['action_mask'])
    batch=dict(states=states,next_states=states+.01,actions=actions,rewards=np.linspace(-.2,.3,12),
               dones=[False]*11+[True],terminated=[False]*11+[True],old_log_probs=probabilities,action_masks=masks)
    wrong=dict(batch,actions=[execute_action(a) for a in actions])
    with pytest.raises(ValueError,match='old_log_probs'):
        agent.update(wrong)
    assert all(np.isfinite(agent.update(batch)))
    assert agent.last_update_diagnostics['old_log_probability_max_absolute_error']<1e-4


def test_environment_executes_feed_sigmoid_and_records_logit_without_rewriting_actions():
    cfg=configuration('preview_time','adaptive',901)
    env=PositiveFeedEnvironment(cfg,'unit',reference=np.array([[0.,0.],[8.,0.]]))
    latent=np.array([0.,-3.,.5]);before=latent.copy()
    env.step(latent)
    np.testing.assert_array_equal(latent,before)
    assert env.trace[-1]['feed_executed_action']==pytest.approx(1/(1+np.exp(3)))
    assert env.trace[-1]['feed_logit']==-3
    assert env.trace[-1]['lookahead_mm']==pytest.approx(2.7)
