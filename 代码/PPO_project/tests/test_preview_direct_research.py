"""Direct-distance control, anchored PPO gradients, and isolated study gates."""
import copy
import importlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import preview_direct_research as study
import preview_direct_batch as batch


REFERENCE=np.array([[0.,0.],[100.,0.]])


def environment(variant='direct'):
    cfg=study.config_for(variant,101)
    cfg['ppo'].update(hidden_dim=32,epochs=1)
    env,obs=study.make_env(cfg,'synthetic',reference=REFERENCE)
    return cfg,env,obs


@pytest.mark.parametrize('u,distance',[(-2.,.8),(0.,.8),(.5,2.7),(1.,4.6),(4.,4.6)])
def test_direct_policy_reaches_whole_distance_bounds(u,distance):
    _,env,_=environment()
    observation,_,_,info=env.step([0.,.05,u])
    assert study.direct_distance(u)==pytest.approx(distance)
    assert env.raw._lookahead_dist_active==pytest.approx(distance)
    assert info['control_status']['direct_selected_mm']==pytest.approx(distance)
    np.testing.assert_array_equal(observation[:18],env.raw.state)
    np.testing.assert_allclose(observation[-env.raw.lookahead_obs_dim-2:-2],env.raw.get_lookahead_observation())


def test_altering_or_disabling_geometry_features_cannot_replace_same_policy_u():
    _,first,_=environment();_,second,_=environment()
    first.geometry.at=lambda s:dict(q=0.,range_over_pi=0.)
    second.geometry.at=lambda s:dict(q=1.,range_over_pi=3.)
    second.raw.lookahead_control_straight_dist=.8
    second.raw.lookahead_control_corner_dist=4.6
    obs1,_,_,_=first.step([.01,.05,.9])
    obs2,_,_,_=second.step([.01,.05,.9])
    assert first.raw._lookahead_dist_active==second.raw._lookahead_dist_active==pytest.approx(4.22)
    np.testing.assert_array_equal(first.current_position,second.current_position)
    np.testing.assert_array_equal(obs1[:18],obs2[:18])
    np.testing.assert_array_equal(obs1[-2:],[0.,0.]);np.testing.assert_array_equal(obs2[-2:],[1.,3.])
    assert set(study.GeometryObservation(REFERENCE).at(0.))=={'q','range_over_pi'}


def test_extra_features_query_next_position_and_reset_returns_initial_three_mm():
    _,env,_=environment()
    queried=[]
    env.geometry.at=lambda s:(queried.append(s) or dict(q=.2,range_over_pi=.4))
    obs,_,_,info=env.step([0.,.05,0.])
    assert queried[-1]==pytest.approx(env.raw._project_onto_progress_path(env.current_position)[2])
    assert queried[-1]>0.
    np.testing.assert_allclose(obs[-2:],[.2,.4])
    assert info['control_status']['direct_mapping_us']>=0.
    assert info['control_status']['direct_observation_features_us']>=0.
    env.reset()
    assert env.raw._lookahead_dist_active==3. and env.state.shape==(20,)


@pytest.mark.parametrize('variant',study.PILOT_VARIANTS)
def test_fixed_and_direct_share_observations_initial_physics_and_anchor(variant):
    cfg,env,obs=environment(variant)
    agent=study.make_agent(cfg,env)
    assert env.observation_dim==20 and env.raw.observation_dim==18
    assert not env.raw.lookahead_control_enabled and not cfg['revision']['geometry_distance_rule_used']
    assert agent.active_action_dims==((0,1,2) if variant.startswith('direct') else (0,1))
    assert agent.anchor_beta==(0. if variant=='direct_no_anchor' else 1.)
    assert cfg['ppo']['actor_lr']==1e-5 and agent.lmbda==1.
    assert agent.gamma==pytest.approx(math.exp(-.001/10.))
    with torch.no_grad():mu,std=agent.actor(torch.as_tensor(obs).unsqueeze(0))
    np.testing.assert_allclose(mu[0],[0.,.05,(3.-.8)/3.8],atol=1e-7)
    np.testing.assert_allclose(std[0],[.01,.01,.01],atol=1e-7)
    assert agent.motion_anchor_kl(mu,std).item()==pytest.approx(0.,abs=1e-10)
    if variant in study.DISTANCES:
        env.step([0.,.05]);assert env.raw._lookahead_dist_active==study.DISTANCES[variant]


def test_motion_anchor_has_zero_preview_mean_and_scale_gradients():
    cfg,env,_=environment();agent=study.make_agent(cfg,env)
    mu=torch.tensor([[.03,.1,.2],[.01,.06,.8]],requires_grad=True)
    std=torch.tensor([[.02,.03,.3],[.04,.02,.7]],requires_grad=True)
    value=agent.motion_anchor_kl(mu,std)
    value.backward()
    assert torch.all(mu.grad[:,2]==0.) and torch.all(std.grad[:,2]==0.)
    assert torch.any(mu.grad[:,:2]!=0.) and torch.any(std.grad[:,:2]!=0.)
    changed_mu=mu.detach().clone();changed_mu[:,2]+=50.
    changed_std=std.detach().clone();changed_std[:,2]*=100.
    assert agent.motion_anchor_kl(changed_mu,changed_std).item()==value.item()


def transitions(agent,n=16):
    random=np.random.default_rng(12)
    states=random.normal(size=(n,20)).astype(np.float32)
    signs=np.tile([-1.,1.],n//2)
    u=(3.-.8)/3.8
    actions=np.c_[np.zeros(n),np.full(n,.05),u+.02*signs]
    return dict(states=states,actions=actions,next_states=states,rewards=10.*signs,
                dones=np.ones(n),terminated=np.ones(n))


@pytest.mark.parametrize('variant',('direct','direct_no_anchor','fixed_center'))
def test_policy_gradient_reaches_preview_head_only_when_active(variant):
    torch.manual_seed(15)
    cfg,env,_=environment(variant);agent=study.make_agent(cfg,env)
    before=agent.actor.mu_head.bias.detach().clone()
    loss=agent.update(transitions(agent))
    assert np.isfinite(loss).all()
    diagnostic=agent.last_update_diagnostics['epochs'][0]
    if variant.startswith('direct'):
        assert diagnostic['preview_mu_weight_gradient_norm']>0.
        assert abs(diagnostic['preview_mu_bias_gradient'])>0.
        assert agent.actor.mu_head.bias[2].item()!=before[2].item()
    else:
        assert diagnostic['preview_mu_weight_gradient_norm']==0.
        assert diagnostic['preview_mu_bias_gradient']==0.
    assert diagnostic['actor_loss']==pytest.approx(diagnostic['policy_loss']-diagnostic['entropy_bonus']+diagnostic['anchor_loss'],abs=1e-6)
    assert diagnostic['anchor_loss']==pytest.approx(agent.anchor_beta*diagnostic['motion_anchor_kl'])


def test_latent_sampling_and_scoring_correspond_before_execution_clipping():
    cfg,env,obs=environment();agent=study.make_agent(cfg,env)
    state=torch.as_tensor(obs).unsqueeze(0)
    with torch.no_grad():mu,std=agent.actor(state)
    torch.manual_seed(144);expected=torch.distributions.Normal(mu,std).sample()[0].numpy()
    torch.manual_seed(144);actual=np.asarray(agent.take_action(obs))
    np.testing.assert_array_equal(actual,expected)
    latent=torch.tensor([[1.1,-.01,1.2]])
    score=agent.latent_log_prob(mu,std,latent)
    expected_score=torch.distributions.Normal(mu,std).log_prob(latent).sum(1,keepdim=True)
    torch.testing.assert_close(score,expected_score)
    clipped=torch.tensor([[1.,0.,1.]])
    assert not torch.allclose(score,agent.latent_log_prob(mu,std,clipped))
    cfg,env,obs=environment('fixed_center');fixed=study.make_agent(cfg,env)
    with torch.no_grad():mu,std=fixed.actor(torch.as_tensor(obs).unsqueeze(0))
    changed=latent.clone();changed[:,2]=-100.
    torch.testing.assert_close(fixed.latent_log_prob(mu,std,latent),fixed.latent_log_prob(mu,std,changed))


def test_critic_update_disables_dropout_but_keeps_parameter_learning():
    cfg,env,_=environment();agent=study.make_agent(cfg,env)
    before={name:value.detach().clone() for name,value in agent.critic.named_parameters()}
    agent.update(transitions(agent))
    assert not agent.critic.training
    states=torch.tensor(np.random.default_rng(13).normal(size=(16,20)),dtype=torch.float32)
    first=agent.critic(states)
    for _ in range(4):
        torch.testing.assert_close(agent.critic(states),first,rtol=0.,atol=0.)
    assert first.requires_grad
    assert any(not torch.equal(before[name],value.detach()) for name,value in agent.critic.named_parameters())


def test_beta_zero_matches_original_ppo_with_common_critic_eval_setting():
    cfg,env,_=environment('direct_no_anchor');anchored=study.make_agent(cfg,env)
    keys=('hidden_dim','actor_lr','critic_lr','lmbda','epochs','eps','gamma','ent_coef')
    original=study.PPOContinuous(state_dim=20,action_dim=3,device=torch.device('cpu'),active_action_dims=[0,1,2],
        **{k:cfg['ppo'][k] for k in keys})
    original.actor.load_state_dict(anchored.actor.state_dict())
    original.critic.load_state_dict(anchored.critic.state_dict())
    original.critic.eval()
    data=transitions(anchored)
    expected=original.update(data);actual=anchored.update(data)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
    for name,value in original.actor.state_dict().items():
        torch.testing.assert_close(anchored.actor.state_dict()[name],value,rtol=1e-6,atol=1e-7)
    for name,value in original.critic.state_dict().items():
        torch.testing.assert_close(anchored.critic.state_dict()[name],value,rtol=1e-6,atol=1e-7)


def test_import_has_no_backend_activation_and_source_includes_batch():
    keys=('OUT','make_env','config_for','sample_row')
    before={k:getattr(study.backend,k) for k in keys}
    importlib.reload(study)
    assert before=={k:getattr(study.backend,k) for k in keys}
    sources=study.source_state()['source_files']
    assert 'PPO_project/tools/preview_direct_research.py' in sources
    assert 'PPO_project/tools/preview_direct_batch.py' in sources
    assert 'PPO_project/tools/preview_research_v2.py' in sources
    assert 'PPO_project/src/algorithms/ppo.py' in sources
    assert 'PPO_project/tools/preview_gate_research.py' not in sources


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    record=dict(paths=[dict(id=f'train_{i}',split='train') for i in range(3)]
        +[dict(id=f'validation_{i}',split='validation') for i in range(3)]+[dict(id='never_read_test',split='test')])
    study.backend.write_json(data/'manifest.json',record)
    monkeypatch.setattr(study,'DATA',data);monkeypatch.setattr(study,'OUT',tmp_path/'runs')
    monkeypatch.setattr(study,'manifest',lambda:copy.deepcopy(record))
    def path(name):
        assert name!='never_read_test','Test geometry must not be read'
        return REFERENCE
    monkeypatch.setattr(study,'path_points',path)
    original=study.config_for
    def config(variant,seed):
        cfg=original(variant,seed);cfg['revision']['episode_max_steps']=8;cfg['ppo'].update(hidden_dim=32,epochs=1)
        return cfg
    monkeypatch.setattr(study,'config_for',config)
    monkeypatch.setattr(study,'source_state',lambda:dict(source_sha256='test-direct',source_files={}))
    for key in ('OUT','DEFAULT_DATA_DIR','load_manifest','load_path','config_for','make_env','make_agent','source_state','sample_row'):
        monkeypatch.setattr(study.backend,key,getattr(study.backend,key))
    study.activate_backend();study.prepare()
    return study


def test_actual_training_resume_gradient_logs_validation_and_test_guard(isolated):
    run=isolated.train('direct',7,'smoke',17)
    before={str(p.relative_to(run)):p.read_bytes() for p in run.rglob('*') if p.is_file()}
    isolated.train('direct',7,'smoke',17)
    assert before=={str(p.relative_to(run)):p.read_bytes() for p in run.rglob('*') if p.is_file()}
    isolated.train('direct',7,'smoke',25)
    checkpoint=torch.load(run/'checkpoint.pt',weights_only=False)
    assert checkpoint['steps']==25 and checkpoint['actor']['shared_fc1.weight'].shape[1]==20
    updates=[json.loads(line) for line in (run/'actor_update_diagnostics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert updates[-1]['step']==25 and updates[-1]['checkpoint_start_steps']==17
    assert all('motion_anchor_kl' in r['epochs'][0] and 'preview_mu_bias_gradient' in r['epochs'][0] for r in updates)
    rows=isolated.evaluate(run,'validation')
    assert len(rows)==3 and all(r['sample_count']==9 for r in rows)
    checked,checked_rows=isolated._validated_run('smoke','direct',7,25)
    assert checked==run and checked_rows==rows
    trace=np.load(run/'evaluation_validation_nominal/validation_0.npz')
    assert 'control_status__direct_mapping_us' in trace['columns']
    with pytest.raises(FileNotFoundError):isolated.evaluate(run,'test')
    with pytest.raises(ValueError,match='Exact registered'):isolated._validated_run('smoke','direct',7,24)


def test_protocol_source_change_and_budget_are_rejected(isolated,monkeypatch):
    with pytest.raises(ValueError,match='Unregistered'):isolated.train('direct',7,'pilot',32)
    with pytest.raises(ValueError,match='Select fixed'):isolated.train('direct',11,'formal',32)
    monkeypatch.setattr(isolated,'source_state',lambda:dict(source_sha256='changed',source_files={}))
    with pytest.raises(ValueError,match='Protocol/source'):isolated.train('direct',101,'pilot',32)


def test_select_all_six_and_freeze_all_fifteen_models(isolated,monkeypatch):
    calls=[]
    def validated(stage,variant,seed,budget):
        calls.append((stage,variant,seed,budget))
        run=isolated.OUT/stage/f'{variant}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
        (run/'checkpoint.pt').write_bytes(f'{variant} {seed}'.encode())
        (run/'config.yaml').write_text(f'method: {variant}\n',encoding='utf-8')
        ok=variant in ('fixed_center','selected_fixed','direct','direct_no_anchor')
        return run,[dict(success=ok,path_completed=ok,geometry_feasible=True,ordered_progress=1. if ok else .2,elapsed_time_s=4.)]*3
    monkeypatch.setattr(isolated,'_validated_run',validated)
    selection=isolated.select();choice=json.loads(selection.read_text(encoding='utf-8'))
    assert choice['selected_fixed']=='fixed_center' and len(calls)==6
    assert isolated.config_for('selected_fixed',11)['revision']['fixed_distance_mm']==3.
    frozen=json.loads(isolated.freeze().read_text(encoding='utf-8'))
    assert len(frozen['models'])==len(frozen['configs'])==15
    assert len(calls)==21 and all(c[-1]==131072 for c in calls[6:])
    assert frozen['primary_adaptive_candidate']=='direct' and frozen['anchor_ablation']=='direct_no_anchor'
    with pytest.raises(FileExistsError):isolated.freeze()


def test_batch_training_only_requests_validation(monkeypatch,tmp_path):
    invoked=[]
    monkeypatch.setattr(batch.study,'OUT',tmp_path)
    monkeypatch.setattr(batch,'call',lambda args,*others:invoked.append(args))
    batch.train('direct',101,'pilot',32768)
    assert len(invoked)==2 and invoked[0][0]=='train'
    assert invoked[1][-1]=='validation' and 'test' not in invoked[1]
