"""D-only attribution: immutable feed weights, matching density and exact resume."""
import copy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PPO_project'),str(ROOT/'PPO_project/tools')]
import preview_only_study as study
from preview_only_ppo import PreviewOnlyPPO,frozen_actor_digest
from test_preview_full_batch import ShortEpisodes,assert_tree_equal


@pytest.fixture(autouse=True)
def one_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def small_agent():
    return PreviewOnlyPPO(state_dim=26,hidden_dim=16,epochs=2,eps=.1,
                          ent_coef=.0003,device=torch.device('cpu'))


def rollout(agent,count=64):
    states=np.random.default_rng(82).normal(0,.1,(count,26)).astype('float32')
    choices=[agent.take_action(state,return_info=True) for state in states]
    return dict(states=states,actions=np.asarray([action for action,_ in choices]),
        next_states=states+.02,rewards=np.linspace(-.5,.8,count,dtype='float32'),
        dones=[False]*(count-1)+[True],terminated=[False]*(count-1)+[True],
        old_log_probs=[choice['old_log_prob'] for _,choice in choices],
        action_masks=np.asarray([choice['action_mask'] for _,choice in choices]))


def test_sampling_execution_density_entropy_are_preview_only():
    agent=small_agent();state=np.zeros(26,dtype='float32')
    state_tensor=torch.as_tensor(state[None])
    with torch.no_grad():mu,std=agent.actor(state_tensor)
    samples=[agent.take_action(state,return_info=True) for _ in range(100)]
    assert np.std([action[2] for action,_ in samples])>.05
    for action,info in samples:
        assert action[0]==0 and action[1]==float(mu[0,1])
        np.testing.assert_array_equal(info['action_mask'],[0.,0.,1.])
        expected=torch.distributions.Normal(mu[0,2],std[0,2]).log_prob(torch.tensor(action[2]))
        assert info['old_log_prob']==float(expected)
        executed=study.full.local.positive.execute_action(action)
        assert executed[0]==0
        assert executed[1]==pytest.approx(float(torch.sigmoid(mu[0,1])),abs=2e-8)
    entropy=agent.masked_entropy(mu,std,torch.tensor([[0.,0.,1.]]))
    assert float(entropy)==float(torch.distributions.Normal(mu[0,2],std[0,2]).entropy())
    # An exact single-variate RNG draw proves feed/angular were not sampled.
    torch.manual_seed(71);action=agent.take_action(state)
    torch.manual_seed(71);expected=torch.distributions.Normal(mu[0,2],std[0,2]).sample()
    assert action[2]==float(expected)
    deterministic=agent.take_action(state,deterministic=True)
    assert deterministic[1]==float(mu[0,1]) and deterministic[2]==float(mu[0,2])


def test_multi_update_frozen_bytes_unchanged_but_preview_and_critic_learn():
    agent=small_agent();signature=frozen_actor_digest(agent.actor)
    before=copy.deepcopy(agent.actor.state_dict());critic=copy.deepcopy(agent.critic.state_dict())
    for _ in range(3):
        agent.update(rollout(agent))
        assert frozen_actor_digest(agent.actor)==signature
        agent.assert_frozen_parameters()
        diagnostics=agent.last_update_diagnostics
        assert diagnostics['decision_count_by_action']==[0,0,64]
        assert diagnostics['active_action_mask']==[0.,0.,1.]
        assert diagnostics['physical_decision_count']==0
        assert diagnostics['old_log_probability_max_absolute_error']<1e-4
        for epoch in diagnostics['epochs']:
            assert epoch['trunk_gradient_norm']==0
            assert epoch['old_new_kl_physics_mean']==0
            for name in ('feed','angular'):
                assert epoch[f'{name}_mu_head_gradient_norm']==0
                assert epoch[f'{name}_std_head_gradient_norm']==0
    for name,value in agent.actor.state_dict().items():
        if name.startswith(('mu_head.','std_head.')):
            assert value[:2].numpy().tobytes()==before[name][:2].numpy().tobytes()
            assert value[2].numpy().tobytes()!=before[name][2].numpy().tobytes()
        else:assert value.numpy().tobytes()==before[name].numpy().tobytes()
    assert any(not torch.equal(value,critic[name]) for name,value in agent.critic.state_dict().items())
    assert len(agent.actor_optimizer.state_dict()['state'])==4
    for value in agent.actor_optimizer.state_dict()['state'].values():
        assert torch.count_nonzero(value['exp_avg'][:2])==0
        assert torch.count_nonzero(value['exp_avg_sq'][:2])==0


@pytest.mark.parametrize('kind',['feed','angular','mask','logp'])
def test_wrong_rollout_sampling_or_density_is_rejected_before_mutation(kind):
    agent=small_agent();data=rollout(agent)
    before=copy.deepcopy(agent.actor.state_dict())
    if kind=='feed':data['actions'][0,1]+=.01
    elif kind=='angular':data['actions'][0,0]=.01
    elif kind=='mask':data['action_masks'][:,1]=1
    else:data['old_log_probs'][0]+=.01
    with pytest.raises(ValueError):agent.update(data)
    assert_tree_equal(before,agent.actor.state_dict())
    assert agent.actor_optimizer.state_dict()['state']=={}


def test_frozen_digest_detects_bitwise_corruption_and_never_repairs_it():
    agent=small_agent();data=rollout(agent)
    with torch.no_grad():agent.actor.mu_head.weight[0,0]=-0.
    assert torch.signbit(agent.actor.mu_head.weight[0,0])
    with pytest.raises(RuntimeError,match='bytes changed'):agent.update(data)
    assert torch.signbit(agent.actor.mu_head.weight[0,0])


def test_checkpoint_load_rejects_changed_feed_before_resume_or_evaluation():
    agent=small_agent();saved=copy.deepcopy(agent.actor.state_dict())
    saved['mu_head.weight'][2,0]+=.1
    agent.actor.load_state_dict(saved)  # Learned preview rows are compatible.
    saved['mu_head.bias'][1]+=.1
    with pytest.raises(RuntimeError,match='bytes changed'):agent.actor.load_state_dict(saved)
    with pytest.raises(RuntimeError,match='sealed once'):agent.seal_frozen_parameters()


def test_empty_or_malformed_rollout_is_rejected():
    agent=small_agent()
    with pytest.raises(ValueError,match='empty'):agent.update(dict(states=[]))
    for key in ('states','actions'):
        data=rollout(agent);data[key]=np.zeros((64,4))
        with pytest.raises(ValueError,match='shape'):agent.update(data)


def test_configuration_changes_only_training_scope_and_same_value_initializer():
    before=study.prior_study.configuration('preview_time','adaptive',101)
    after=study.configuration('preview_time','adaptive',101)
    allowed={'study','init_checkpoint','init_checkpoint_sha256','active_action_dims',
             'candidate_change','initialization','likelihood','frozen_feed_scope'}
    for key in before:
        if key!='revision':assert before[key]==after[key]
    for key in before['revision'].keys()|after['revision'].keys():
        if key not in allowed:assert before['revision'][key]==after['revision'][key]
    assert after['revision']['active_action_dims']==[2]
    assert after['revision']['adaptive_preview'] is True
    assert after['revision']['validation_milestones']==[0,256,512,1024]
    env=type('Env',(),{'observation_dim':26})()
    agent=study.make_agent(after,env)
    initial=torch.load(study.INITIAL/'checkpoint.pt',map_location='cpu',weights_only=False)
    fixed=torch.load(study.FIXED/'checkpoint.pt',map_location='cpu',weights_only=False)
    for name in ('actor','critic'):
        assert_tree_equal(getattr(agent,name).state_dict(),initial[name])
        assert study._model_state_digest(initial[name])==study._model_state_digest(fixed[name])
    assert agent.actor_optimizer.state_dict()['state']=={}
    assert agent.critic_optimizer.state_dict()['state']=={}
    assert all(parameter.requires_grad==name.startswith(('mu_head.','std_head.'))
               for name,parameter in agent.actor.named_parameters())


def test_loaded_initializer_exact_restore_across_partial64_batch(tmp_path):
    cfg=study.configuration('preview_time','adaptive',901)
    cfg['revision']['training_path_ids']=['three','five']
    first=study.BASE_TRAINER(cfg,tmp_path/'complete',ShortEpisodes,study.make_agent,'only-D')
    initial_digest=frozen_actor_digest(first.agent.actor)
    first.advance(128)
    expected_rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
    partial=study.BASE_TRAINER(cfg,tmp_path/'resume',ShortEpisodes,study.make_agent,'only-D')
    partial.advance(17)
    resumed=study.BASE_TRAINER(cfg,tmp_path/'resume',ShortEpisodes,study.make_agent,'only-D')
    resumed.advance(67)
    resumed=study.BASE_TRAINER(cfg,tmp_path/'resume',ShortEpisodes,study.make_agent,'only-D')
    resumed.advance(128)
    for name in ('actor','critic','actor_optimizer','critic_optimizer'):
        assert_tree_equal(getattr(first.agent,name).state_dict(),getattr(resumed.agent,name).state_dict())
    assert_tree_equal(first.updates,resumed.updates)
    assert_tree_equal(first.episodes,resumed.episodes)
    assert_tree_equal(first.buffer,resumed.buffer)
    assert_tree_equal(expected_rng,(torch.get_rng_state(),np.random.get_state(),random.getstate()))
    assert frozen_actor_digest(resumed.agent.actor)==initial_digest
    resumed.agent.assert_frozen_parameters()
    with pytest.raises(ValueError,match='source mismatch'):
        study.BASE_TRAINER(cfg,tmp_path/'resume',ShortEpisodes,study.make_agent,'changed')


def test_scoped_installer_restores_trainer_evaluator_and_exception_state(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path/'study')
    names=('OUT','configuration','source_state','prepare','FullBatchTrainer')
    before={name:getattr(study.full,name) for name in names}
    core_names=('OUT','configuration','source_state','prepare','MacroEnvironment','make_agent')
    core_before={name:getattr(study.full.core,name) for name in core_names}
    with pytest.raises(RuntimeError,match='deliberate'):
        with study.installed():
            assert study.full.FullBatchTrainer is study._trainer
            assert study.full.core.make_agent is study.make_agent
            assert study.full.core.MacroEnvironment is study.full.local.LocalCornerEnvironment
            with pytest.raises(RuntimeError,match='nest'):
                with study.installed():pass
            raise RuntimeError('deliberate')
    assert all(getattr(study.full,name) is value for name,value in before.items())
    assert all(getattr(study.full.core,name) is value for name,value in core_before.items())


@pytest.mark.parametrize('kind',['kinematics','brake'])
def test_foreign_kinematics_or_v3_brake_is_rejected(kind,monkeypatch):
    if kind=='kinematics':
        monkeypatch.setattr(study.prior_study.cnc_env,'apply_kinematic_constraints',lambda *args:None)
    else:monkeypatch.setattr(study.full.core,'plan_terminal_brake',lambda *args:None)
    with pytest.raises(RuntimeError,match='clean process'):
        with study.installed():pass


@pytest.mark.parametrize('decisions,stage',[(2048,'development'),(4096,'development'),(16,'development'),(17,'smoke'),(0,'formal')])
def test_train_and_evaluate_reject_outside_registered_budget(decisions,stage):
    with pytest.raises(ValueError):study.train(decisions,stage)
    with pytest.raises(ValueError):study.evaluate(decisions,stage)


def test_changed_parent_source_is_rejected_before_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path/'study')
    monkeypatch.setattr(study.prior_study,'BASE_SOURCE',lambda:{'source_sha256':'changed'})
    with pytest.raises(ValueError,match='full_batch source changed'):study.prepare()
    assert not (study.OUT/'protocol.json').exists()


def test_protocol_registers_two_distinct_same_weight_controls(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path/'study')
    protocol=study.prepare()
    assert protocol==study.prepare()
    assert protocol['initial_checkpoint_sha256']==study.old.sha(study.INITIAL/'checkpoint.pt')
    assert protocol['exact_3mm_control']['checkpoint_sha256']==study.old.sha(study.FIXED/'checkpoint.pt')
    assert protocol['additional_budget']==1024 and protocol['test_access'] is False
    assert protocol['validation_milestones']==[0,256,512,1024]
    assert protocol['initial_model_state_digests'].keys()=={'actor','critic'}


def test_reused_control_trace_and_path_identity_are_checked(tmp_path,monkeypatch):
    fixed=tmp_path/'fixed';(fixed/'validation').mkdir(parents=True)
    monkeypatch.setattr(study,'FIXED',fixed)
    monkeypatch.setattr(study.prior_study,'checked_snapshot',lambda snapshot:None)
    for name in ('checkpoint.pt','config.yaml'):(fixed/name).write_bytes(b'unit-content')
    study.old.write(fixed/'provenance.json',dict(source_sha256='unit-source'))
    rows=[]
    for path_id in ('a','b','c'):
        trace=fixed/'validation'/f'{path_id}.npz';trace.write_bytes(path_id.encode())
        rows.append(dict(path_id=path_id,checkpoint_sha256=study.old.sha(fixed/'checkpoint.pt'),
            config_sha256=study.old.sha(fixed/'config.yaml'),source_sha256='unit-source',
            trace_sha256=study.old.sha(trace)))
    summary=fixed/'validation/summary.json';study.old.write(summary,dict(rows=rows))
    assert study.checked_fixed_rows(['a','b','c'])==rows
    (fixed/'validation/a.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='binding changed'):study.checked_fixed_rows(['a','b','c'])
    study.old.write(summary,dict(rows=[rows[0],rows[0],rows[2]]))
    with pytest.raises(ValueError,match='all3'):study.checked_fixed_rows(['a','b','c'])
