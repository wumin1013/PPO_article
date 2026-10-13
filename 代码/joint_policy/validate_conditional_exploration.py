"""Finite likelihood, gradient, active-action and context-codec checks."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from conditional_exploration import ConditionalExplorationPolicy,RHO
from conditional_update import update
from joint_policy import JointPolicy
from learning_update_v4 import optimizers
from resume_codec_v5 import save_resume,load_resume
from study_io import BASE,register,receipt,write_json


def main():
    stage=register('conditional_exploration_validation_20260916',
        [Path(__file__),BASE/'conditional_exploration.py',BASE/'conditional_update.py',
         BASE/'conditional_exploration_design.md',BASE/'train_conditional_exploration.py',
         BASE/'joint_policy.py',BASE/'learning_update_v4.py',BASE/'resume_codec_v5.py'],
        dict(synthetic_sampling_steps=128,synthetic_ppo_updates=2,codec_save=1,codec_load=1,
             physics_ticks=0,real_training_decisions=0),
        'Check exact conditional likelihood and gradient, original deterministic execution, inactive head and recovery context before physical training.')
    torch.set_num_threads(1);torch.manual_seed(916)
    checks=[];costs=[]
    for mode in ('joint','fixed'):
        policy=ConditionalExplorationPolicy(72,mode=mode)
        reference=JointPolicy(72,mode=mode);reference.load_state_dict(policy.state_dict())
        observation=torch.randn(64,72)*.2
        with torch.no_grad():assert torch.equal(policy.actor(observation),reference.actor(observation))
        first=policy.conditional_distribution(observation,observation,torch.zeros(64,3),torch.zeros(64,dtype=torch.bool))
        ordinary=policy.distribution(observation)
        assert torch.equal(first.mean,ordinary.mean) and torch.equal(first.stddev,ordinary.stddev)
        keys=['observation','latent','log_probability','value','reward','next_value','terminal',
              'previous_observation','previous_latent','context_live']
        batch={k:[] for k in keys}
        for i,obs in enumerate(observation):
            if i%16==0:policy.reset_context()
            z,logp,value=policy.act(obs.numpy())
            done=(i+1)%16==0
            for k,v in zip(keys,[obs.numpy(),z,logp,value,float(z[1]),0.,done]):batch[k].append(v)
            for k,v in policy.last_context.items():batch[k].append(v)
        context=torch.tensor(batch['context_live'])
        args=[observation,torch.tensor(np.array(batch['latent'])),torch.tensor(np.array(batch['previous_observation'])),
              torch.tensor(np.array(batch['previous_latent'])),context]
        logp,entropy,value=policy.evaluate_conditioned(*args)
        error=float((logp-torch.tensor(batch['log_probability'])).abs().max().detach())
        assert error<1e-4
        distribution=policy.conditional_distribution(args[0],args[2],args[3],context)
        grad=torch.autograd.grad(logp.sum(),policy.actor[-1].bias,retain_graph=True)[0]
        score=(args[1]-distribution.mean)/distribution.stddev.square()
        expected=(score*(1-RHO*context.float()).unsqueeze(-1)*policy.active_mask).sum(0)
        assert torch.allclose(grad,expected,atol=2e-3,rtol=1e-4)
        if mode=='joint':
            snapshot=dict(batch=batch,episode_rows=[],exploration_context=policy.context_state())
            save_resume(stage/'context_roundtrip.pt',snapshot)
            restored=load_resume(stage/'context_roundtrip.pt')
            for k in batch:assert np.array_equal(np.asarray(batch[k]),np.asarray(restored['batch'][k]))
            for k,v in policy.context_state().items():assert np.array_equal(v,restored['exploration_context'][k])
        old=copy.deepcopy(policy.state_dict());opt=optimizers(policy)
        for g in opt['actor'].param_groups:g['lr']=3e-5
        stats=update(policy,opt,batch,epochs=1,minibatch=32)
        if mode=='fixed':
            assert torch.equal(old['actor.4.weight'][2],policy.actor[-1].weight[2])
            assert torch.equal(old['actor.4.bias'][2],policy.actor[-1].bias[2])
            assert torch.equal(old['log_std'][2],policy.log_std[2])
        assert stats['actor_minibatches']>0 and np.isfinite(list(stats.values())).all()
        checks.append(dict(mode=mode,initial_distribution_equal=True,deterministic_actor_equal=True,
            conditional_likelihood_max_error=error,history_mean_gradient_pass=True,
            inactive_head_unchanged=mode=='fixed',context_codec_pass=mode=='joint'))
        costs.append(stats)
    write_json(stage/'results.json',dict(checks=checks,synthetic_update_statistics=costs,
        theoretical_stationary_variance_identity=(1-RHO**2)/(1-RHO**2),rho=RHO,
        scope='Synthetic policy/optimizer and codec checks only. No physical efficacy or stationarity of state-conditioned marginal actions is asserted.'))
    receipt(stage,dict(status='passed_conditional_exploration_contract',synthetic_sampling_steps=128,
        synthetic_ppo_updates=2,codec_save=1,codec_load=1,physics_ticks=0,real_training_decisions=0))
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':main()
