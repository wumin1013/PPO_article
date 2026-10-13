"""Finite optimizer isolation checks and a shared straight-path curriculum."""
import copy
import json
import numpy as np
import torch
from joint_policy import JointPolicy
from learning_update_v4 import optimizers,update
from mixed_curriculum import _check
from study_io import BASE,register,receipt,write_json


if __name__=='__main__':
    stage=register('learning_preparation_v4',[__file__,BASE/'learning_design_v4.md',
        BASE/'learning_update_v4.py',BASE/'joint_policy.py',BASE/'mixed_curriculum.py',
        BASE/'mixed_curriculum_preparation_v2/paths.json'],
        dict(synthetic_optimizer_updates=4,synthetic_samples=64,new_straight_paths=16,
             environments=0,physics_ticks=0),
        'Validate actor/critic isolation and KL stopping; freeze shared curriculum additions.')
    torch.set_num_threads(1)
    torch.manual_seed(915104)
    policy=JointPolicy(72,mode='fixed')
    obs=torch.randn(64,72)*.2
    with torch.no_grad():
        latent=policy.distribution(obs).sample()
        logp=policy.evaluate(obs,latent)[0]
    batch=dict(observation=obs.numpy(),latent=latent.numpy(),log_probability=logp.numpy(),
        reward=(latent[:,1]-.1).numpy(),value=np.zeros(64),next_value=np.zeros(64),
        terminal=np.array([(i+1)%16==0 for i in range(64)]))
    twin=copy.deepcopy(policy)
    with torch.no_grad():
        twin.critic[-1].bias.add_(1000.)
    initial=copy.deepcopy(policy.state_dict())
    torch.manual_seed(44)
    a=update(policy,optimizers(policy),batch,epochs=2,minibatch=32)
    torch.manual_seed(44)
    b=update(twin,optimizers(twin),batch,epochs=2,minibatch=32)
    assert all(torch.equal(x,y) for x,y in zip(policy.actor.parameters(),twin.actor.parameters()))
    assert torch.equal(policy.actor[-1].weight[2],initial['actor.4.weight'][2])
    assert torch.equal(policy.actor[-1].bias[2],initial['actor.4.bias'][2])
    assert not torch.equal(policy.actor[-1].bias[:2],initial['actor.4.bias'][:2])
    rejected=copy.deepcopy(policy)
    before=copy.deepcopy(rejected.state_dict())
    mismatched=dict(batch,log_probability=batch['log_probability']-3.)
    stopped=update(rejected,optimizers(rejected),mismatched,epochs=2,minibatch=32)
    assert stopped['actor_minibatches']==0 and stopped['actor_early_stop']
    assert all(torch.equal(rejected.state_dict()[key],value) for key,value in before.items()
               if key.startswith('actor.') or key=='log_std')
    assert any(not torch.equal(rejected.state_dict()[key],value) for key,value in before.items()
               if key.startswith('critic.'))
    joint=JointPolicy(72)
    with torch.no_grad():
        z=joint.distribution(obs).sample()
        old=joint.evaluate(obs,z)[0]
    joint_before=joint.actor[-1].bias.detach().clone()
    joint_batch=dict(batch,latent=z.numpy(),log_probability=old.numpy(),
        reward=(z[:,0]+2*z[:,1]+3*z[:,2]).numpy())
    full=update(joint,optimizers(joint),joint_batch,epochs=2,minibatch=32)
    assert torch.all((joint.actor[-1].bias-joint_before).abs()>1e-8)
    assert all(np.isfinite([r[k] for k in ['actor_loss','value_loss','approximate_kl']]).all()
               for r in [a,b,stopped,full])
    data=json.loads((BASE/'mixed_curriculum_preparation_v2/paths.json').read_text(encoding='utf-8'))
    rng=np.random.default_rng(915104)
    for angle in np.linspace(-np.pi,np.pi,16,endpoint=False):
        length=float(rng.uniform(30.,150.))
        reference=np.array([[0.,0.],[length*np.cos(angle),length*np.sin(angle)]])
        assert _check(reference)
        data['training'].append(reference.tolist())
    data['warmup_indices']=[0,1,2,3]+list(range(48,64))
    data['split']='Development only; original validation unchanged; new straight paths are training only.'
    write_json(stage/'paths.json',data)
    write_json(stage/'checks.json',dict(actor_independent_of_critic_gradient=True,
        inactive_head_parameters_unchanged=True,joint_three_heads_updated=True,
        kl_stop_preserves_actor_and_updates_critic=True,new_paths_valid=True,
        statistics=[a,b,stopped,full]))
    receipt(stage,dict(status='optimizer_checks_and_curriculum_preparation_completed',
        synthetic_optimizer_updates=4,synthetic_samples=64,new_straight_paths=16,
        training_paths=64,validation_paths=12,environments=0,physics_ticks=0))
    print('Optimizer isolation, active heads, KL stop and 16 new straight paths checked.')
