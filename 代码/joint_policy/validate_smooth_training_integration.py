"""Check PPO-update, noise expiration and compact recovery together."""
import copy
import numpy as np
import torch
from study_io import BASE, register, receipt, write_json
from smooth_policy_candidate import SmoothExplorationPolicy
from learning_update_v4 import optimizers, update
from resume_codec_v5 import save_resume, load_resume
from profile_snapshot_codec import equivalent
from train_pilot import BATCH_KEYS


def main():
    stage = register('smooth_training_integration_validation_v1',
        [__file__,BASE/'train_smooth_learning_probe.py',BASE/'smooth_policy_candidate.py',
         BASE/'learning_update_v4.py',BASE/'resume_codec_v5.py',BASE/'joint_policy.py',
         BASE/'profile_snapshot_codec.py',BASE/'smooth_learning_probe_plan_v1.md'],
        dict(modes=2,synthetic_ppo_updates=2,rows_per_update=24,epochs=2,
             compact_saves=2,compact_loads=2,physical_ticks=0,real_training_decisions=0),
        'Check only the new integration across a PPO update and noise reset; reuse previous sampler and codec tests.')
    torch.set_num_threads(1)
    torch.manual_seed(915388)
    results=[]
    for mode in ['joint','fixed']:
        policy=SmoothExplorationPolicy(72,mode=mode)
        with torch.no_grad():
            policy.log_std.copy_(torch.log(torch.tensor([.03,.10,.20])))
        optimizer=optimizers(policy)
        batch={key:[] for key in BATCH_KEYS}
        observations=torch.randn(33,72)*.1
        before=copy.deepcopy(policy.state_dict())
        for i in range(24):
            action,logp,value=policy.act(observations[i])
            terminal=i==23
            next_value=0. if terminal else float(policy.value(observations[i+1]).detach())
            for key,item in zip(BATCH_KEYS,[observations[i].numpy(),action,logp,value,
                float(.03*np.cos(i)),next_value,terminal]):
                batch[key].append(item)
        statistics=update(policy,optimizer,batch,epochs=2,minibatch=8)
        assert statistics['actor_minibatches']>0
        assert not torch.equal(before['actor.4.bias'][:2],policy.actor[-1].bias[:2])
        if mode=='fixed':
            assert torch.equal(before['actor.4.weight'][2],policy.actor[-1].weight[2])
            assert torch.equal(before['actor.4.bias'][2],policy.actor[-1].bias[2])
            assert torch.equal(before['log_std'][2],policy.log_std[2])
        policy.exploration_age.fill_(policy.noise_interval)
        state=dict(policy=policy.state_dict(),optimizer={k:v.state_dict() for k,v in optimizer.items()},
            batch={key:[] for key in BATCH_KEYS},episode_rows=[],torch_rng=torch.get_rng_state(),
            phase='training',decision=24,sampler='gsde8',noise_interval=8)
        save_resume(stage/f'{mode}.pt',state)
        expected=[policy.act(observations[24+i])[0] for i in range(9)]
        restored=load_resume(stage/f'{mode}.pt')
        # State_dict entries can alias live tensors, so compare optimizer after restoring its saved state.
        policy.load_state_dict(restored['policy'])
        for key,opt in optimizer.items():
            opt.load_state_dict(restored['optimizer'][key])
            assert equivalent(opt.state_dict(),restored['optimizer'][key])
        torch.set_rng_state(restored['torch_rng'])
        actual=[policy.act(observations[24+i])[0] for i in range(9)]
        assert np.array_equal(np.array(actual),np.array(expected))
        assert int(policy.exploration_age)==1
        results.append(dict(mode=mode,ppo=statistics,post_update_noise_expiration=True,
            compact_recovery_across_resampling_bitwise=True,optimizer_roundtrip=True,
            inactive_D_preserved=mode=='fixed',sampling_actor_calls=42))
    write_json(stage/'results.json',dict(results=results,physics_ticks=0,real_training_decisions=0))
    receipt(stage,dict(status='new_PPO_noise_reset_compact_recovery_integration_passed',
        modes=2,synthetic_ppo_updates=2,synthetic_rows=48,sampling_actor_calls=84,
        compact_saves=2,compact_loads=2,physics_ticks=0,real_training_decisions=0))
    print('Both integration checks passed; no physical trajectory executed')


if __name__=='__main__':
    main()
