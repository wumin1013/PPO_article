"""Small offline fitter shared by feedback-migration experiments.

Targets are conditional teacher functions during training only. A normal
fourteen-input actor is exported after absorbing feature standardization.
"""
import copy
import time
import numpy as np
import torch
from evaluate_historical_dense import write, sha


def fit(policy, groups, label_function, stage, checks, *, seed, deadline,
        initial_path, checkpoint_steps=(8000,16000,30000), screen=None):
    rng=np.random.default_rng(seed)
    weights=torch.cat([torch.full((len(g['x']),),g['fraction']/len(g['x'])) for g in groups])
    x=torch.cat([g['x'] for g in groups])
    assert abs(float(weights.sum())-1.)<1e-6
    center=(x*weights[:,None]).sum(0)
    scale=torch.sqrt(((x-center).square()*weights[:,None]).sum(0)).clamp_min(.03)
    actor=copy.deepcopy(policy.actor)
    with torch.no_grad():
        raw=actor[0].weight.clone()
        actor[0].weight.mul_(scale)
        actor[0].bias.add_(raw@center)
        assert float(abs(actor((x-center)/scale)-policy.actor(x)).max())<3e-5
    optimizer=torch.optim.Adam(actor.parameters(),lr=2e-4)
    action_scale=torch.tensor([.015,.15,.3])
    noise=torch.tensor([.006,.003,.003,.008,.008,.006,.006,.003,.003,.003,.003,.003,.003,.006])
    log=[]
    evaluations=[]
    started=time.perf_counter()
    for iteration in range(1,max(checkpoint_steps)+1):
        if time.perf_counter()>deadline-240.:
            break
        pieces=[]
        targets=[]
        for group in groups:
            ids=torch.from_numpy(rng.integers(0,len(group['x']),size=group['batch_size']))
            value=group['x'][ids].clone()
            if iteration%2==0:
                value+=torch.randn_like(value)*noise
                value[:,6].clamp_(-.995,.995)
            with torch.no_grad():
                y=label_function(value,group,ids)
            pieces.append(value)
            targets.append(y)
        inputs=torch.cat(pieces)
        target=torch.cat(targets)
        prediction=actor((inputs-center)/scale)
        # Equivalent upper/lower clipped actions do not need to reproduce an
        # arbitrary latent mean. A .02 margin keeps the desired boundary action.
        error=prediction-target
        error=torch.where(target>=1.,-torch.relu(1.02-prediction),error)
        error=torch.where(target<=-1.,torch.relu(prediction+1.02),error)
        loss=(error/action_scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step()
        if iteration in checkpoint_steps[:-1]:
            for group in optimizer.param_groups:
                group['lr']*=.3
        if iteration%500==0:
            log.append(dict(iteration=iteration,loss=float(loss.detach()),wall_s=time.perf_counter()-started))
            write(stage/'fit_log.json',log)
            print('FIT',iteration,'loss',log[-1]['loss'],'wall',log[-1]['wall_s'],flush=True)
        if iteration not in checkpoint_steps:
            continue
        exported=copy.deepcopy(actor)
        with torch.no_grad():
            exported[0].weight.div_(scale)
            exported[0].bias.sub_(exported[0].weight@center)
            export_error=float(abs(exported(x)-actor((x-center)/scale)).max())
        assert export_error<3e-5
        policy.actor.load_state_dict(exported.state_dict())
        checkpoint=stage/f'fit_{iteration:05d}.pt'
        torch.save(dict(policy=policy.state_dict(),fit_optimizer=optimizer.state_dict(),
            normalized_actor=actor.state_dict(),center=center,scale=scale,offline_iterations=iteration,
            interface=[14,3,seed],feed_gain=1.,policy_class='FineCompactPolicy',runtime_class='FarVectorRuntime',
            observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,
            new_ppo_updates=0,inherited_ppo_updates=88,source_checkpoint=str(initial_path),
            source_checkpoint_sha256=sha(initial_path),
            observation_features=checks.runtime_class(checks.references['dense']).feature_names(),
            training_kind='Offline supervised local feedback fit with clipped-action-equivalence loss',
            export_equivalence_error=export_error),checkpoint)
        record=checks.run(policy,checkpoint,iteration)
        results=record['results']
        eligible=all(r['success'] for r in results.values())
        if eligible:
            c=results['original']['corner_comparisons']
            eligible=bool(results['original']['complete_time_s']<=444.296+1e-9 and
                results['dense']['complete_time_s']<=15.06985+1e-9 and
                c['historical_main']['not_worse']>=23 and c['S2']['not_worse']>=28 and
                c['historical_strong_fixed']['not_worse']>=27 and
                c['historical_main']['total_change_mm']<=-1.18794470955+1e-7 and
                (c['historical_main']['not_worse']>=25 or results['original']['complete_time_s']<444.276-1e-9))
        if screen is not None:
            eligible=bool(screen(results))
        record['eligible_for_refinement']=eligible
        evaluations.append(record)
        write(stage/'evaluations.json',evaluations)
    return evaluations,iteration
