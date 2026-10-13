"""Audit conditional mean-shift credit on two closed behavior batches, without updating."""
import json
from pathlib import Path
import numpy as np
import torch
from conditional_exploration import ConditionalExplorationPolicy, RHO
from joint_policy import advantages
from resume_codec_v5 import load_resume
from study_io import BASE, register, receipt, write_json, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    names=[f'ar_conditional_{m}_20260916' for m in ('joint','fixed')]
    sources=[Path(__file__),BASE/'conditional_exploration.py',BASE/'joint_policy.py',
             BASE/'resume_codec_v5.py',BASE/'conditional_exploration_design.md',
             BASE/'conditional_exploration_results_20260916/receipt.json']
    external=[]
    for name in names:
        sources += [BASE/name/'receipt.json',BASE/name/'storage.json']
        folder=Path(read(BASE/name/'storage.json')['rolling_state_directory'])
        assert read(folder/'resume_pointer.json')['training_complete']
        path=folder/'resume_1.pt'
        external.append(dict(stage=name,path=str(path),sha256=sha(path)))
    stage=register('conditional_credit_geometry_20260916',sources,
        dict(buffer_loads=2,model_constructions=2,actor_batch_forwards=4,
             autograd_calls=2,maximum_rows=4096,training_updates=0,physics_ticks=0),
        'Test exact history-conditional constant-output-bias credit and compare GAE with completed-suffix returns; descriptive frozen-batch diagnosis, no optimizer or physical replay.')
    write_json(stage/'external_sources.json',external)
    torch.set_num_threads(1)
    results=[]
    for binding in external:
        path=Path(binding['path']);state=load_resume(path)
        assert state['decision']==59999 and state['sampler']=='ar_conditional8'
        batch=state['batch'];n=len(batch['reward']);assert 0<n<=2048
        policy=ConditionalExplorationPolicy(state['observation_dim'],mode=state['mode'])
        policy.load_state_dict(state['policy'])
        obs=torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
        oldobs=torch.as_tensor(np.asarray(batch['previous_observation']),dtype=torch.float32)
        oldz=torch.as_tensor(np.asarray(batch['previous_latent']),dtype=torch.float32)
        live=torch.as_tensor(batch['context_live'],dtype=torch.bool)
        z=torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
        dist=policy.conditional_distribution(obs,oldobs,oldz,live)
        logp=(dist.log_prob(z)*policy.active_mask).sum(-1)
        log_error=float((logp-torch.tensor(batch['log_probability'])).abs().max().detach())
        assert log_error<1e-3
        gae,returns=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'])
        mc,mc_returns=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],lam=1.)
        scale=float(np.std(gae));normalized=(gae-gae.mean())/(scale+1e-8)
        # Same additive actor-output bias affects the current and previous means.
        factor=1-RHO*live.to(torch.float32)
        score=((z-dist.mean)/dist.stddev.square()*factor[:,None]*policy.active_mask).detach().numpy()
        loss=-(logp*torch.as_tensor(normalized)).mean()
        actual=-torch.autograd.grad(loss,policy.actor[-1].bias)[0].detach().numpy()
        analytic=(score*normalized[:,None]).mean(0)
        assert np.max(np.abs(actual-analytic))<2e-5
        labels=np.full(n,'censored',dtype='U10')
        # Mark only suffixes ending in a terminal within the captured batch.
        low=state['decision']-n+1;end=0
        for ep in state['episodes']:
            start=end+1;end+=ep['decisions']
            left,right=max(start,low),min(end,state['decision'])
            if left<=right:
                labels[left-low:right-low+1]='success' if ep['success'] else 'failure'
        tangents=obs[:,42:62].numpy().reshape(n,10,2)
        straight=np.max(np.linalg.norm(tangents-tangents[:,:1],axis=2),axis=1)<1e-5
        groups={**{k:labels==k for k in ('success','failure','censored')},
                'straight_30mm':straight,'turn_within_30mm':~straight}
        parts=[]
        # Retain the original GAE normalization for both scores to isolate credit differences.
        centered_mc=(mc-gae.mean())/(scale+1e-8)
        for key,mask in groups.items():
            if not mask.any():continue
            parts.append(dict(group=key,rows=int(mask.sum()),
                gae_bias_ascent_full_batch=(score[mask]*normalized[mask,None]).sum(0).tolist(),
                mc_bias_ascent_full_batch=(score[mask]*centered_mc[mask,None]).sum(0).tolist(),
                mean_mc_minus_gae=float((mc[mask]-gae[mask]).mean()),
                mean_return_minus_value=float(mc[mask].mean()),
                mean_absolute_return_minus_value=float(np.abs(mc[mask]).mean())))
            for k in ('gae_bias_ascent_full_batch','mc_bias_ascent_full_batch'):
                parts[-1][k]=[v/n for v in parts[-1][k]]
        results.append(dict(stage=binding['stage'],rows=n,likelihood_max_error=log_error,
            bias_autograd_error=float(np.max(np.abs(actual-analytic))),gae_std=scale,
            gae_bias_ascent=actual.tolist(),
            mc_bias_ascent_using_gae_normalization=(score*centered_mc[:,None]).mean(0).tolist(),
            live_rows=int(live.sum()),parts=parts))
        assert sha(path)==binding['sha256']
    result=dict(groups=results,rho=RHO,
        continuing_step_constant_bias_fisher_ratio=(1-RHO)/(1+RHO),
        interpretation='Exact ratio to independent Gaussian at equal marginal std for a constant latent mean shift, conditional on fixed history. It is neither whole-network Fisher nor a measured learning-rate or speed gain.',
        credit_boundary='MC on terminated suffixes uses observed remaining return; batch-tail censored rows bootstrap. This does not identify an optimal action or prove the critic needs extra state. State-conditional groups overlap outcome groups; never sum them together.',
        costs=dict(buffer_loads=2,model_constructions=2,actor_batch_forwards=4,autograd_calls=2,
                   rows=sum(g['rows'] for g in results),training_updates=0,physics_ticks=0))
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_frozen_conditional_credit_diagnostic',**result['costs']))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
