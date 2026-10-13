"""Outcome and geometry conditioned feed response on frozen behavior batches."""
import copy
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from joint_policy import JointPolicy, advantages
from learning_update_v4 import optimizers, update
from resume_codec_v5 import load_resume
from study_io import BASE, register, receipt, write_json, sha


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def geometry(ref):
    d=np.diff(np.asarray(ref,float),axis=0)
    lengths=np.linalg.norm(d,axis=1)
    t=d/lengths[:,None]
    angles=np.arccos(np.clip(np.sum(t[:-1]*t[1:],axis=1),-1,1))*180/np.pi
    return dict(length=float(lengths.sum()),max_turn=float(angles.max(initial=0)),segments=len(d))


def main():
    manifest=BASE/'current_state_reconciliation_20260916/external_buffers.json'
    sources=[Path(__file__),manifest,BASE/'joint_policy.py',BASE/'learning_update_v4.py',
             BASE/'resume_codec_v5.py',BASE/'study_io.py',BASE/'learning_preparation_v4/paths.json',
             BASE/'saved_actor_gradient_decomposition_v1/results.json',
             BASE/'v7_cruise_request_diagnosis_v1/receipt.json']
    stage=register('feed_update_context_20260916',sources,
        dict(frozen_buffer_loads=5,actor_batch_forwards=8,offline_original_ppo_updates=3,
             autograd_calls_upper=200,wall_s=180,training_decisions=0,physics_ticks=0),
        'Quantify local feed-output response to success/failure and action gradients across path length and local geometry; run the unchanged optimizer only on three previously unanalyzed late/wide batches. No physical training or policy selection.')
    started=time.perf_counter()
    costs=dict(buffer_loads=0,actor_batch_forwards=0,autograd_calls=0,offline_updates=0,
               actor_minibatches=0,critic_minibatches=0,physics_ticks=0,training_decisions=0)
    try:
        torch.set_num_threads(1)
        paths=read(BASE/'learning_preparation_v4/paths.json')['training']
        metas=[geometry(p) for p in paths]
        median_length=float(np.median([m['length'] for m in metas]))
        groups=[]
        for binding in read(manifest):
            assert time.perf_counter()-started<180
            source=Path(binding['path'])
            assert sha(source)==binding['sha256']
            state=load_resume(source);costs['buffer_loads']+=1
            batch=state['batch'];n=len(batch['reward'])
            assert 0<n<=2048 and state['sampler']=='independent'
            policy=JointPolicy(state['observation_dim'],mode=state['mode'])
            policy.load_state_dict(state['policy'])
            params=list(policy.actor.parameters())+[policy.log_std]
            obs=torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
            latent=torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
            dist=policy.distribution(obs);costs['actor_batch_forwards']+=1
            headlog=dist.log_prob(latent)*policy.active_mask
            old=torch.as_tensor(batch['log_probability'],dtype=torch.float32)
            ratio=(headlog.sum(-1)-old).exp()
            error=float((headlog.sum(-1)-old).abs().max().detach())
            assert error<1e-3 and bool(((ratio>.8)&(ratio<1.2)).all())
            gae,_=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'])
            adv=torch.as_tensor(gae);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            losses=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)-.001*(dist.entropy()*policy.active_mask).sum(-1)
            feed=50*(1+dist.mean[:,1].tanh())
            labels=np.full(n,'censored',dtype='U12')
            path_id=np.full(n,int(state['path_index']),dtype=int)
            lower=state['decision']-n+1;end=0
            for ep in state['episodes']:
                start=end+1;end+=ep['decisions']
                first,last=max(start,lower),min(end,state['decision'])
                if first<=last:
                    labels[first-lower:last-lower+1]='success' if ep['success'] else 'failure'
                    path_id[first-lower:last-lower+1]=ep['path']
            assert end+state['environment']['decisions']==state['decision'], (end,state['decision'])
            lengths=np.array([metas[p]['length'] for p in path_id])
            turns=np.array([metas[p]['max_turn'] for p in path_id])
            # Existing 72-D sensor tangents, not new geometry or a control rule.
            tangents=obs[:,42:62].detach().numpy().reshape(n,10,2)
            straight=np.max(np.linalg.norm(tangents-tangents[:,0:1],axis=2),axis=1)<1e-5
            masks={'all':np.ones(n,bool),'success':labels=='success','failure':labels=='failure',
                   'censored':labels=='censored','short_path':lengths<=median_length,
                   'long_path':lengths>median_length,'max_turn_le90':turns<=90.00001,
                   'max_turn_gt90':turns>90.00001,'straight_30mm':straight,'turn_within_30mm':~straight}
            masks={k:v for k,v in masks.items() if v.any()}
            def gradient(loss):
                costs['autograd_calls']+=1
                gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                  for p,g in zip(params,gs)]).detach()
            total=gradient(losses.mean())
            outcomes={k:gradient((losses*torch.as_tensor(v)).mean())
                      for k,v in masks.items() if k in ('success','failure','censored')}
            heads={name:gradient(-(ratio.detach()*adv*headlog[:,i]).mean()
                                -.001*(dist.entropy()[:,i]*policy.active_mask[i]).mean())
                   for i,name in enumerate(('steering','feed','preview')) if policy.active_mask[i]}
            outcome_error=float((sum(outcomes.values())-total).abs().max())
            head_error=float((sum(heads.values())-total).abs().max())
            assert max(outcome_error,head_error)<2e-5
            probes=[]
            for label,mask in masks.items():
                g=gradient(feed[torch.as_tensor(mask)].mean())
                # -J_feed dot grad(loss): first derivative under unit plain gradient descent.
                response=lambda direction:float(-torch.dot(g,direction))
                probes.append(dict(group=label,rows=int(mask.sum()),paths=int(len(set(path_id[mask]))),
                    mean_requested_feed=float(feed[torch.as_tensor(mask)].mean().detach()),
                    feed_sgd_derivative=response(total),
                    by_outcome={k:response(v) for k,v in outcomes.items()},
                    by_action={k:response(v) for k,v in heads.items()}))
            score=(latent[:,1]-dist.mean[:,1])/dist.stddev[:,1].square()
            feed_bias_signal=(ratio.detach()*adv*score).detach().numpy()
            bias={k:dict(rows=int(v.sum()),full_batch_sum=float(feed_bias_signal[v].sum()/n))
                  for k,v in masks.items()}
            record=dict(stage=binding['stage'],source_sha256=binding['sha256'],decision=state['decision'],
                rows=n,likelihood_max_error=error,gae_std=float(np.std(gae)),
                sample_path_length_min=float(lengths.min()),sample_path_length_max=float(lengths.max()),
                full_batch_feed_bias_ascent=bias,feed_response_probes=probes,
                additive_error=dict(outcomes=outcome_error,heads=head_error))
            if binding['stage'].endswith('_refinement_v7') or binding['stage']=='wide_preview_independent_v1':
                before=feed.detach().clone()
                optimizer=optimizers(policy)
                for key,value in state['optimizer'].items():
                    optimizer[key].load_state_dict(copy.deepcopy(value))
                torch.set_rng_state(state['torch_rng'].clone())
                statistics=update(policy,optimizer,batch)
                costs['offline_updates']+=1
                costs['actor_minibatches']+=statistics['actor_minibatches']
                costs['critic_minibatches']+=statistics['critic_minibatches']
                with torch.no_grad():
                    after=50*(1+policy.distribution(obs).mean[:,1].tanh())
                costs['actor_batch_forwards']+=1
                delta=(after-before).numpy()
                record['unchanged_optimizer_offline_update']=dict(statistics=statistics,
                    feed_delta={k:float(delta[v].mean()) for k,v in masks.items()},
                    purpose='Actual unchanged shuffled PPO/clip-grad/Adam update on this saved batch; not a deployed model, not additive outcome attribution.')
            assert sha(source)==binding['sha256']
            groups.append(record)
            write_json(stage/'partial_results.json',dict(groups=groups,costs=costs))
            print(binding['stage'],n,{p['group']:round(p['feed_sgd_derivative'],5) for p in probes if p['group'] in ('all','success','failure','straight_30mm')},
                  record.get('unchanged_optimizer_offline_update',{}).get('feed_delta',{}),flush=True)
        assert costs['autograd_calls']<=200
        result=dict(groups=groups,costs=costs,training_geometry=metas,
            training_length_median_mm=median_length,wall_s=time.perf_counter()-started,
            scope='Five frozen independent-Gaussian batches. Gradient probes are before clipping and Adam and use original full-batch normalization. Their derivatives are parameterization-dependent local responses, not physical speed gains. One original optimizer update is measured separately on three new batches. Outcome conditioning is descriptive; sampled rows and paths are correlated. Training-length groups are relative to the training median, not 30-m deployment.',
            sealed_test_opened=False,production_models_modified=False,goal_complete=False)
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_feed_update_context_diagnosis',**costs,wall_s=result['wall_s']))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs,
            wall_s=time.perf_counter()-started,automatic_retry=False))
        raise


if __name__=='__main__':
    main()
