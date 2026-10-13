"""Learn local joint corrections in the nullspace of successful entry prefixes."""
import argparse
import copy
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.linalg import null_space
import learn_stratified_suffix_joint_credit as suffix
from deep_joint_credit_policy import credit_normalization
from local_suffix_credit_policy import LocalSuffixCreditPolicy, local_basis
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def prefix_indices(reference, rows, observations):
    u=reference[1]-reference[0];u=u/np.linalg.norm(u)
    heading=np.arctan2(u[0]*rows[:,3]-u[1]*rows[:,2],rows[:,2:4]@u)
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    changed=np.flatnonzero((abs(heading)>np.deg2rad(.05)) & (speed>5.) & (rows[:,12]==0))
    # Include the entire first significant-turn command. Straight paths keep all observations.
    count=min(len(observations),int(changed[0]//20+1)) if len(changed) else len(observations)
    return np.arange(count)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--numerical-prefix',action='store_true')
    args=parser.parse_args()
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_fast_absolute_shape_joint_credit_20260919'
    previous=BASE/'canonical_gate_band_stratified_suffix_joint_credit_20260919'
    local=BASE/'canonical_gate_band_curved_local_suffix_credit_20260919'
    data=read(source/'results.json');baseline=next(c for c in data['candidates'] if c['tag']==data['best'])
    parent=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    queries=sorted(read(previous/'query_results.json')['queries'],key=lambda q:q['query'])
    stage=register('canonical_gate_band_numerical_prefix_suffix_credit_20260919' if args.numerical_prefix else
                   'canonical_gate_band_prefix_preserving_suffix_credit_20260919',
        [__file__,BASE/'local_suffix_credit_policy.py',BASE/'deep_joint_credit_policy.py',
         BASE/'learn_stratified_suffix_joint_credit.py',local/'results.json',local/'receipt.json',
         local/'local_quadratic_0.125.pt',previous/'query_results.json',source/'research_candidate.pt',source/'results.json'],
        dict(wall_s=240,workers=4,optimizer_steps=250,new_candidates_maximum=4,paths=20,
             protected_heading_threshold_deg=.05,steering_feed_preview_scales=[.125,.5,1.],nullspace_relative_cutoff=1e-12),
        ('The preceding strict bitwise preflight stopped BEFORE any new physics: '
         'maximum action difference was4.74e-16 at near-zero float32 actions. '
         'Revise only this claimed numerical identity: require saved-state latent '
         'differences<=1e-12, explicitly report not bitwise identical, and rely on '
         'unchanged full physical/shape acceptance. No rollout is repeated. ' if args.numerical_prefix else '')+
        'The local quadratic update reduced physical reversal but introduced1.90e-6mm '
        'incoming excursion, exceeding the unchanged1e-6 primary screen. Preserve all '
        'original primary/hard limits. Construct a TRAINING parameter nullspace over '
        'each successful saved prefix through the first significant physical turn, and '
        'all states on paths without a significant turn. Optimize the SAME local joint '
        'quadratic-credit objective inside this nullspace. Use float64 only for the '
        'small residual algebra, then return float32 actions as before; verify '
        'protected saved-state action differences under the registered identity criterion. '
        'No prefix mask, phase ID, angle table or time switch is deployed: runtime '
        'only evaluates the same smooth state kernel and trained coefficients. '
        'This finite training-state equality is not a generalization or full-path '
        'safety certificate. Compare protected scales0.125/0.5/1 and unprotected0.0625 '
        'as a smaller-step control, reusing all prior larger controls and720 branches. '
        'Accept research updates only from complete actual primary-preserving improvement; '
        'all original shape/cruise64/fair/generalization requirements remain. If projection '
        'removes all useful directions, change representation rather than rerun unchanged.')
    started=time.perf_counter();outcomes=[];pool=None
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next','在训练参数空间保护成功入弯前缀，联合学习其余响应并复核完整轨迹。'],
        check=True,stdout=subprocess.DEVNULL)
    try:
        support,protected,counts=[],[],[]
        for r in baseline['records']:
            with np.load(source/r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
            indices=prefix_indices(ref,rows,obs);support.append(obs);protected.append(obs[indices]);counts.append(dict(path=r['path'],protected_decisions=len(indices)))
        support=np.vstack(support);protected=np.vstack(protected)
        qobs=np.array([q['observation'] for q in queries],np.float32)
        mean,std=credit_normalization(support);centers,radius=local_basis(qobs,mean,std)
        p=LocalSuffixCreditPolicy(parent,mean,std,centers,radius,precision='float64')
        obs,anchors,px=torch.tensor(qobs),torch.tensor(support),torch.tensor(protected)
        fq,fa,fp=[p.features(x).detach() for x in (obs,anchors,px)]
        transforms=[torch.tensor(null_space(fp[:,:,h].numpy(),rcond=1e-12),dtype=torch.float64) for h in range(3)]
        dimensions=[t.shape[1] for t in transforms]
        write_json(stage/'prefix_basis.json',dict(protected_counts=counts,free_dimensions=dimensions,
            maximum_nullspace_residual=[float(abs(fp[:,:,h]@transforms[h]).max()) if dimensions[h] else 0. for h in range(3)]))
        print('PREFIX_NULLSPACE',len(protected),dimensions,flush=True)
        if sum(dimensions)==0:raise RuntimeError('No free protected state-function directions')
        g=torch.tensor([q['gradient'] for q in queries],dtype=torch.float64)
        h=torch.zeros_like(g)
        for i,q in enumerate(queries):
            for head in range(3):
                lo,hi=q['records'][2*head:2*head+2]
                if lo['success'] and hi['success']:
                    h[i,head]=max(0.,(hi['absolute_loss']+lo['absolute_loss']-2*q['baseline_absolute_loss'])/suffix.AMPLITUDE[head]**2)
        weights=torch.tensor([q['stratum_size'] for q in queries],dtype=torch.float64)[:,None]
        limits=torch.tensor(suffix.LIMITS,dtype=torch.float64)
        normalization=(abs(g)*weights*limits).sum()
        z=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(n,dtype=torch.float64)) for n in dimensions])
        optimizer=torch.optim.Adam(z,lr=.003)
        coefficients=lambda:torch.stack([transforms[h]@z[h] for h in range(3)],-1)
        for step in range(250):
            c=coefficients();delta=(fq*c).sum(-2)*p.output_scale;anchor=(fa*c).sum(-2)*p.output_scale
            loss=((weights*(g*delta+.5*h*delta.square())).sum()/normalization+.02*(anchor/limits).square().mean())
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():
                maximum=float(abs((fa*coefficients()).sum(-2)*p.output_scale/limits).max())
                if maximum>1:
                    for v in z:v.div_(maximum)
        with torch.no_grad():p.coefficients.copy_(coefficients())
        identities=[np.array_equal(p.act(x,True)[0],p.base.act(x,True)[0]) for x in protected]
        max_error=max(float(abs(p.act(x,True)[0]-p.base.act(x,True)[0]).max()) for x in protected)
        write_json(stage/'protected_action_identity.json',dict(states=len(protected),all_bitwise_equal=all(identities),max_latent_error=max_error))
        assert (max_error<=1e-12 if args.numerical_prefix else all(identities)),f'Protected latent differences {max_error}'
        md=dict(architecture='local_suffix_credit',parent=parent,mean=mean,std=std,centers=centers,radius=radius,
            precision='float64',coefficients=p.coefficients.detach().numpy().copy(),step_scale=1.,
            environment_architecture='gate_band_canonical_kernel',learning_method='prefix_nullspace_joint_quadratic_credit')
        jobs=[]
        for scale in (.125,.5,1.):
            candidate=copy.deepcopy(md);candidate['step_scale']=scale;tag=f'protected_{scale}'
            torch.save(candidate,stage/(tag+'.pt'));jobs.append((tag,candidate))
        small=torch.load(local/'local_quadratic_0.125.pt',map_location='cpu',weights_only=False);small['step_scale']=.0625
        torch.save(small,stage/'unprotected_0.0625.pt');jobs.append(('unprotected_0.0625',small))
        pool=mp.get_context('spawn').Pool(4,suffix.initialize,(stage,parent,baseline,retained,source,started+230))
        for r in pool.imap_unordered(suffix.evaluate_network,jobs):
            outcomes.append(r);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('PROTECTED_CREDIT_ACTUAL',r['tag'],r['absolute_objective'],r['summary'],r['promotion'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['primary_preserved'] and c['absolute_objective']<queries[0]['baseline_absolute_loss']-1e-5]
        best=min(good,key=lambda c:c['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_prefix_preserving_joint_suffix_learning',new_candidates=len(outcomes),
            free_dimensions=dimensions,wall_s=time.perf_counter()-started,best=None if best is None else best['tag'],adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_prefix_preserving_joint_suffix_learning',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next','复核前缀保持的完整联合学习结果，继续修正原目标差距。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
