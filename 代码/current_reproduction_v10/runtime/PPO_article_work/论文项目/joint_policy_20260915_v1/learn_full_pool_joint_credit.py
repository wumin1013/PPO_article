"""Full64-path measured joint parameter credit with fixed learned capacity."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.linalg import null_space
from scipy.optimize import minimize
from deep_joint_credit_policy import load_credit_policy
from dense_trace_credit_model import values,coordinates,coefficients,objective
from learn_prefix_preserving_suffix_credit import prefix_indices
from learn_trace_resolved_joint_credit import build_trace_response,trace_deficits
from learn_retained_joint_growth import growth_deficits,smooth_cruise
from learn_absolute_shape_joint_credit import absolute_shape_response,absolute_actual
from learn_nonlinear_joint_feedback import rollout,promotion
from canonical_gate_state import checkpoint_environment_factory
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}


def aggregate(records):
    success=all(r['success'] for r in records)
    shapes=[r for r in records if 'shape' in r]
    return dict(paths=len(records),successes=sum(r['success'] for r in records),
        complete_time_s=sum(r['complete_time_s'] for r in records) if success else None,
        curved_cruise99_s=sum(r['motion']['cruise_ge99_s'] for r in records if r['curved_path']) if success else None,
        smooth_curved_cruise99_s=sum(r['smooth_cruise99_s'] for r in records if r['curved_path']) if success else None,
        reverse_sum_deg=sum(r['shape']['full_physical_reverse_deg'] for r in shapes) if success else None,
        exit_sum_mm=sum(r['shape']['out_edge_outward_mm'] for r in shapes) if success else None,
        energy_sum=sum(r['shape']['full_normalized_energy'] for r in shapes) if success else None,
        reverse_lateral_sum_mm=sum(r['shape']['reverse_lateral_total_mm'] for r in shapes)
            if success and all('reverse_lateral_total_mm' in r['shape'] for r in shapes) else None,
        inward_first_turns=sum(r['target_motion']['first_turn_inside'] is True for r in shapes))


def candidate_metadata(metadata,directions,u):
    candidate=copy.deepcopy(metadata)
    candidate['coefficients']=metadata['coefficients']+np.einsum('ich,i->ch',directions,u)
    candidate['learning_method']='full64_complete_policy_joint_parameter_optimization'
    return candidate


def initialize(stage,metadata,directions,references,curved,retained,deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,metadata=metadata,directions=directions,references=references,curved=curved,
        retained=retained,factory=checkpoint_environment_factory('gate_band_canonical_kernel'),deadline=deadline)


def evaluate(job):
    tag,u=job
    md=candidate_metadata(WORK['metadata'],WORK['directions'],np.asarray(u));policy=load_credit_policy(md)
    folder=WORK['stage']/tag;folder.mkdir();records=[];costs=dict(actor_calls=0,physics_ticks=0)
    for path,reference in WORK['references'].items():
        r,rows,actions,observations=rollout(policy,WORK['factory'],reference,costs,WORK['deadline'])
        file=folder/f'path{path}.npz'
        np.savez_compressed(file,reference=reference,rows=rows,actions=actions,observations=observations)
        r.update(path=path,trace=file.relative_to(WORK['stage']).as_posix(),curved_path=WORK['curved'][path])
        if r['success']:r['smooth_cruise99_s']=smooth_cruise(rows)
        records.append(r)
    summary=aggregate(records)
    checks=promotion(records,WORK['retained'])
    checks['curved_cruise_present']=bool(summary['successes']==64 and summary['curved_cruise99_s']>0.)
    result=dict(tag=tag,parameters=np.asarray(u).tolist(),records=records,summary=summary,costs=costs,
                promotion=checks,primary_preserved=all(checks[k] for k in ('all_complete','inward_first_turns','entry_nonworse')))
    result['absolute_objective']=absolute_actual(result,WORK['retained']) if summary['successes']==64 else None
    write_json(WORK['stage']/(tag+'.json'),result)
    return result


def prepare_directions(metadata,baseline,older):
    policy=load_credit_policy(metadata);support=[];protected=[]
    for r in baseline['records']:
        with np.load(r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
        support.append(obs)
        if not r['curved_path']:protected.append(obs)
        elif 'shape' in r:protected.append(obs[prefix_indices(ref,rows,obs)])
    support=np.vstack(support);protected=np.vstack(protected)
    f=policy.features(torch.tensor(support)).detach().numpy()*policy.output_scale.numpy()
    fp=policy.features(torch.tensor(protected)).detach().numpy()
    transforms=[null_space(fp[:,:,h],rcond=1e-12) for h in range(3)]
    earlier_metadata=copy.deepcopy(metadata);earlier_metadata['coefficients']=metadata['coefficients'][:108]
    z=coordinates(older,earlier_metadata);v,j=values(older,z)
    _,overall=objective(older,z)
    gradients=[overall,j[9*14+5],j[9*14+4],j[9*14+3],j[9*14+6],
               j[9*12+4]+j[9*13+4]+j[9*12+5]+j[9*13+5],
               sum(j[9*p+3] for p in range(16)),j[4]+j[5]+j[9*3+4]+j[9*3+5]]
    a=older['anchors'];hessian=j.T@j/max(objective(older,z)[0],1.)+.04*a.T@a/len(a)+1e-5*np.eye(len(z))
    origin=coefficients(older,np.zeros_like(z));raw=[]
    for gradient in gradients:
        step=-np.linalg.solve(hessian,gradient)
        direction=np.vstack([coefficients(older,step)-origin,np.zeros((6,3))])
        for head,n in enumerate(transforms):direction[:,head]=n@(n.T@direction[:,head])
        raw.append(direction)
    raw=np.array(raw)
    response=np.einsum('sch,ich->shi',f,raw)/np.array([.003,.025,.025])[None,:,None]
    matrix=response.reshape(-1,len(raw))
    _,singular,vt=np.linalg.svd(matrix,full_matrices=False)
    rank=int((singular>singular[0]*1e-7).sum())
    directions=np.einsum('ij,jch->ich',vt[:rank],raw)/singular[:rank,None,None]
    response=np.einsum('sch,ich->shi',f,directions)/np.array([.003,.025,.025])[None,:,None]
    maximum=np.max(abs(response),axis=(0,1))
    directions*=.5/maximum[:,None,None]
    response=np.einsum('sch,ich->shi',f,directions)/np.array([.003,.025,.025])[None,:,None]
    trust=response.reshape(-1,rank)
    prefix=np.einsum('sch,ich->shi',fp,directions)*policy.output_scale.numpy()[None,:,None]
    assert abs(prefix).max()<1e-12,abs(prefix).max()
    return directions,trust,dict(directions=rank,centers=len(metadata['centers']),
        protected_states=len(protected),support_states=len(support),
        maximum_prefix_latent_response=float(abs(prefix).max()),
        maximum_head_latent_response=np.max(abs(response)*np.array([.003,.025,.025])[None,:,None],axis=0).tolist(),
        source_metric_gradient_only_proposes_subspace=True)


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919'
    full=BASE/'canonical_gate_band_repaired_training_pool_check_20260919'
    old=BASE/'canonical_gate_band_recentered_fine_trace_learning_20260919'
    original=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918'
    metadata=torch.load(parent/'research_candidate.pt',map_location='cpu',weights_only=False)
    assert sha(parent/'research_candidate.pt')==read(full/'receipt.json')['source_model_sha256']
    records=copy.deepcopy(read(full/'results.json')['records']);refs={};curved={}
    for r in records:
        r['trace']=str(BASE/r['trace'])
        with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy();rows=z['rows'].copy()
        r['smooth_cruise99_s']=smooth_cruise(rows);curved[r['path']]=r['curved_path']
    baseline=dict(tag='current64',records=records,summary=aggregate(records))
    pool_data=read(original/'results.json')['full_candidate']
    shapes={r['path']:r for r in pool_data['summary']['shapes']}
    retained=[]
    for r in pool_data['results']:
        row=dict(path=r['original_path'],complete_time_s=r['completion_time_s'],complete_time_change_s=0.)
        if row['path'] in shapes:row['baseline_shape']=shapes[row['path']]
        retained.append(row)
    baseline['absolute_objective']=absolute_actual(baseline,retained)
    older=torch.load(old/'response_model.pt',map_location='cpu',weights_only=False)
    stage=register('canonical_gate_band_full_pool_joint_credit_learning_20260919',
        [__file__,BASE/'learn_trace_resolved_joint_credit.py',BASE/'complete_local_credit.py',
         BASE/'local_suffix_credit_policy.py',BASE/'learn_absolute_shape_joint_credit.py',
         parent/'research_candidate.pt',full/'results.json',full/'receipt.json',old/'response_model.pt',original/'results.json'],
        dict(wall_s=720,workers=4,maximum_joint_directions=8,maximum_response_candidates=16,
             proposal_candidates=2,paths_per_candidate=64,centers=114,network_layers_added=0),
        'Current repaired learned actor passed64/64 with88.818s,6.867s curved99 '
        'cruise,16.1111deg reversal and0.033614mm exit. Remaining155deg/140deg '
        'curvature,energy and apex-speed defects preclude adoption. Replace '
        '20-path-only parameter credit with EXACT full64-policy responses. '
        'Use earlier small-amplitude gradients ONLY to propose at most eight '
        'joint parameter directions emphasizing the worst shape deficits; '
        'project onto current observed straight/isolated-entry nullspaces, '
        'orthogonalize in state-action response space, preserve114 centers '
        'and learned architecture, and measure both signs on all64 paths. '
        'Only fully complete sides supply complete metric/trace response; '
        'one-sided feasible directions or exclusion handle failures. Compose '
        'signedXY/angular/spatial traces before nonlinear shape aggregation. '
        'Constrain every modeled path margin, per-corner regret, complete '
        'time and curved cruise before half/full joint proposals. Execute '
        'both on full64. Failed prefixes receive no complete-time benefit. '
        'Retain86cd and the64/64 research checkpoint; no sealed test, fair '
        'baseline claim, PPO claim or runtime hand rule is introduced.')
    started=time.perf_counter();pool=None;outcomes=[];jobs=[]
    write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next',
        '直接用完整64路径的联合参数响应修正大角度形状，同时守住整池可行性、时间和巡航。'],check=True,stdout=subprocess.DEVNULL)
    try:
        directions,trust,audit=prepare_directions(metadata,baseline,older)
        n=len(directions);write_json(stage/'direction_audit.json',audit)
        np.savez_compressed(stage/'parameter_space.npz',directions=directions,trust=trust)
        print('FULL_POOL_DIRECTIONS',audit,flush=True)
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,directions,refs,curved,retained,started+700))
        response_jobs=[(f'axis{i}_{s:+d}',s*np.eye(n)[i]) for i in range(n) for s in (-1,1)]
        for c in pool.imap_unordered(evaluate,response_jobs):
            outcomes.append(c);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('FULL_POOL_RESPONSE',c['tag'],c['summary'],flush=True)
        base=growth_deficits(baseline,retained);columns=[];bounds=[];margins=[];paired=[];modes=[]
        bm=np.array([r['corridor']['margin_lower_mm'] for r in records])
        for i in range(n):
            lo=next(c for c in outcomes if c['tag']==f'axis{i}_-1');hi=next(c for c in outcomes if c['tag']==f'axis{i}_+1')
            paired.append((lo,hi));lok=lo['primary_preserved'];hik=hi['primary_preserved']
            if lok and hik:
                columns.append((growth_deficits(hi,retained)-growth_deficits(lo,retained))/2.);bounds.append((-1.,1.));modes.append('central_complete64')
                margins.append((np.array([r['corridor']['margin_lower_mm'] for r in hi['records']])-np.array([r['corridor']['margin_lower_mm'] for r in lo['records']]))/2.)
            elif hik:
                columns.append(growth_deficits(hi,retained)-base);bounds.append((0.,1.));modes.append('positive_complete64')
                margins.append(np.array([r['corridor']['margin_lower_mm'] for r in hi['records']])-bm)
            elif lok:
                columns.append(base-growth_deficits(lo,retained));bounds.append((-1.,0.));modes.append('negative_complete64')
                margins.append(bm-np.array([r['corridor']['margin_lower_mm'] for r in lo['records']]))
            else:
                columns.append(np.zeros_like(base));bounds.append((0.,0.));modes.append('neither_complete64_frozen');margins.append(np.zeros_like(bm))
        matrix=np.column_stack(columns);margin_jac=np.column_stack(margins)
        traces=build_trace_response(stage,baseline,paired)
        guards=[];time_indices=[];offset=0
        for r in records:
            if 'shape' in r:guards.extend(offset+k for k in range(2,7));time_indices.append(offset+8);offset+=9
            else:time_indices.append(offset+1);offset+=2
        guards=np.array(guards);upper=np.maximum(base[guards],0.)
        normalization=max(baseline['absolute_objective'],1.)
        def response(u):return trace_deficits(u,base,matrix,traces,baseline,retained)
        def fun(u):
            v,j=response(u);v,j=absolute_shape_response(v,j,records,retained);positive=np.maximum(v,0.)
            return float(positive@positive/normalization+.001*(u@u)),2*j.T@positive/normalization+.002*u
        time_jac=.02*matrix[time_indices].sum(0)
        # growth_deficits[-1]=(3-smooth_curved_cruise)*5.
        constraints=[dict(type='ineq',fun=lambda u:np.r_[2.-trust@u,2.+trust@u],jac=lambda u:np.vstack([-trust,trust])),
            dict(type='ineq',fun=lambda u:(bm+margin_jac@u-np.minimum(bm,.02))/.01,jac=lambda u:margin_jac/.01),
            dict(type='ineq',fun=lambda u:upper-response(u)[0][guards],jac=lambda u:-response(u)[1][guards]),
            dict(type='ineq',fun=lambda u:np.array([.02-time_jac@u,.1-matrix[-1]@u]),jac=lambda u:np.vstack([-time_jac,-matrix[-1]]))]
        best=np.zeros(n);best_value=fun(best)[0]
        def remember(u):
            nonlocal best,best_value
            if all(np.min(c['fun'](u))>=-1e-7 for c in constraints) and fun(u)[0]<best_value:
                best=u.copy();best_value=fun(u)[0]
                np.savez_compressed(stage/'feasible_optimizer_iterate.npz',parameters=best)
        solution=minimize(fun,np.zeros(n),jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                          callback=remember,options=dict(maxiter=250,ftol=1e-9))
        remember(solution.x)
        write_json(stage/'optimizer.json',dict(converged=bool(solution.success),message=solution.message,
            response_modes=modes,parameters=best.tolist(),predicted_objective=fun(best)[0]*normalization,
            maximum_function_change=float(abs(trust@best).max()),all_modeled_constraints_pass=all(np.min(c['fun'](best))>=-1e-7 for c in constraints)))
        proposal_jobs=[(f'full_pool_joint_{s}',s*best) for s in (.5,1.)] if np.max(abs(best))>1e-7 else []
        for tag,u in proposal_jobs:torch.save(candidate_metadata(metadata,directions,u),stage/(tag+'.pt'))
        for c in pool.imap_unordered(evaluate,proposal_jobs):
            outcomes.append(c);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('FULL_POOL_JOINT_ACTUAL',c['tag'],c['summary'],c['promotion'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['primary_preserved'] and c['absolute_objective']<baseline['absolute_objective']
              and c['summary']['complete_time_s']<=baseline['summary']['complete_time_s']+.02
              and c['summary']['curved_cruise99_s']>=baseline['summary']['curved_cruise99_s']-.05
              and c['summary']['reverse_sum_deg']<baseline['summary']['reverse_sum_deg']
              and c['summary']['exit_sum_mm']<=baseline['summary']['exit_sum_mm']
              and c['summary']['energy_sum']<=baseline['summary']['energy_sum']]
        selected=min(good,key=lambda c:c['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if selected:
            result['best']=selected['tag'];md=candidate_metadata(metadata,directions,np.array(selected['parameters']))
            torch.save(md,stage/(selected['tag']+'.pt'));torch.save(md,stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_full_pool_joint_credit_learning',full64_candidates=len(outcomes),
            best=None if selected is None else selected['tag'],wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_full_pool_joint_credit_learning',error=str(exc),full64_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '依据完整64路径参数对照继续修正大角度形状和整路效率，尚未冻结公平比较及泛化方案。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
