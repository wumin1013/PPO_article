"""Learn earlier and later action coordination with robust observable support."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from audit_context_restoration_scaling import solve_correction
from consolidate_joint_teacher_policy import collect_data
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--complete-preparation',action='store_true')
    parser.add_argument('--recover-credit',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_direct_stable_joint_learning_20260920'
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)=='2d482d47c0c5869a2155dd9021c4c7771f425c5242d06f738ff40d6ad50a5610'
    md=torch.load(source/'earlier_geometry_updated_initial_policy.pt',map_location='cpu',weights_only=False)
    assert not np.any(md['coefficients'])
    data=read(source/'baseline.json');data['baseline']=read(source/'results.json')['candidates'][0]
    baseline=data['baseline']['records'];support=read(source/'earlier_geometry_updated_support_audit.json')
    paths=support['affected_paths'];selected=[r for r in baseline if r['path'] in paths]
    with np.load(source/'earlier_geometry_updated_affected_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    assert all(identity.unchanged(p,np.zeros(27)) for p in paths)
    failed=BASE/'canonical_gate_band_early_geometry_joint_learning_20260920'
    previous=BASE/'canonical_gate_band_early_geometry_joint_completion_20260920'
    if args.recover_credit:
        assert read(previous/'receipt.json')['status'].startswith('completed')
        assert not read(previous/'receipt.json')['research_acceptable']
    if args.complete_preparation:
        assert read(failed/'receipt.json')['actual_new_candidates']==0
    stage=register('canonical_gate_band_early_geometry_credit_recovery_20260920' if args.recover_credit else
                  'canonical_gate_band_early_geometry_joint_completion_20260920' if args.complete_preparation
                   else 'canonical_gate_band_early_geometry_joint_learning_20260920',
        [__file__,parent_file,source/'receipt.json',source/'selection_review.json',
         source/'earlier_geometry_updated_initial_policy.pt',source/'earlier_geometry_updated_support_audit.json',
         source/'earlier_geometry_updated_affected_replay.npz',BASE/'saved_memory_action_identity.py',
         BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',BASE/'shape_event_credit.py',
         BASE/'learn_intrinsic_joint_recovery.py',BASE/'learn_early_memory_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'gate_band_state.py',BASE/'joint_runtime_v2.py']+
         ([failed/'receipt.json',failed/'decoder_identity_audit.json'] if args.complete_preparation else [])+
         ([previous/'receipt.json',previous/'state_joint/results.json',previous/'joint120_shape_localization.json',
           BASE/'audit_context_restoration_scaling.py'] if args.recover_credit else []),
        dict(wall_s=180 if args.recover_credit else 360,workers=4,
             symmetric_responses=54 if args.recover_credit else 0,
             maximum_recovery_vectors=6 if args.recover_credit else 0,
             response_steps=[.0002,.0005,.0005] if args.recover_credit else None,
             arms=[] if args.recover_credit else [dict(name='early_only',parameters=9,max_evaluations=96),
             dict(name='joint',parameters=27,max_evaluations=224)],coefficient_bounds=[-1.,1.],
             initial_radius=.5,parameter_scales=[.02,.05,.05],additional_lateral_squared_weight=4.),
        ('The completed early/joint comparison measured233 new points and retained no model. '
         'joint_120 reduces104 significant lateral motion0.014428951 to0.013767403mm '
         'but misses exactly one protection: total physical reverse rises0.000274923deg, '
         'localized to the approach. Do not extend COBYLA. Reuse every old result, '
         'collect54 central joint parameter responses around this ACTUAL improved '
         'trajectory with steps0.0002/0.0005/0.0005, and solve small L1 corrections '
         'under the same exact event-count equivalence and raw physical guards. '
         'Compare predicted zero versus1e-6 positive optimization margin using the '
         'same measured Jacobian; test at most6 new corrections with measured secant '
         'updates. Margins tighten predictions, never loosen acceptance.180s total. '
         'On persistent nonlinearity or infeasibility change action support, not budget. '
         if args.recover_credit else '')+
        'The latest actual-return optimizer improves104 outgoing excursion and '
        'all curve guards but leaves the188-266ms incoming wrong-way motion '
        'unchanged because its new action support begins300ms. Preserve that '
        'audited model as parent. Learn27 weights over three observed-state '
        'centers: early100ms full-state support and300/320ms reference-geometry '
        'support, each with three actions and two causal memories. Time/path '
        'indices only select training observations. Compare an earlier-only '
        '9-weight arm against the27-weight joint arm, each from the exact parent; '
        '96 versus224 evaluations allow for dimension, not a final equal-budget '
        'performance claim. Add4 times the squared continuous incoming lateral '
        'movement term to the existing actual complete shape loss, leaving '
        'ALL original/current physical,shape,curve,speed,cruise and completion '
        'guards untouched. Support reaches104 plus tiny features on15/207/307. '
        'Use shared exact action arithmetic to prove any unchanged complete '
        'route from saved parent states; otherwise execute it fully. No numerical '
        'feature threshold, saved trajectory template, PPO or sealed testing. '
        'At360s retain only real feasible improvement; on failure audit learned '
        'early/later support and responses, not a longer identical training run.')
    started=time.perf_counter();deadline=started+(165 if args.recover_credit else 340);folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);choice=dict(paths=paths,variable_kernels=[0,1,2])
    write_json(stage/'support_audit.json',dict(selected=choice,source_audit=support))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
    def score(records):
        return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r)+
            (4*(r['shape']['reverse_lateral_total_mm']/.005)**2 if r['success'] else 1e6)
            for r,b,o in zip(records,selected,retained)))
    def entry(records,u,tag,actual,unchanged):
        original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).reshape(-1).tolist(),records=records,score=score(records),
            measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=actual,exact_zero_support_paths=unchanged,
            reuse_basis='Identical decoded parent actions at every saved observation, not a small-feature threshold.')
    zero=entry(copy.deepcopy(selected),np.zeros(27),'reused_parent',[],paths);history=[zero];md['reused_candidates']=1
    if args.recover_credit:
        history=copy.deepcopy(read(previous/'state_joint/results.json')['candidates'])
        assert history[0]['score']==zero['score']
        md['reused_candidates']=len(history)
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;blocks=[];cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    try:
        # Recheck the exact actor after sharing residual arithmetic with the identity helper.
        checked=collect_data(md,md['parent'],selected,include_latents=True,deadline=deadline)
        assert np.array_equal(checked['parent_latents'],replay['parent_latents'])
        write_json(stage/'exact_action_reuse_audit.json',dict(paths=paths,decisions=len(replay['path_ids']),
            zero_policy_replay_decoded_bitwise=True,parent_latents_bitwise=True,
            shared_execution_arithmetic=True,no_new_physical_runs=True))
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u,arm):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if not parameters_in_bounds(u,-1.,1.):
                return dict(score=1e8+float(np.square(u).sum()),augmented_learning_inequalities=[-100.]*len(zero['augmented_learning_inequalities']))
            if key not in cache:
                if time.perf_counter()>deadline:raise TimeoutError('Earlier joint learning budget')
                tag=f'{arm}_{len(history)-1}'
                unchanged=[p for p in paths if identity.unchanged(p,u)]
                affected=[p for p in paths if p not in unchanged]
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in affected]);by_path={r['path']:r for r in measured}
                records=[by_path.get(r['path'],copy.deepcopy(r)) for r in selected]
                c=entry(records,u,tag,affected,unchanged);history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                if len(history)%10==0:
                    r=next(r for r in records if r['path']==104)
                    print('EARLY_GEOMETRY',arm,len(history)-1,c['score'],min(c['augmented_measured_inequalities']),
                        r['shape']['reverse_lateral_significant_mm'],r['shape']['out_edge_outward_mm'],best()['score'],flush=True)
            return cache[key]
        if args.recover_credit:
            origin=next(c for c in history if c['tag']=='joint_120')
            u=np.array(origin['coefficients']);steps=np.tile([.0002,.0005,.0005],9)
            pairs=[]
            for j in range(27):
                delta=np.eye(27)[j]*steps
                plus=evaluate(u+delta,'response_plus');minus=evaluate(u-delta,'response_minus')
                pairs.append((plus,minus))
            jac=np.array([(np.array(a['augmented_learning_inequalities'])-
                           np.array(b['augmented_learning_inequalities']))/2 for a,b in pairs]).T
            gradient=np.array([(a['score']-b['score'])/2 for a,b in pairs])
            write_json(stage/'measured_joint_credit.json',dict(anchor=origin['tag'],steps=steps.tolist(),
                jacobian=jac.tolist(),loss_gradient=gradient.tolist(),
                response_pairs=[[a['tag'],b['tag']] for a,b in pairs]))
            current=origin;proposals=[]
            # Two matched margins use the identical origin and measured response model.
            for margin in (0.,1e-6):
                solution=solve_correction(jac,gradient,np.array(origin['augmented_learning_inequalities']),
                    0.,margin=margin,radius=4.)
                row=dict(anchor=origin['tag'],margin=margin,solution=solution)
                if solution['success']:
                    delta=np.array(solution['scaled_parameter_step'])
                    measured=evaluate(u+delta*steps,'margin_recovery')
                    row.update(measured=measured['tag'],actual_guard_min=min(measured['augmented_measured_inequalities']))
                    proposals.append(measured)
                blocks.append(row)
            if proposals:
                current=min(proposals,key=lambda c:max(0.,-min(c['augmented_measured_inequalities'])))
            for iteration in range(4):
                if feasible(current):break
                solution=solve_correction(jac,gradient,np.array(current['augmented_learning_inequalities']),
                    max(0.,origin['score']-current['score']),margin=1e-6,radius=4.)
                row=dict(anchor=current['tag'],margin=1e-6,solution=solution)
                if not solution['success']:
                    blocks.append(row);break
                delta=np.array(solution['scaled_parameter_step'])
                measured=evaluate(np.array(current['coefficients'])+delta*steps,'secant_recovery')
                row.update(measured=measured['tag'],actual_guard_min=min(measured['augmented_measured_inequalities']))
                blocks.append(row)
                norm=float(delta@delta)
                if norm<=1e-18:break
                observed=np.array(measured['augmented_learning_inequalities'])-current['augmented_learning_inequalities']
                jac+=np.outer(observed-jac@delta,delta)/norm
                gradient+=(measured['score']-current['score']-gradient@delta)*delta/norm
                current=measured
            write_json(stage/'credit_recovery_audit.json',dict(blocks=blocks,
                original_acceptance_unchanged=True,selected=best()['tag']))
        for arm,budget in ([] if args.recover_credit else [('early_only',96),('joint',224)]):
            indices=np.arange(27).reshape(3,3,3)[:,0].reshape(-1) if arm=='early_only' else np.arange(27)
            scales=np.tile([.02,.05,.05],9)[indices]
            def expand(x):
                u=np.zeros(27);u[indices]=x*scales;return u
            previous=len(history)
            solved=minimize(lambda x:evaluate(expand(x),arm)['score'],np.zeros(len(indices)),method='COBYLA',
                bounds=[(-1./s,1./s) for s in scales],
                constraints=[dict(type='ineq',fun=lambda x:np.asarray(evaluate(expand(x),arm)['augmented_learning_inequalities']))],
                options=dict(maxiter=budget,rhobeg=.5,tol=1e-5,catol=1e-8))
            blocks.append(dict(arm=arm,parameters=len(indices),new_candidates=len(history)-previous,
                solver_message=str(solved.message),best=best()['tag']))
            write_json(stage/'arm_completions.json',blocks)
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_early_geometry_joint',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=len(history)-1,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
