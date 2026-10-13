"""Restore shared-policy feasibility using measured local credit at the actual candidate."""
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
from scipy.optimize import linprog
from duration_joint_metrics import guards,penalty
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_complete_joint_teachers import parameters_in_bounds
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read
from audit_context_restoration_scaling import solve_correction


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--zero-margin',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    original_source=BASE/'canonical_gate_band_context_postpeak_memory_learning_20260920'
    source=BASE/'canonical_gate_band_recentered_context_memory_credit_20260920' if args.zero_margin else original_source
    assert not read(source/'receipt.json')['research_acceptable']
    audit=read(original_source/'prediction_error_audit.json')
    assert [r['tag'] for r in audit['alternate_feasible_candidates']]==['reused_zero_function']
    parent_file=BASE/'canonical_gate_band_main_turn_memory_joint_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    model_file=source/('recentered_credit_model.json' if args.zero_margin else 'local_credit_model.json')
    model=read(model_file);steps=np.array(model['steps'])
    stage=register('canonical_gate_band_zero_margin_context_credit_20260920' if args.zero_margin else
                   'canonical_gate_band_recentered_context_memory_credit_20260920',
        [__file__,BASE/'duration_joint_metrics.py',BASE/'observable_memory_joint_policy.py',
         BASE/'learn_early_memory_joint_policy.py',BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py',parent_file,source/'receipt.json',
         original_source/'prediction_error_audit.json',original_source/'context_credit_validation.json',
         model_file,source/'state_joint/results.json',source/'state_joint/candidate.pt']+
         ([BASE/'audit_context_restoration_scaling.py',source/'restoration_scaling_audit.json'] if args.zero_margin else []),
        dict(wall_s=120 if args.zero_margin else 300,workers=4,parameters=27,maximum_bias_corrected_updates=12 if args.zero_margin else 6,
             maximum_recentered_axis_responses=0 if args.zero_margin else 54,maximum_fine_joint_updates=0 if args.zero_margin else 6,
             local_axis_step_factor=.125,coefficient_bounds=[-1.,1.]),
        ('The matched finer response model fails only the EXTRA positive optimization '
         'margin test; both raw and row-equilibrated LPs find a zero-margin correction '
         'with nonnegative original predicted guards. Reuse all160 completed vectors '
         'and the54 finer matched responses. Change only this optional solver margin '
         'to zero, with row equilibration and1e-9 LP residual tolerance. Validate at '
         'most12 measured-intercept joint updates, preserving every actual acceptance '
         'threshold including1e-8 normalized feasibility, narrow304 research protection '
         'and all physical limits. No additional axis sampling or optimizer run. '
         if args.zero_margin else '')+
        'No saved contextual candidate passes all actual guards even when the tiny '
        'narrow304 improvement is removed in a diagnostic-only comparison. Retain that '
        'improvement and every original guard. The nearest candidate fails continuous '
        'shape constraints at normalized1e-7 to5e-7 while their zero-centered model '
        'predicts positive slack. Test an observed-intercept correction using the same '
        '27-parameter function and cached gradients, at most six complete four-route '
        'joint updates. If none passes, collect matched plus/minus responses at one '
        'eighth of old axis scale around the ACTUAL low-violation candidate, then test '
        'at most six new joint corrections. All106 prior vectors are reused; no '
        'optimizer continuation or extra feature is added. Candidates must improve '
        'the protected full loss and every recorded shape/duration/speed/cruise/physics '
        'guard. This tests local complete-response credit accuracy, not PPO or bias.')
    started=time.perf_counter();deadline=started+(100 if args.zero_margin else 280);folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records'];paths=[14,104,204,304]
    selected=[r for r in baseline if r['path'] in paths]
    support=read(source/'support_audit.json');choice=support['selected']
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    old_score,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    def score(records):return old_score(records)+sum(penalty(r) for r in records)
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    reference_score=score(selected)
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    for c in history:
        assert abs(c['score']-score(c['records']))<1e-10
        assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
        c['tag']='reused_'+c['tag']
    md['reused_candidates']=len(history);cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():
        candidates=[c for c in history if feasible(c) and c['score']<reference_score-.001]
        return min(candidates,key=lambda c:c['score']) if candidates else None
    def anchor():
        candidates=[c for c in history if c['score']<reference_score-.001 and parameters_in_bounds(c['coefficients'],-1.,1.)]
        return min(candidates,key=lambda c:np.square(np.minimum(c['measured_inequalities'],0.)).sum())
    print('RECENTER_CONTEXT',anchor()['tag'],'score',anchor()['score'],'guard',min(anchor()['measured_inequalities']),flush=True)
    blocks=[];pool=None
    def finalize(exhausted=False):
        selected_best=best()
        if selected_best:
            finish(stage,md,baseline,data,selected_best,dict(score=reference_score),history,blocks,choice,parent_file,started,budget_exhausted=exhausted)
        else:
            torch.save(md,folder/'candidate.pt')
            write_json(folder/'results.json',dict(candidates=history,best=None,nearest=anchor()['tag']))
            write_json(stage/'optimization_blocks.json',blocks)
            write_json(stage/'results.json',dict(candidates=[],research_acceptable=False,adopted=False,goal_complete=False))
            receipt(stage,dict(status='completed_recentered_context_no_feasible_update',research_acceptable=False,
                actual_new_candidates=len(history)-md['reused_candidates'],reused_candidates=md['reused_candidates'],
                source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,budget_exhausted=exhausted,
                adopted=False,goal_complete=False))
            print('RECENTER_COMPLETE no feasible update',anchor()['tag'],min(anchor()['measured_inequalities']),flush=True)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'recenter_{len(history)}';records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=paths,exact_zero_support_paths=[])
                history.append(c);cache[key]=c;winner=best()
                write_json(folder/'results.json',dict(candidates=history,best=winner['tag'] if winner else None,nearest=anchor()['tag']))
                if (len(history)-md['reused_candidates'])%5==0:
                    print('RECENTER_MEASURED',len(history)-md['reused_candidates'],'best',winner['score'] if winner else None,
                        'near',anchor()['tag'],min(anchor()['measured_inequalities']),flush=True)
            return cache[key]
        def restore(jac,g,scale_steps,rounds,label):
            for iteration in range(rounds):
                current=anchor();u=np.array(current['coefficients']);values=np.array(current['measured_inequalities'])
                # A positive margin is an optimization aim, not a relaxed acceptance tolerance.
                variable=np.max(abs(jac),axis=1)>1e-9
                margin=np.where(variable,2e-7,0.)
                n=len(u)
                a=np.vstack([np.c_[-jac,np.zeros((len(values),n))],
                    np.c_[np.eye(n),-np.eye(n)],np.c_[-np.eye(n),-np.eye(n)],np.r_[g,np.zeros(n)][None]])
                b=np.r_[values-margin,np.zeros(2*n),reference_score-.001-current['score']]
                bounds=[(max(-.5,(-1.-v)/h),min(.5,(1.-v)/h)) for v,h in zip(u,scale_steps)]+[(0.,None)]*n
                if args.zero_margin:
                    solved=solve_correction(jac,g,values,reference_score-.001-current['score'])
                    success=solved['success'];message=solved['status']
                    delta=np.asarray(solved['scaled_parameter_step']) if success else None
                else:
                    lp=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=a,b_ub=b,bounds=bounds,method='highs')
                    success=lp.success;message=str(lp.message);delta=lp.x[:n] if success else None
                row=dict(method=label,iteration=iteration,anchor=current['tag'],lp_status=message)
                if not success:
                    blocks.append(row);break
                proposal=evaluate(u+scale_steps*delta)
                row.update(tag=proposal['tag'],predicted_minimum=float(np.min(values+jac@delta)),
                    actual_minimum=min(proposal['measured_inequalities']),actual_loss=proposal['score'])
                blocks.append(row);write_json(stage/'optimization_blocks.json',blocks)
                if best():return
                if anchor()['tag']==current['tag']:break
        restore(np.asarray(model['central_constraints']),np.asarray(model['central_loss']),steps,12 if args.zero_margin else 6,
                'zero_margin_measured_intercept_correction' if args.zero_margin else 'measured_intercept_correction')
        if not best() and not args.zero_margin:
            center=anchor();u=np.asarray(center['coefficients']);fine_steps=.125*steps
            jac=[];g=[];pairs=[]
            for i,h in enumerate(fine_steps):
                delta=np.zeros(27);delta[i]=h;plus=evaluate(u+delta);minus=evaluate(u-delta)
                jac.append(.5*(np.asarray(plus['measured_inequalities'])-minus['measured_inequalities']))
                g.append(.5*(plus['score']-minus['score']))
                pairs.append(dict(index=i,step=h,plus=plus['tag'],minus=minus['tag']))
            jac=np.asarray(jac).T;g=np.asarray(g)
            write_json(stage/'recentered_credit_model.json',dict(anchor=center['tag'],steps=fine_steps.tolist(),pairs=pairs,
                central_constraints=jac.tolist(),central_loss=g.tolist(),
                old_slope_at_fine_scale_max_difference=float(abs(jac-.125*np.asarray(model['central_constraints'])).max())))
            restore(jac,g,fine_steps,6,'fine_recentered_complete_credit')
        pool.close();pool.join();pool=None;finalize()
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finalize(True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_recentered_context_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
