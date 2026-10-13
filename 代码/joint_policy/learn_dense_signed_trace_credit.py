"""Joint learning with dense signed-trajectory credit and per-corner guards."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
import learn_stratified_suffix_joint_credit as suffix
from dense_trace_credit_model import build_model,values,objective,coefficients,coordinates
from learn_retained_joint_growth import growth_deficits
from learn_absolute_shape_joint_credit import absolute_actual
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-valid-half',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_dense_margin_suffix_learning_20260919'
    sampled=BASE/'canonical_gate_band_complete_current_suffix_credit_20260919'
    previous=BASE/'canonical_gate_band_per_corner_dense_credit_20260919'
    plan=read(sampled/'query_plan.json');baseline=plan['baseline'];parent=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    assert sha(source/'research_candidate.pt')==plan['parent_sha256']
    metadata=torch.load(previous/'per_corner_constrained_0.125.pt',map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    queries=sorted(read(sampled/'query_results.json')['queries'],key=lambda q:q['query'])
    prior_trace=BASE/'canonical_gate_band_dense_signed_trace_joint_credit_20260919'
    stage=register('canonical_gate_band_dense_signed_trace_completion_20260919' if args.resume_valid_half else
                   'canonical_gate_band_dense_signed_trace_joint_credit_20260919',
        [__file__,BASE/'dense_trace_credit_model.py',BASE/'learn_trace_resolved_joint_credit.py',
         BASE/'local_suffix_credit_policy.py',BASE/'learn_absolute_shape_joint_credit.py',
         sampled/'query_plan.json',sampled/'query_results.json',sampled/'receipt.json',
         source/'research_candidate.pt',previous/'per_corner_constrained_0.125.pt',previous/'results.json',
         previous/'actual_guard_violations.json',previous/'receipt.json']+
         ([prior_trace/'signed_trace_0.0625.pt',prior_trace/'optimizer.json',prior_trace/'model_validation.json',prior_trace/'receipt.json'] if args.resume_valid_half else []),
        dict(wall_s=600,reused_queries=638,reused_branches=3828,new_candidates_maximum=2,
             paths=20,workers=4,trust_scales=[.0625,.125]),
        ('The previous run solved and saved the0.0625 actor but stopped before ALL '
         'new physics when the0.125 solve reached its iteration limit. Reuse the '
         'saved half-scale actor directly and evaluate it while preparing the larger '
         'solve. Warm-start0.125 from that feasible half-scale iterate, not zero. '
         'Save solver iterates and feasibility even if convergence is not reported. '
         'A feasible, improving but nonconverged iterate may be physically tested '
         'and must be explicitly labeled nonconverged; never claim optimality. '
         'An infeasible larger solve does not discard the valid smaller actor. '
         'No previously completed physical candidate is repeated. ' if args.resume_valid_half else '')+
        'The dense linear per-corner guards reduced total energy but actual peak '
        'curvature still regressed at100deg/155deg;155deg energy and apex speed '
        'also slightly violated the local guards. Reuse all638 complete-decision '
        'action responses at the SAME parent and SAME108-center actor. Before '
        'aggregating max/hinge/integral/energy metrics, sum the action-response '
        'Jacobians of signed incoming/outgoing XY distances, actual1ms angular '
        'velocity and0.1mm spatial turning profiles into the parameter Jacobian. '
        'Apply the original nonlinear metrics only AFTER simultaneous parameter '
        'effects are composed. Full timing/cruise/apex-speed retain their dense '
        'scalar response for this controlled change. Completed shorter trajectories '
        'stay at their recorded stopped endpoint; failed prefixes supply no shape '
        'derivatives or efficiency and are never extended. The nominal temporal/'
        'spatial grid is an approximate response model, not a deployed trajectory '
        'template. Check baseline metric identity and predict the two ALREADY '
        'measured per-corner policies. If normalized shape prediction error improves, '
        'optimize two new full policies with the same prefix, margin and per-corner '
        'guards, then validate exact full trajectories. Otherwise stop this model '
        'before new physical trials and change mechanism. No PPO claim or adoption '
        'without original physical/shape/cruise/full64/fair/generalization gates.')
    started=time.perf_counter();pool=None;outcomes=[];half_future=None;reused_half=None
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next','用逐点带符号轨迹组成联合响应，再计算完整形状指标并校验逐角约束。'],
        check=True,stdout=subprocess.DEVNULL)
    try:
        if args.resume_valid_half:
            reused_half=torch.load(prior_trace/'signed_trace_0.0625.pt',map_location='cpu',weights_only=False)
            torch.save(reused_half,stage/'signed_trace_0.0625.pt')
            pool=mp.get_context('spawn').Pool(1,suffix.initialize,(stage,parent,baseline,retained,source,started+590))
            half_future=pool.apply_async(suffix.evaluate_network,(('signed_trace_0.0625',reused_half),))
        m=build_model(baseline,metadata,source,sampled,queries,retained,np.array(plan['amplitude']))
        n=m['metric_jac'].shape[1];zero=np.zeros(n);value,jac=values(m,zero)
        identity=abs(objective(m,zero)[0]-absolute_actual(baseline,retained))
        assert identity<1e-7,identity
        indices=[];offset=0
        for r in baseline['records']:
            if 'shape' in r:indices.extend([offset+k for k in (2,3,4,5,6)]);offset+=9
            else:offset+=2
        indices=np.array(indices);upper=np.maximum(m['base'][indices],0.)
        checks=[];linear_error=trace_error=0.
        controls=read(previous/'results.json')['candidates']
        for scale in (.0625,.125):
            tag=f'per_corner_constrained_{scale}'
            md=torch.load(previous/(tag+'.pt'),map_location='cpu',weights_only=False)
            z=coordinates(m,md);actual=next(c for c in controls if c['tag']==tag)
            av=growth_deficits(actual,retained);lv,_=values(m,z,True);tv,_=values(m,z)
            le=float(np.square(lv[indices]-av[indices]).sum());te=float(np.square(tv[indices]-av[indices]).sum())
            linear_error+=le;trace_error+=te
            checks.append(dict(tag=tag,linear_normalized_error=le,trace_normalized_error=te,
                predicted_linear_objective=objective(m,z,True)[0],predicted_trace_objective=objective(m,z)[0],
                actual_objective=actual['absolute_objective']))
        write_json(stage/'model_validation.json',dict(baseline_identity_error=identity,controls=checks,
            total_linear_error=linear_error,total_trace_error=trace_error,improved=trace_error<linear_error))
        print('DENSE_TRACE_VALIDATION',checks,'errors',linear_error,trace_error,flush=True)
        if not trace_error<linear_error:
            receipt(stage,dict(status='completed_dense_trace_model_not_improved',new_candidates=0,
                wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));return
        anchors=m['anchors'];active=m['margins']<.05;mg=m['margin_jac'][active];mb=m['margins'][active]
        normalization=max(absolute_actual(baseline,retained),1.)
        diagonal=np.maximum(np.sqrt((jac*jac).sum(0)/normalization+.04*(anchors*anchors).mean(0)),1e-4)
        trust=anchors/diagonal;gs=mg/diagonal
        jobs=[('signed_trace_0.0625',reused_half)] if args.resume_valid_half else [];audits=[]
        for scale in (.0625,.125):
            if args.resume_valid_half and scale==.0625:
                audits.append(dict(read(prior_trace/'optimizer.json')['candidates'][0],reused_optimized_parameters=True))
                continue
            def fun(y):
                z=y/diagonal;value,gradient=objective(m,z)
                anchor=anchors@z
                return value/normalization+.02*float(np.mean(anchor*anchor)),(gradient/normalization+.04*anchors.T@anchor/len(anchor))/diagonal
            def guard(y):return upper-values(m,y/diagonal)[0][indices]
            def guard_jac(y):return -values(m,y/diagonal)[1][indices]/diagonal
            constraints=[dict(type='ineq',fun=lambda y:np.r_[scale-trust@y,scale+trust@y],jac=lambda y:np.vstack([-trust,trust])),
                dict(type='ineq',fun=lambda y:(mb+gs@y-np.minimum(mb,.02))/.01,jac=lambda y:gs/.01),
                dict(type='ineq',fun=guard,jac=guard_jac)]
            initial=coordinates(m,reused_half)*diagonal if args.resume_valid_half else np.zeros(n)
            solution=minimize(fun,initial,jac=True,method='SLSQP',constraints=constraints,
                options=dict(maxiter=200,ftol=1e-8))
            z=solution.x/diagonal
            feasible=bool(abs(anchors@z).max()<=scale+1e-7 and guard(solution.x).min()>=-1e-6
                and ((mb+gs@solution.x-np.minimum(mb,.02))/.01).min()>=-1e-6)
            np.savez_compressed(stage/f'solver_iterate_{scale}.npz',coordinates=z)
            write_json(stage/f'solver_status_{scale}.json',dict(converged=bool(solution.success),feasible=feasible,
                message=solution.message,iterations=int(solution.nit),predicted_objective=objective(m,z)[0]))
            if args.resume_valid_half:
                if not feasible or objective(m,z)[0]>=objective(m,zero)[0]:continue
            else:assert solution.success and feasible,solution.message
            candidate=copy.deepcopy(metadata);candidate['coefficients']=coefficients(m,z);candidate['step_scale']=1.
            candidate['learning_method']='dense_signed_trace_joint_parameter_optimization'
            with torch.no_grad():m['policy'].coefficients.copy_(torch.tensor(candidate['coefficients']))
            error=max(float(abs(m['policy'].act(x,True)[0]-m['policy'].base.act(x,True)[0]).max()) for x in m['protected'])
            assert error<=1e-12,error
            tag=f'signed_trace_{scale}';torch.save(candidate,stage/(tag+'.pt'));jobs.append((tag,candidate))
            audits.append(dict(tag=tag,iterations=int(solution.nit),predicted_objective=objective(m,z)[0],
                solver_converged=bool(solution.success),solver_message=solution.message,
                maximum_protected_latent_difference=error,minimum_model_guard= float(guard(solution.x).min())))
            write_json(stage/'optimizer.json',dict(candidates=audits));print('DENSE_TRACE_PROPOSAL',audits[-1],flush=True)
        if pool is None:pool=mp.get_context('spawn').Pool(4,suffix.initialize,(stage,parent,baseline,retained,source,started+590))
        pending=jobs[1:] if args.resume_valid_half else jobs
        actual_results=([half_future.get()] if args.resume_valid_half else [])
        actual_results.extend(pool.imap_unordered(suffix.evaluate_network,pending))
        for result in actual_results:
            outcomes.append(result);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('DENSE_TRACE_ACTUAL',result['tag'],result['absolute_objective'],result['summary'],result['promotion'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['primary_preserved'] and c['absolute_objective']<absolute_actual(baseline,retained)-1e-5
            and c['summary']['reverse_sum_deg']<=baseline['summary']['reverse_sum_deg']+1e-6
            and c['summary']['exit_sum_mm']<=baseline['summary']['exit_sum_mm']+1e-7]
        best=min(good,key=lambda c:c['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_dense_signed_trace_joint_learning',new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,best=None if best is None else best['tag'],adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_dense_signed_trace_joint_learning',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next','检查带符号轨迹模型与完整联合策略实测结果，继续修正形状、巡航与全路径目标差距。'],
            check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
