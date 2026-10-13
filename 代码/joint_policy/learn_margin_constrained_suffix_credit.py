"""Matched complete-margin constraints for current joint action-credit learning."""
import argparse
import copy
import multiprocessing as mp
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.linalg import null_space
from scipy.optimize import minimize
from contextlib import nullcontext
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    def threadpool_limits(**kwargs):
        return nullcontext()
import learn_stratified_suffix_joint_credit as suffix
from deep_joint_credit_policy import credit_normalization
from local_suffix_credit_policy import LocalSuffixCreditPolicy, local_basis
from learn_prefix_preserving_suffix_credit import prefix_indices
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--linear-margin-only',action='store_true')
    parser.add_argument('--dense-margin',action='store_true')
    parser.add_argument('--complete-current',action='store_true')
    parser.add_argument('--per-corner-guards',action='store_true')
    args=parser.parse_args()
    if args.per_corner_guards:args.complete_current=True
    if args.dense_margin:args.linear_margin_only=True
    if args.complete_current:args.linear_margin_only=True
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
    source=BASE/('canonical_gate_band_dense_margin_suffix_learning_20260919' if args.complete_current else
                 'canonical_gate_band_numerical_prefix_suffix_credit_20260919')
    sampled=BASE/('canonical_gate_band_complete_current_suffix_credit_20260919' if args.complete_current else
                  'canonical_gate_band_current_protected_joint_credit_20260919/iteration0')
    data=read(source/'results.json')
    baseline=next(c for c in data['candidates'] if c['tag']==(data['best'] if args.complete_current else 'protected_0.125'))
    model=source/('research_candidate.pt' if args.complete_current else 'protected_0.125.pt')
    parent=torch.load(model,map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    queries=sorted(read(sampled/'query_results.json')['queries'],key=lambda q:q['query'])
    amplitude=np.array(read(sampled/'query_plan.json')['amplitude'])
    if args.complete_current:assert read(sampled/'receipt.json')['status']=='completed_all_current_decision_suffix_credit'
    stage=register('canonical_gate_band_per_corner_dense_credit_20260919' if args.per_corner_guards else
                   'canonical_gate_band_complete_decision_joint_learning_20260919' if args.complete_current else
                   'canonical_gate_band_dense_margin_suffix_learning_20260919' if args.dense_margin else
                   'canonical_gate_band_linear_margin_suffix_credit_20260919' if args.linear_margin_only else
                   'canonical_gate_band_margin_constrained_suffix_credit_20260919',
        [__file__,BASE/'local_suffix_credit_policy.py',BASE/'learn_prefix_preserving_suffix_credit.py',
         BASE/'deep_joint_credit_policy.py',BASE/'learn_stratified_suffix_joint_credit.py',
         sampled/'query_results.json',sampled/'query_plan.json',model,source/'results.json',source/'receipt.json']+
         ([sampled/'receipt.json'] if args.complete_current else [sampled/'results.json'])+
         ([BASE/'audit_dense_shape_credit.py',
           BASE/'canonical_gate_band_complete_decision_joint_learning_20260919/results.json',
           BASE/'canonical_gate_band_complete_decision_joint_learning_20260919/receipt.json',
           BASE/'canonical_gate_band_complete_decision_joint_learning_20260919/per_corner_credit_prediction.json'] if args.per_corner_guards else [])+
         ([BASE/'canonical_gate_band_dense_boundary_suffix_credit_20260919/query_results.json',
           BASE/'canonical_gate_band_dense_boundary_suffix_credit_20260919/receipt.json',
           BASE/'canonical_gate_band_linear_margin_suffix_credit_20260919/results.json',
           BASE/'canonical_gate_band_linear_margin_suffix_credit_20260919/receipt.json'] if args.dense_margin else []),
        dict(wall_s=300 if args.complete_current else 240,workers=4,reused_queries=len(queries),reused_branches=6*len(queries),
             new_candidates=2 if args.dense_margin or args.per_corner_guards else 4,paths=20,trust_scales=[.0625,.125],
             margin_target_mm=.01 if not args.linear_margin_only else None,
             active_margin_band_mm=.05,optimizer='analytic convex quadratic plus margin hinge; SLSQP'),
        ('Matched constraint revision at the SAME19.97685deg center and complete638 '
         'decision dataset. The previous update improved reversal/exit but raised '
         'total energy by0.010349; its credit model had predicted a0.010186 increase. '
         'Thus the scalar objective allowed this tradeoff, rather than simply missing '
         'the energy effect. Add separate TRAINING inequalities for each of16 corners '
         'on exit excursion, reversal, normalized energy, peak curvature and negative '
         'apex speed. A metric already worse than retained86cd may not be predicted '
         'to get worse; a better metric may not cross the retained bound. Keep current '
         'dense margin and prefix constraints, objective, basis, optimizer and trust '
         'scales fixed. These linear metric guards remain approximations, especially '
         'max/hinge and apex-index changes; actual per-corner measurements remain '
         'decisive. Reuse all four already measured controls, run only TWO fresh '
         'guarded policies. No new physical-response collection or new model adoption. '
         'The following full-decision description is the shared parent experiment. ' if args.per_corner_guards else '')+
        ('Use the NEW actual complete research center with19.97685deg reversal and '
         '3.691s curved99 cruise, still unadopted. All638 unprotected ordinary '
         'decisions have current complete +/- action consequences. Use UNIT temporal '
         'weights for both shape and corridor-margin response, with the predeclared '
         '108 basis-center subset instead of increasing actor capacity to638 centers. '
         'Match shape-only versus margin-preserving quadratic parameter optimization '
         'at both trust scales, four fresh full20-path policies. Retain numerical '
         'entry-prefix protection and physical limits. Dense finite differences '
         'remove time-subsampling error on these paths, not all approximation or '
         'generalization error. Earlier studies below motivate the method and refer '
         'to their historical starting model, not this new center. ' if args.complete_current else '')+
        ('Keep the sparse SHAPE credit, state basis, prefix constraints, solver and trust '
         'limits exactly fixed. Replace only path6/path9 margin credit with every actual '
         'ordinary decision after the protected prefix, each with UNIT weight. Compare '
         'predicted margins for the already completed sparse-constrained policies before '
         'learning fresh dense-constrained policies. Reuse both shape-only complete '
         'rollouts; exactly TWO new complete20-path policies at the same trust scales. '
         'This isolates temporal sampling of constraint credit; it does not remove '
         'finite-amplitude or simultaneous-update nonlinearity. ' if args.dense_margin else '')+
        ('The preceding augmented-margin solver failed before any new physical rollout. '
         'Reuse its two already solved shape-only policy checkpoints; do not solve or '
         'simulate completed work again. Compare only HARD linear predicted-margin '
         'preservation, with the identical original quadratic shape objective and '
         'optimizer, omitting the added margin-hinge objective. This clarifies the '
         'causal contrast and reduces numerical scaling problems. ' if args.linear_margin_only and not args.dense_margin and not args.complete_current else '')+
        'All four refreshed-credit policy updates failed at a corner where the current '
        'complete segment margin is only0.000418mm.647/648 isolated action branches '
        'completed, so independent feasibility masks miss accumulated constraint drift. '
        'Reuse those exact consequences to estimate full command-segment minimum-margin '
        'action derivatives, then sum weighted derivatives through the learned state '
        'functions. Successful pairs use central differences; one failed side supplies '
        'no efficiency and uses only the successful margin side. Compare the SAME '
        'prefix-nullspace actor, scalar quadratic model, analytic optimizer and action '
        'trust region with versus without predicted margin preservation at currently '
        'near-boundary paths. '+('The constrained arm additionally rewards0.01mm interior '
        'margin through a smooth squared hinge. ' if not args.linear_margin_only else
        'No added margin-hinge reward is used in this comparison. ')+
        'These are TRAINING constraints/objectives; '
        'the actual allowed distance stays1.25mm with identical full physical checks. '
        'Compare both arms at trust scales0.0625/0.125. No new suffix collection in this optimizer, '
        'runtime rule, extra decision or claim that predicted margins certify safety. '
        'Research selection requires actual primary preservation and nonworse aggregate '
        'reversal/exit plus improved absolute shape objective. If the model misses '
        'the actual margins, repair constraint credit rather than increasing training.')
    started=time.perf_counter();pool=None;outcomes=[]
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next','联合学习纳入完整欧氏距离余量信用，复用既有响应并做匹配闭环对照。'],
        check=True,stdout=subprocess.DEVNULL)
    try:
        support,protected=[],[]
        for r in baseline['records']:
            with np.load(source/r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
            support.append(obs);protected.append(obs[prefix_indices(ref,rows,obs)])
        support,protected=np.vstack(support),np.vstack(protected)
        qobs=np.array([q['observation'] for q in queries],np.float32)
        mean,std=credit_normalization(support)
        basis_obs=qobs[np.asarray(read(sampled/'query_plan.json')['basis_query_ids'],int)] if args.complete_current else qobs
        centers,radius=local_basis(basis_obs,mean,std)
        p=LocalSuffixCreditPolicy(parent,mean,std,centers,radius,precision='float64')
        fq,fa,fp=[p.features(torch.tensor(x)).detach().numpy() for x in (qobs,support,protected)]
        transforms=[null_space(fp[:,:,h],rcond=1e-12) for h in range(3)]
        dimensions=[t.shape[1] for t in transforms];n=sum(dimensions)
        qa=np.zeros((len(queries),3,n));aa=np.zeros((len(support),3,n));offset=0
        for head,t in enumerate(transforms):
            k=t.shape[1];qa[:,head,offset:offset+k]=fq[:,:,head]@t*p.output_scale[head].item()
            aa[:,head,offset:offset+k]=fa[:,:,head]@t*p.output_scale[head].item();offset+=k
        weights=np.array([q['stratum_size'] for q in queries])[:,None]
        gradients=np.array([q['gradient'] for q in queries]);curvature=np.zeros_like(gradients)
        old={r['path']:r for r in baseline['records']}
        paths=[r['path'] for r in baseline['records']]
        margins=np.array([r['corridor']['margin_lower_mm'] for r in baseline['records']])
        margin_jac=np.zeros((len(paths),n));modes=[]
        for i,q in enumerate(queries):
            dm=np.zeros(3);b=old[q['path']]['corridor']['margin_lower_mm']
            for h in range(3):
                lo,hi=q['records'][2*h:2*h+2]
                if lo['success'] and hi['success']:
                    curvature[i,h]=max(0.,(hi['absolute_loss']+lo['absolute_loss']-2*q['baseline_absolute_loss'])/amplitude[h]**2)
                    dm[h]=(hi['corridor']['margin_lower_mm']-lo['corridor']['margin_lower_mm'])/(2*amplitude[h])
                elif hi['success']:
                    dm[h]=(hi['corridor']['margin_lower_mm']-b)/amplitude[h]
                    modes.append(dict(query=q['query'],head=h,mode='positive_complete_margin_side'))
                elif lo['success']:
                    dm[h]=(b-lo['corridor']['margin_lower_mm'])/amplitude[h]
                    modes.append(dict(query=q['query'],head=h,mode='negative_complete_margin_side'))
            margin_jac[paths.index(q['path'])]+=q['stratum_size']*dm@qa[i]
        if args.dense_margin:
            dense_source=BASE/'canonical_gate_band_dense_boundary_suffix_credit_20260919'
            assert read(dense_source/'receipt.json')['status'].startswith('completed')
            dense=read(dense_source/'query_results.json')['queries']
            sparse_margin_jac=margin_jac.copy()
            dfeatures=p.features(torch.tensor(np.array([q['observation'] for q in dense],np.float32))).detach().numpy()
            da=np.zeros((len(dense),3,n));offset=0
            for head,t in enumerate(transforms):
                k=t.shape[1];da[:,head,offset:offset+k]=dfeatures[:,:,head]@t*p.output_scale[head].item();offset+=k
            for path in (6,9):margin_jac[paths.index(path)]=0.
            for i,q in enumerate(dense):
                dm=np.zeros(3);b=old[q['path']]['corridor']['margin_lower_mm']
                for h in range(3):
                    lo,hi=q['records'][2*h:2*h+2]
                    if lo['success'] and hi['success']:
                        dm[h]=(hi['corridor']['margin_lower_mm']-lo['corridor']['margin_lower_mm'])/(2*amplitude[h])
                    elif hi['success']:dm[h]=(hi['corridor']['margin_lower_mm']-b)/amplitude[h]
                    elif lo['success']:dm[h]=(b-lo['corridor']['margin_lower_mm'])/amplitude[h]
                margin_jac[paths.index(q['path'])]+=dm@da[i]
            comparisons=[]
            prior=BASE/'canonical_gate_band_linear_margin_suffix_credit_20260919'
            prior_results=read(prior/'results.json')['candidates']
            for scale in (.0625,.125):
                tag=f'margin_constrained_{scale}'
                md=torch.load(prior/(tag+'.pt'),map_location='cpu',weights_only=False)
                zz=np.concatenate([t.T@md['coefficients'][:,h] for h,t in enumerate(transforms)])
                actual=next(c for c in prior_results if c['tag']==tag)
                for path in (6,9):
                    i=paths.index(path)
                    comparisons.append(dict(tag=tag,path=path,baseline_margin_mm=float(margins[i]),
                        sparse_predicted_margin_mm=float(margins[i]+sparse_margin_jac[i]@zz),
                        dense_predicted_margin_mm=float(margins[i]+margin_jac[i]@zz),
                        actual_margin_mm=next(r for r in actual['records'] if r['path']==path)['corridor']['margin_lower_mm']))
            write_json(stage/'prior_policy_margin_prediction.json',dict(comparisons=comparisons,dense_queries=len(dense)))
            print('DENSE_MARGIN_PREDICTION',comparisons,flush=True)
        limits=np.array([.003,.025,.025]);normalization=float((abs(gradients)*weights*limits).sum())
        a=qa.reshape(-1,n);w=np.broadcast_to(weights,gradients.shape).ravel()
        linear=a.T@(w*gradients.ravel())/normalization
        matrix=(a.T*(w*curvature.ravel()))@a/normalization
        anchors=(aa/limits[None,:,None]).reshape(-1,n)
        matrix+=.04*(anchors.T@anchors)/len(anchors)+np.eye(n)*1e-10
        active=np.flatnonzero(margins<.05);mj=margin_jac[active];mb=margins[active]
        # Scale latent coordinates only for the numerical solve; runtime uses coefficients.
        diagonal=np.maximum(np.sqrt(np.diag(matrix)),1e-4)
        ls=linear/diagonal;hs=matrix/diagonal[:,None]/diagonal[None,:]
        ast=anchors/diagonal;ms=mj/diagonal
        if args.per_corner_guards:
            from audit_dense_shape_credit import shape_derivative,KEYS
            corner_paths=[r['path'] for r in baseline['records'] if 'shape' in r]
            old_shapes={r['path']:r['baseline_shape'] for r in retained if 'baseline_shape' in r}
            signs=np.array([1.,1.,1.,1.,-1.]);units=np.array([.01,1.,.01,.1,2.])
            sj=np.zeros((len(corner_paths)*len(KEYS),n));sb=[];sl=[]
            for path in corner_paths:
                current=np.array([old[path]['shape'][key] for key in KEYS])*signs
                reference=np.array([old_shapes[path][key] for key in KEYS])*signs
                sb.extend(current);sl.extend(np.maximum(current,reference))
            for i,q in enumerate(queries):
                if q['path'] not in corner_paths:continue
                k=corner_paths.index(q['path'])*len(KEYS)
                sj[k:k+len(KEYS)]+=shape_derivative(q,old[q['path']],amplitude)*signs[:,None]@qa[i]
            shape_margin=np.array(sl)-np.array(sb)
            shape_units=np.tile(units,len(corner_paths));ss=sj/diagonal
            np.savez_compressed(stage/'per_corner_constraints.npz',margin=shape_margin,jacobian=sj,
                paths=corner_paths,metric_names=np.array(KEYS),baseline_signed_values=sb,maximum_signed_values=sl)
            control=BASE/'canonical_gate_band_complete_decision_joint_learning_20260919'
            for r in read(control/'results.json')['candidates']:
                copied=copy.deepcopy(r);copied['reused_complete_evaluation']=True
                for record in copied['records']:record['trace']=str(control/record['trace'])
                outcomes.append(copied)
        np.savez_compressed(stage/'response_model.npz',linear=linear,hessian=matrix,margin=margins,
                            margin_jacobian=margin_jac,paths=paths,protected_free_dimensions=dimensions)
        jobs=[];audits=[]
        for arm in (('per_corner_constrained',) if args.per_corner_guards else ('shape_only','margin_constrained')):
            for scale in (.0625,.125):
                tag=f'{arm}_{scale}'
                if args.linear_margin_only and arm=='shape_only' and not args.complete_current:
                    previous=BASE/'canonical_gate_band_margin_constrained_suffix_credit_20260919'
                    md=torch.load(previous/(tag+'.pt'),map_location='cpu',weights_only=False)
                    torch.save(md,stage/(tag+'.pt'));jobs.append((tag,md))
                    prior=next(a for a in read(previous/'optimizer.json')['arms'] if a['tag']==tag)
                    audits.append(dict(prior,reused_solved_parameters=True))
                    if args.dense_margin:
                        known=BASE/'canonical_gate_band_linear_margin_suffix_credit_20260919'
                        record=copy.deepcopy(next(c for c in read(known/'results.json')['candidates'] if c['tag']==tag))
                        record['reused_complete_evaluation']=True
                        for r in record['records']:r['trace']=str(known/r['trace'])
                        outcomes.append(record)
                        jobs.pop()
                    continue
                def objective(y):
                    value=float(ls@y+.5*y@hs@y);grad=ls+hs@y
                    if arm=='margin_constrained' and not args.linear_margin_only:
                        defect=np.maximum((.01-mb-ms@y)/.01,0.)
                        value+=float(defect@defect);grad-=2*ms.T@defect/.01
                    return value,grad
                constraints=[dict(type='ineq',fun=lambda y:np.r_[scale-ast@y,scale+ast@y],
                                  jac=lambda y:np.vstack([-ast,ast]))]
                if arm in ('margin_constrained','per_corner_constrained'):
                    constraints.append(dict(type='ineq',fun=lambda y:(mb+ms@y-np.minimum(mb,.02))/.01,
                                            jac=lambda y:ms/.01))
                if args.per_corner_guards:
                    constraints.append(dict(type='ineq',fun=lambda y:(shape_margin-ss@y)/shape_units,
                        jac=lambda y:-ss/shape_units[:,None]))
                with threadpool_limits(limits=1):
                    solution=minimize(objective,np.zeros(n),jac=True,method='SLSQP',constraints=constraints,
                        options=dict(maxiter=200,ftol=1e-9))
                z=solution.x/diagonal;maximum=float(abs(anchors@z).max())
                assert solution.success and maximum<=scale+1e-7,solution.message
                coeff=[];offset=0
                for t in transforms:
                    k=t.shape[1];coeff.append(t@z[offset:offset+k]);offset+=k
                coefficients=np.column_stack(coeff)
                with torch.no_grad():p.coefficients.copy_(torch.tensor(coefficients))
                error=max(float(abs(p.act(x,True)[0]-p.base.act(x,True)[0]).max()) for x in protected)
                assert error<=1e-12,error
                md=dict(architecture='local_suffix_credit',parent=parent,mean=mean,std=std,centers=centers,radius=radius,
                    precision='float64',coefficients=coefficients,step_scale=1.,environment_architecture='gate_band_canonical_kernel',
                    learning_method='complete_margin_credit_'+arm)
                torch.save(md,stage/(tag+'.pt'));jobs.append((tag,md))
                predicted=margins+margin_jac@z
                audit=dict(tag=tag,solver_message=solution.message,iterations=int(solution.nit),
                    maximum_normalized_action_change=maximum,maximum_protected_latent_difference=error,
                    predicted_shape_change=float(normalization*(linear@z+.5*z@matrix@z)),
                    predicted_margins=[dict(path=path,margin_mm=float(m)) for path,m in zip(paths,predicted)])
                audits.append(audit);write_json(stage/'optimizer.json',dict(arms=audits,margin_modes=modes))
                print('MARGIN_CREDIT_PROPOSAL',tag,audit['predicted_shape_change'],[(paths[i],float(predicted[i])) for i in active],flush=True)
        pool=mp.get_context('spawn').Pool(4,suffix.initialize,(stage,parent,baseline,retained,source,started+(290 if args.complete_current else 230)))
        for result in pool.imap_unordered(suffix.evaluate_network,jobs):
            result['minimum_margin_mm']=min(r['corridor']['margin_lower_mm'] for r in result['records'])
            outcomes.append(result);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            print('MARGIN_CREDIT_ACTUAL',result['tag'],result['absolute_objective'],result['summary'],
                result['promotion'],'margin',result['minimum_margin_mm'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if not c.get('reused_complete_evaluation',False) and c['primary_preserved'] and c['absolute_objective']<queries[0]['baseline_absolute_loss']-1e-5
            and c['summary']['reverse_sum_deg']<=baseline['summary']['reverse_sum_deg']+1e-6
            and c['summary']['exit_sum_mm']<=baseline['summary']['exit_sum_mm']+1e-7]
        best=min(good,key=lambda c:c['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_margin_constrained_suffix_comparison',
            new_candidates=sum(not c.get('reused_complete_evaluation',False) for c in outcomes),
            reused_candidates=sum(c.get('reused_complete_evaluation',False) for c in outcomes),
            wall_s=time.perf_counter()-started,best=None if best is None else best['tag'],adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_margin_constrained_suffix_comparison',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next','核实余量信用是否改善真实完整约束与形状，再决定下一次联合学习。'],
            check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
