"""PPO-style clipped residual credit with an integrated linear action control variate."""
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
from observable_memory_joint_policy import actor_features
from joint_teacher_state_policy import OUTPUT_SCALE
from head_counterfactual_rollouts import cost
from train_factorial_joint_ppo import worker_initialize,worker_evaluate,metric_functions,guards
from train_factorial_joint_ppo import shape_event_learning_guards,intrinsic_guards,finish,parameters_in_bounds
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def constrained_proposals(source,deterministic,deadline):
    from scipy.optimize import minimize
    previous=read(source/'control_variate_update_audit.json')
    target=np.array(next(c['coefficients'] for c in previous['checkpoints'] if c['tag']=='dense_action_time_cv'))
    with np.load(deterministic/'geometry_complete_gradient.npz') as z:
        # Select the original constant-context rows from each of the three feature/history blocks.
        ids=np.concatenate([np.arange(18)+54*i for i in range(3)])
        jac=z['constraint_jacobian'][:,ids];margins=z['baseline_margins'];parent=z['parent_matrix'][:,ids]
        gradient=z['loss_gradient'][ids]
    scale=.001
    matrix=np.vstack([-jac,parent,-parent,gradient[None,:]])*scale
    rhs=np.r_[margins,np.full(2*len(parent),.15),0.]
    norms=np.maximum(np.linalg.norm(matrix,axis=1),1e-12)
    matrix=matrix/norms[:,None];rhs=rhs/norms
    result=[];reports=[]
    action=parent.T@parent/len(parent)
    action/=max(np.trace(action)/len(target),1e-30)
    for mode,hessian in (('coefficient_projection',np.eye(len(target))),
                          ('action_projection',action+.001*np.eye(len(target)))):
        destination=target/scale
        def objective(v):
            d=v-destination
            return .5*d@hessian@d,hessian@d
        def budget(v):
            if time.perf_counter()>deadline:raise TimeoutError('Guarded projection budget')
        solved=minimize(objective,np.zeros_like(target),jac=True,method='SLSQP',
            bounds=[(-10.,10.)]*len(target),constraints=[dict(type='ineq',
                fun=lambda v:rhs-matrix@v,jac=lambda v:-matrix)],callback=budget,
            options=dict(maxiter=160,ftol=1e-11))
        u=scale*solved.x
        report=dict(mode=mode,success=bool(solved.success),message=str(solved.message),iterations=int(solved.nit),
            projected_norm=float(np.linalg.norm(u)),distance_from_proposal=float(np.linalg.norm(u-target)),
            predicted_guard_min=float(np.min(margins+jac@u)),predicted_cost_change=float(gradient@u),
            maximum_standardized_action_shift=float(np.max(abs(parent@u))),all_guard_rows_retained=True)
        if report['predicted_guard_min']>=-1e-8 and report['predicted_cost_change']<=1e-8 and np.linalg.norm(u)>1e-10:
            result.append(dict(tag=mode,coefficients=u.tolist()))
        reports.append(report);print('GUARDED_PPO_PROPOSAL',report,flush=True)
    return result,dict(logs=reports,proposal_source_sha256=sha(source/'control_variate_update_audit.json'),
        original_proposal=target.tolist(),new_ppo_optimizer_steps=0,
        scope='Deterministic constrained projection of a saved modified-PPO proposal; not new PPO training.')


def reference_credit(episodes):
    credit={}
    for e in episodes:
        by={(b['decision'],b['head'],b['radius'],b['sign']):b for b in e['branches']}
        for s in e['samples']:
            gradient=[]
            for h in range(3):
                lo,hi=[by[s['decision'],h,.1,sign] for sign in (-1,1)]
                gradient.append(-(hi['score']-lo['score'])/(2*hi['epsilon']))
            credit[e['path'],s['decision']]=np.array(gradient)
    return credit


def prepare(md,episodes,credit):
    tables=[];dense=[];exact=0
    for e in episodes:
        with np.load(e['tables'][0]['cells'][0]['record']['trace']) as z:observations=z['observations'].copy()
        memory=np.zeros((2,len(md['centers']),3));previous=np.zeros_like(memory[0])
        alpha=np.asarray(md['memory_decays'])[:,None,None]
        by={t['sample']['decision']:t for t in e['tables']}
        full=np.zeros_like(md['coefficients'])
        for j,x in enumerate(observations):
            # This actor's metric has zero innovation weights; verify exactness on EVERY saved query.
            f=actor_features(md,x,np.zeros(3));memory=alpha*memory+(1.-alpha)*previous;previous=f.copy()
            design=np.concatenate([f,*memory],axis=0)*OUTPUT_SCALE
            g=credit.get((e['path'],j),np.zeros(3))
            full+=design*g
            if j in by:
                t=copy.deepcopy(by[j]);assert np.array_equal(design,np.array(t['sample']['design']))
                exact+=1;t['reference_action_gradient']=g.tolist();tables.append(t)
        dense.append(full)
    assert exact==24
    return tables,np.sum(dense,axis=0),dense


def optimize(tables,dense,scale):
    design=torch.tensor(np.array([t['sample']['design'] for t in tables]),dtype=torch.float64)
    sigma=torch.tensor(np.array([t['sample']['sigma'] for t in tables]),dtype=torch.float64)
    noise=torch.tensor(np.array([[c['noise'] for c in t['cells']] for t in tables]),dtype=torch.float64)
    q=torch.tensor(np.array([[c['total_return'] for c in t['cells']] for t in tables]),dtype=torch.float64)
    g=torch.tensor(np.array([t['reference_action_gradient'] for t in tables]),dtype=torch.float64)
    w=torch.tensor([t['sample']['query_weight'] for t in tables],dtype=torch.float64)
    masks=torch.arange(8)^7
    proxy=(noise*sigma[:,None,:]*g[:,None,:]).sum(-1)
    residual=q-proxy
    advantage=(residual-residual[:,masks])/scale
    sparse=(design*g[:,None,:]*w[:,None,None]).sum(0)
    baselines={'sparse_action_cv':sparse,'dense_action_time_cv':torch.tensor(dense,dtype=torch.float64)}
    def loss(u,integrated,adv=advantage):
        shift=(design*u).sum(1)/sigma
        ratio=(noise*shift[:,None,:]-.5*shift[:,None,:].square()).sum(-1).exp()
        surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean(1)
        return -(surrogate*w).sum()/w.sum()-(integrated*u).sum()/(w.sum()*scale),.5*shift.square().sum(-1).mean()
    # If Q is exactly the fixed linear proxy, residual credit vanishes for ANY action draws.
    zero=torch.zeros(design.shape[1:],dtype=torch.float64,requires_grad=True)
    synthetic=torch.autograd.grad(loss(zero,baselines['dense_action_time_cv'],torch.zeros_like(advantage))[0],zero)[0]
    expected=-baselines['dense_action_time_cv']/(w.sum()*scale)
    assert torch.allclose(synthetic,expected,rtol=1e-13,atol=1e-13)
    logs=[];checkpoints=[];initial=[]
    for arm,integrated in baselines.items():
        u=torch.nn.Parameter(torch.zeros(design.shape[1:],dtype=torch.float64))
        gradient=torch.autograd.grad(loss(u,integrated)[0],u)[0].detach().numpy()
        initial.append(dict(arm=arm,gradient=gradient.ravel().tolist()))
        optimizer=torch.optim.Adam([u],lr=2e-4)
        for epoch in range(1,7):
            objective,kl=loss(u,integrated)
            if float(kl.detach())>.01:break
            optimizer.zero_grad(set_to_none=True);objective.backward()
            torch.nn.utils.clip_grad_norm_([u],.5);optimizer.step()
            logs.append(dict(arm=arm,epoch=epoch,loss=float(objective.detach()),kl_before=float(kl.detach())))
        checkpoints.append(dict(tag=arm,coefficients=u.detach().numpy().ravel().tolist()))
    return checkpoints,dict(logs=logs,initial_gradients=initial,linear_proxy_gradient_exact=True,
        global_advantage_scale=scale,dense_baseline_gradient=dense.ravel().tolist(),
        sparse_baseline_gradient=sparse.numpy().ravel().tolist(),proxy_is_frozen_and_independent_of_current_draw=True,
        scope='Modified clipped PPO surrogate with analytic linear-control-variate term; '
        'not standard PPO or proof of an unbiased deterministic/full-trajectory gradient.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--guarded-projection',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+160.
    source=BASE/'canonical_gate_band_factorial_joint_ppo_20260920'
    deterministic=BASE/'canonical_gate_band_deterministic_full_support_credit_20260920'
    parent_file=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert not read(source/'receipt.json')['research_acceptable']
    md=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    md['learning_method']='clipped_ppo_residual_credit_with_linear_action_control_variate'
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    episodes=read(source/'factorial_episodes.json')['episodes']
    episodes.sort(key=lambda e:(e['path'],e['seed']))
    credit=reference_credit(read(deterministic/'deterministic_episodes.json')['episodes'])
    scale=read(source/'ppo_update_audit.json')['global_advantage_scale']
    previous=BASE/'canonical_gate_band_reference_control_variate_ppo_20260920'
    stage=register('canonical_gate_band_guarded_control_variate_proposal_20260920' if args.guarded_projection else
        'canonical_gate_band_reference_control_variate_ppo_20260920',
        [__file__,parent_file]+[source/f for f in ('receipt.json','initial_policy.pt','baseline.json',
        'support_audit.json','factorial_episodes.json','ppo_update_audit.json','deterministic_alignment_audit.json')]+
        [deterministic/'receipt.json',deterministic/'deterministic_episodes.json']+
        [BASE/f for f in ('train_factorial_joint_ppo.py','observable_memory_joint_policy.py',
        'head_counterfactual_rollouts.py','learn_early_memory_joint_policy.py','learn_direct_observable_joint_policy.py')]+
        ([previous/'receipt.json',previous/'control_variate_update_audit.json',previous/'state_joint/results.json',
          deterministic/'geometry_complete_gradient.npz'] if args.guarded_projection else []),
        dict(wall_s=180,workers=4,new_episodes=0,new_suffixes=0,reused_factorial_cells=192,
            parameters=54,epochs=6,learning_rate=2e-4,clip=.2,kl_stop=.01,gradient_clip=.5,
            deterministic_new_policies=2,global_advantage_scale=scale,
            actual_new_ppo_epochs=0 if args.guarded_projection else 6),
        ('CURRENT STAGE OVERRIDE: reuse the completed dense-control-variate PPO proposal, '
         'without repeating its updates or data. The complete deterministic203-guard model '
         'predicts54 violated channels and55 actually fail. Project the SAVED proposal onto '
         'all original linearized guards plus nonincreasing predicted deterministic cost, '
         'same54 parameters and standardized-action trust0.15. Compare coefficient-distance '
         'and mean-action-distance quadratic metrics; both start at feasible zero. Keep '
         'every constraint row, record solver status, then execute at most2 nonzero proposals '
         'on all7 affected routes. This stage is DETERMINISTIC CONSTRAINED PARAMETER '
         'OPTIMIZATION of a PPO proposal, not new PPO interaction or training. No claim '
         'linearized safety guarantees real trajectories. If rejected, audit model error '
         'and zero-step obstruction instead of unmodified extension. The text below is '
         'the reused proposal algorithm context only. ' if args.guarded_projection else '')+
        'Saved complete deterministic credit predicts the factorial PPO degradation accurately. '
        'Use its fixed action derivatives as a linear Q control variate. At each old query '
        'subtract proxy action/complement differences from measured complete Q differences '
        'and integrate the proxy analytically under the Gaussian mean update. Compare sparse '
        'inclusion-weighted proxy integration with dense integration over ALL recorded supported '
        'decisions of the SAME8 behavior episodes. Same residual samples,54-weight actor,Adam '
        'and6 updates; original joint-complement PPO is reused control. Exact saved-query feature '
        'identity and linear-proxy gradient test required. No new collection,labels,or value fit. '
        'This is a modified PPO-style clipped residual-credit algorithm, not standard PPO; '
        'reference derivatives from deterministic states need not equal stochastic Q derivatives. '
        'At the old policy the conditional linear baseline cancels in expectation; finite batch, '
        'query selection,transition nonsmoothness and deterministic deployment gaps remain. '
        'The reference path/decision mapping is training-only; deployed actor uses observations '
        'and memory only. All original203 guards and protected models unchanged.180s; if '
        'neither passes,inspect residual-credit stability instead of adding identical epochs.')
    folder=stage/'state_joint';folder.mkdir()
    torch.save(md,stage/'initial_policy.pt');write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    def entry(records,u,tag):
        original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,
            score=sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected)),
            measured_inequalities=original.tolist(),augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=[] if tag=='reused_parent' else paths,exact_zero_support_paths=paths if tag=='reused_parent' else [])
    zero=entry(copy.deepcopy(selected),np.zeros(md['coefficients'].size),'reused_parent');history=[zero]
    md['reused_candidates']=1
    def best():return min((c for c in history if parameters_in_bounds(c['coefficients'],-1.,1.) and
        min(c['augmented_measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
    pool=None
    try:
        if args.guarded_projection:
            md['learning_method']='deterministic_guard_projection_of_control_variate_ppo_proposal'
            torch.save(md,stage/'initial_policy.pt')
            checkpoints,audit=constrained_proposals(previous,deterministic,deadline)
            per_episode=[]
        else:
            tables,dense,per_episode=prepare(md,episodes,credit)
            checkpoints,audit=optimize(tables,dense,scale)
        write_json(stage/'control_variate_update_audit.json',dict(audit,checkpoints=checkpoints,
            query_features_bitwise=None if args.guarded_projection else True,
            per_episode_dense_proxy=[g.ravel().tolist() for g in per_episode]))
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for cp in checkpoints:
            tag=cp['tag'];u=np.array(cp['coefficients']);actor=copy.deepcopy(md);actor['coefficients']=u.reshape(-1,3)
            torch.save(actor,stage/(tag+'.pt'))
            measured=pool.map(worker_evaluate,[(p,u,tag) for p in paths]);by={r['path']:r for r in measured}
            c=entry([by[r['path']] for r in selected],u,tag);history.append(c)
            write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
            print('CONTROL_VARIATE_PPO',tag,c['score'],min(c['augmented_measured_inequalities']),flush=True)
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,audit['logs'],choice,parent_file,started)
        receipt(stage,dict(status='completed_guarded_control_variate_proposal' if args.guarded_projection else
            'completed_reference_control_variate_ppo',new_episodes=0,new_suffixes=0,
            actor_optimizer_steps=0 if args.guarded_projection else len(audit['logs']),
            constrained_optimizations=len(audit['logs']) if args.guarded_projection else 0,
            critic_optimizer_steps=0,new_deterministic_policies=len(history)-1,
            selected=best()['tag'],research_acceptable=(stage/'research_candidate.pt').exists(),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_guarded_control_variate_proposal' if args.guarded_projection else
            'failed_reference_control_variate_ppo',error=str(exc),
            new_deterministic_policies=len(history)-1,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise


if __name__=='__main__':
    mp.freeze_support();main()
