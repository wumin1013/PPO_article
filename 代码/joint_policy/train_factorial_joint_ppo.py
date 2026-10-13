"""Matched factorial Gaussian-action PPO: head ratios versus joint ratios."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from head_counterfactual_rollouts import action_distribution,cost
from observable_memory_joint_policy import ObservableMemoryJointPolicy
from gate_band_state import environment_factory
from analyze_nonlinear_joint_response import measure
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from duration_joint_metrics import guards
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_direct_observable_joint_policy import finish
from learn_complete_joint_teachers import parameters_in_bounds
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


WORK={}


def initialize(md,records,retained,folder,deadline):
    torch.set_num_threads(1)
    WORK.update(md=md,records={r['path']:r for r in records},
        retained={r['path']:r['baseline_shape'] for r in retained if 'baseline_shape' in r},folder=folder,deadline=deadline)


def collect(e):
    path=e['path'];baseline=WORK['records'][path]
    with np.load(e['record']['trace']) as z:
        ref,full,held,observed,noise=[z[k].copy() for k in ('reference','rows','actions','observations','noise')]
    policy=ObservableMemoryJointPolicy(copy.deepcopy(WORK['md']));env=policy.bind(environment_factory(ref))
    samples={s['decision']:s for s in e['samples']};snapshots={}
    while not env.done and env.decisions<=max(samples):
        if time.perf_counter()>WORK['deadline']:raise TimeoutError('Factorial prefix budget')
        j=env.decisions;x=env.observation();mean,design,sigma=action_distribution(policy,x)
        assert np.array_equal(x,observed[j])
        if j in samples:
            s=samples[j]
            assert np.array_equal(mean,np.array(s['mean'])) and np.array_equal(design,np.array(s['design']))
            assert np.array_equal(sigma,np.array(s['sigma']))
            snapshots[j]=copy.deepcopy((env,policy))
        _,_,_,info=env.step(mean.astype(float)+sigma*noise[j])
        assert np.array_equal(info['rows'],full[j*20:j*20+len(info['rows'])])
        assert np.array_equal(info['held_action'],held[j])
    by={(b['decision'],b['head'],b['repeat']):b for b in e['counterfactuals']}
    tables=[]
    for j,s in sorted(samples.items()):
        alternatives=np.array([by[j,h,0]['replacement_noise'] for h in range(3)])
        cells=[]
        for mask in range(8):
            bits=np.array([(mask>>h)&1 for h in range(3)],dtype=bool)
            epsilon=np.where(bits,alternatives,noise[j])
            if mask==0:
                record=e['record'];value=e['observed_return'];reused=True
            elif mask in (1,2,4):
                b=by[j,mask.bit_length()-1,0];record=b['record'];value=b['total_return'];reused=True
            else:
                branch_env,branch_policy=copy.deepcopy(snapshots[j]);chunks=[];actions=[];observations=[]
                while not branch_env.done:
                    if time.perf_counter()>WORK['deadline']:raise TimeoutError('Factorial suffix budget')
                    k=branch_env.decisions;x=branch_env.observation();mean,_,sigma=action_distribution(branch_policy,x)
                    _,_,_,info=branch_env.step(mean.astype(float)+sigma*(epsilon if k==j else noise[k]))
                    chunks.append(info['rows']);actions.append(info['held_action']);observations.append(x)
                rows=np.vstack([full[:j*20],*chunks])
                file=WORK['folder']/f'p{path}_s{e["seed"]}_d{j}_mask{mask}.npz'
                np.savez_compressed(file,reference=ref,rows=rows,actions=np.vstack([held[:j],actions]),
                    observations=np.vstack([observed[:j],observations]),current_noise=epsilon,mask=mask)
                record=measure(ref,rows,branch_env);record.update(path=path,trace=str(file),curved_path=True)
                value=-cost(record,baseline,WORK['retained'][path]);reused=False
            cells.append(dict(mask=mask,noise=epsilon.tolist(),total_return=value,record=record,reused=reused))
        tables.append(dict(sample=s,cells=cells,path=path,seed=e['seed']))
    result=dict(path=path,seed=e['seed'],tables=tables,original_prefix_bitwise=True)
    write_json(WORK['folder']/f'p{path}_s{e["seed"]}_factorial.json',result)
    print('FACTORIAL_EPISODE',path,e['seed'],'new_suffixes',4*len(tables),flush=True)
    return result


def updates(tables,scale):
    design=torch.tensor(np.array([t['sample']['design'] for t in tables]),dtype=torch.float64)
    sigma=torch.tensor(np.array([t['sample']['sigma'] for t in tables]),dtype=torch.float64)
    noise=torch.tensor(np.array([[c['noise'] for c in t['cells']] for t in tables]),dtype=torch.float64)
    values=torch.tensor(np.array([[c['total_return'] for c in t['cells']] for t in tables]),dtype=torch.float64)
    inclusion=torch.tensor([t['sample']['query_weight'] for t in tables],dtype=torch.float64)
    inclusion/=inclusion.mean()
    masks=torch.arange(8)
    joint=(values-values[:,masks^7])*inclusion[:,None]/scale
    head=torch.stack([values-values[:,masks^(1<<h)] for h in range(3)],dim=-1)*inclusion[:,None,None]/scale
    def loss(weights,arm):
        shift=(design*weights).sum(1)/sigma
        logratio=noise*shift[:,None,:]-.5*shift[:,None,:].square()
        if arm=='head_conditional':
            ratio=logratio.exp();surrogate=torch.minimum(ratio*head,ratio.clamp(.8,1.2)*head).sum(-1)
        else:
            ratio=logratio.sum(-1).exp();surrogate=torch.minimum(ratio*joint,ratio.clamp(.8,1.2)*joint)
        return -surrogate.mean(),.5*shift.square().sum(-1).mean()
    zero=torch.zeros(design.shape[1:],dtype=torch.float64,requires_grad=True)
    gradients=[torch.autograd.grad(loss(zero,arm)[0],zero)[0] for arm in ('head_conditional','joint_complement')]
    error=float(torch.max(abs(gradients[0]-gradients[1])))
    assert torch.allclose(gradients[0],gradients[1],rtol=1e-10,atol=1e-12),('matched_initial_gradient',error)
    checkpoints,logs=[],[]
    for arm in ('head_conditional','joint_complement'):
        weights=torch.nn.Parameter(torch.zeros(design.shape[1:],dtype=torch.float64))
        optimizer=torch.optim.Adam([weights],lr=2e-4)
        for epoch in range(1,7):
            objective,kl=loss(weights,arm)
            if float(kl.detach())>.01:break
            optimizer.zero_grad(set_to_none=True);objective.backward()
            torch.nn.utils.clip_grad_norm_([weights],.5);optimizer.step()
            logs.append(dict(arm=arm,epoch=epoch,loss=float(objective.detach()),kl_before=float(kl.detach())))
        checkpoints.append(dict(tag=arm,coefficients=weights.detach().numpy().ravel().tolist()))
    return checkpoints,dict(logs=logs,initial_gradient_max_error=error,initial_gradient_matched=True,
        query_states=len(tables),factorial_cells=8*len(tables),cells_are_not_independent_episodes=True,
        joint_baseline_uses_independent_complement_actions=True,head_baseline_uses_single_head_flip=True)


def interaction_audit(tables):
    """Walsh effects describe these paired draws, not independent population estimates."""
    rows=[]
    for t in tables:
        values=np.array([c['total_return'] for c in t['cells']])
        coefficients={str(mask):float(np.mean(values*np.array([
            (-1.)**((cell&mask).bit_count()) for cell in range(8)]))) for mask in range(1,8)}
        energy={str(order):sum(v*v for k,v in coefficients.items() if int(k).bit_count()==order)
                for order in (1,2,3)}
        total=sum(energy.values())
        rows.append(dict(path=t['path'],seed=t['seed'],decision=t['sample']['decision'],
            coefficients=coefficients,energy=energy,
            interaction_fraction=(energy['2']+energy['3'])/total if total else 0.))
    return dict(rows=rows,scope='Finite paired-action draws at 24 correlated training query states; '
        'not population significance, estimator bias, or a universal root-cause claim.')


def main():
    torch.set_num_threads(1)
    started=time.perf_counter()
    deadline=started+280.
    source=BASE/'canonical_gate_band_stratified_head_credit_ppo_update_20260920'
    parent_file=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    replay_file=BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920/affected_replay.npz'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert read(source/'receipt.json')['status'].startswith('completed')
    md=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    assert not np.any(md['coefficients']) and md['coefficients'].size==54
    md['learning_method']='balanced_factorial_clipped_ppo_head_vs_joint'
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    episodes=read(source/'on_policy_episodes.json')['episodes']
    episodes.sort(key=lambda e:(e['path'],e['seed']))
    assert len(episodes)==8 and sum(len(e['samples']) for e in episodes)==24
    with np.load(replay_file) as z:replay={k:z[k].copy() for k in z.files}
    identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    assert all(identity.unchanged(p,np.zeros(md['coefficients'].size)) for p in paths)
    scale=read(source/'ppo_update_audit.json')['global_advantage_scale']
    stage=register('canonical_gate_band_factorial_joint_ppo_20260920',
        [__file__,parent_file,replay_file]+[source/f for f in ('receipt.json','initial_policy.pt',
        'baseline.json','support_audit.json','on_policy_episodes.json','ppo_update_audit.json')]+
        [BASE/f for f in ('head_counterfactual_rollouts.py','observable_memory_joint_policy.py',
        'learn_early_memory_joint_policy.py','learn_direct_observable_joint_policy.py',
        'duration_joint_metrics.py','shape_event_credit.py','learn_intrinsic_joint_recovery.py')],
        dict(wall_s=300,workers=4,base_episodes=8,query_states=24,new_suffixes=96,factorial_cells=192,
            reused_episode_references=24,reused_single_head_suffixes=72,parameters=54,
            actor_epochs=6,learning_rate=2e-4,clip=.2,kl_stop=.01,gradient_clip=.5,
            global_advantage_scale=scale,deterministic_validation_paths=paths),
        'Complete the eight Gaussian action combinations at each saved stratified query. '
        'Reuse8 base episodes,24 mask0 references and72 single-head suffixes; collect only96 '
        'missing multi-head suffixes. Same real future feedback and noise,bitwise parent-prefix checks. '
        'Compare head-conditional advantages with per-head ratios against complement-independent '
        'joint advantages with joint ratios. Same54 parameters,data,inclusion weights,scale,Adam '
        'and6 updates. Assert identical starting gradients. This changes both baseline structure '
        'and likelihood clipping; not an isolated clipping-only attribution. New conditional '
        'interaction then PPO actor updates; no value-critic training or feed-label fitting. '
        'Evaluate both final policies on all7 affected paths;81 zero-support paths reuse exact '
        'parent records. Keep all existing shape/physical/cruise/time guards. If neither passes, '
        'inspect measured joint effects and deterministic-vs-stochastic objective mismatch; do '
        'not repeat this batch or extend epochs. No sealed tests,templates,final adoption or '
        'estimator-bias claim. Budget300s including preparation.')
    folder=stage/'state_joint';folder.mkdir()
    raw=stage/'episodes';raw.mkdir()
    torch.save(md,stage/'initial_policy.pt')
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    def entry(records,u,tag):
        original=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
        extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,
            score=sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected)),
            measured_inequalities=original.tolist(),
            augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=[] if tag=='reused_parent' else paths,
            exact_zero_support_paths=paths if tag=='reused_parent' else [])
    zero=entry(copy.deepcopy(selected),np.zeros(md['coefficients'].size),'reused_parent')
    history=[zero];md['reused_candidates']=1
    def feasible(c):
        return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;collected=[];audit={}
    try:
        initialize(md,selected,data['retained'],raw,deadline)
        pool=mp.get_context('spawn').Pool(4,initialize,(md,selected,data['retained'],raw,deadline))
        for row in pool.imap_unordered(collect,episodes):
            collected.append(row)
            write_json(stage/'factorial_episodes.json',dict(episodes=collected))
        pool.close();pool.join();pool=None
        collected.sort(key=lambda e:(e['path'],e['seed']))
        tables=[t for e in collected for t in e['tables']]
        assert len(tables)==24 and all(len(t['cells'])==8 for t in tables)
        write_json(stage/'action_interaction_audit.json',interaction_audit(tables))
        checkpoints,audit=updates(tables,scale)
        write_json(stage/'ppo_update_audit.json',dict(audit,checkpoints=checkpoints,
            global_advantage_scale=scale,critic_updates=0,source_episode_file_sha256=sha(source/'on_policy_episodes.json')))
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        for checkpoint in checkpoints:
            tag=checkpoint['tag'];u=np.array(checkpoint['coefficients'])
            actor=copy.deepcopy(md);actor['coefficients']=u.reshape(-1,3)
            torch.save(actor,stage/(tag+'.pt'))
            records=pool.map(worker_evaluate,[(p,u,tag) for p in paths])
            by={r['path']:r for r in records}
            c=entry([by[r['path']] for r in selected],u,tag);history.append(c)
            write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
            print('FACTORIAL_PPO_CHECKPOINT',tag,c['score'],min(c['augmented_measured_inequalities']),flush=True)
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,audit['logs'],choice,parent_file,started)
        receipt(stage,dict(status='completed_factorial_joint_ppo',base_episodes=8,new_base_episodes=0,
            query_states=24,factorial_cells=192,new_complete_suffixes=96,reused_single_head_suffixes=72,
            reused_original_episode_references=24,all_cells_successful=all(c['record']['success'] for t in tables for c in t['cells']),
            actor_optimizer_steps=len(audit['logs']),critic_optimizer_steps=0,deterministic_checkpoints=2,
            selected=best()['tag'],research_acceptable=(stage/'research_candidate.pt').exists(),
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_factorial_joint_ppo',error=str(exc),
            completed_base_episode_tables=len(collected),deterministic_checkpoints=len(history)-1,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise


if __name__=='__main__':
    mp.freeze_support()
    main()
