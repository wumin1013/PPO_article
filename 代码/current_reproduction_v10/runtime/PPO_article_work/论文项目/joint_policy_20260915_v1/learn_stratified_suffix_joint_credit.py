"""Full-suffix action credit and shared-network deterministic policy optimization."""
import copy
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
from deep_joint_credit_policy import DeepJointCreditPolicy,load_credit_policy,credit_normalization
from canonical_gate_state import checkpoint_environment_factory
from analyze_nonlinear_joint_response import branch
from probe_coordinated_suffix import reconstruct
from learn_nonlinear_joint_feedback import rollout,aggregate,promotion
from learn_retained_joint_growth import smooth_cruise
from learn_absolute_shape_joint_credit import absolute_actual
from reflection_policy import reflect_observation
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}
AMPLITUDE=np.array([.005,.025,.025])
LIMITS=np.array([.003,.025,.025])


def initialize(stage,parent,baseline,retained,source,deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,parent=parent,policy=load_credit_policy(parent),baseline=baseline,
        retained=retained,source=source,factory=checkpoint_environment_factory('gate_band_canonical_kernel'),deadline=deadline)


def with_branch_loss(record,rows):
    rows_list=[record if r['path']==record['path'] else r for r in WORK['baseline']['records']]
    summary=aggregate(rows_list)
    if not record['success'] or summary['successes']!=20:return None
    record['smooth_cruise99_s']=smooth_cruise(rows)
    summary['smooth_curved_cruise99_s']=sum(r['smooth_cruise99_s'] for r in rows_list if r['path'] not in (48,57))
    return absolute_actual(dict(records=rows_list,summary=summary),WORK['retained'])


def collect_query(query):
    path,frame=query['path'],query['frame']
    old=next(r for r in WORK['baseline']['records'] if r['path']==path)
    with np.load(WORK['source']/old['trace']) as z:
        ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
    costs=dict(actor_calls=0,physics_ticks=0,saved_prefix_geometry_ticks=0)
    env=reconstruct(ref,rows,actions,frame,costs,env_factory=WORK['factory'])
    observation,initial=env.observation(),env.state()
    latent=WORK['policy'].act(observation,True)[0]
    assert np.array_equal(env.decode_action(latent),actions[frame//20])
    _,_,_,info=env.step(latent)
    assert np.array_equal(info['rows'],rows[frame:frame+len(info['rows'])])
    costs['physics_ticks']+=len(info['rows']);costs['actor_calls']+=1
    folder=WORK['stage']/f"query{query['query']}"
    folder.mkdir()
    records=[]
    for head in range(3):
        for sign in (-1,1):
            offset=sign*AMPLITUDE[head]*np.eye(3)[head]
            result,full,held,obs=branch(WORK['policy'],WORK['factory'],ref,rows,actions,
                initial,offset,costs,WORK['deadline'])
            result.update(path=path,head=head,sign=sign,
                trace=f"query{query['query']}/head{head}_{sign:+d}.npz")
            result['absolute_loss']=with_branch_loss(result,full)
            checks=promotion([result],WORK['retained'])
            result['credit_feasible']=all(checks[k] for k in ('all_complete','inward_first_turns','entry_nonworse'))
            np.savez_compressed(WORK['stage']/result['trace'],reference=ref,rows=full,
                actions=held,suffix_observations=obs,frame=frame,offset=offset)
            records.append(result)
    baseline_loss=absolute_actual(WORK['baseline'],WORK['retained'])
    gradient=np.zeros(3)
    modes=[]
    for head in range(3):
        lo,hi=records[2*head:2*head+2]
        if lo['credit_feasible'] and hi['credit_feasible']:
            gradient[head]=(hi['absolute_loss']-lo['absolute_loss'])/(2*AMPLITUDE[head])
            modes.append('central_complete_primary_feasible')
        elif hi['credit_feasible'] and hi['absolute_loss'] < baseline_loss:
            gradient[head]=(hi['absolute_loss']-baseline_loss)/AMPLITUDE[head]
            modes.append('positive_complete_improving_side')
        elif lo['credit_feasible'] and lo['absolute_loss'] < baseline_loss:
            gradient[head]=(baseline_loss-lo['absolute_loss'])/AMPLITUDE[head]
            modes.append('negative_complete_improving_side')
        else:
            modes.append('no_feasible_improving_local_derivative')
    return dict(**query,observation=observation.tolist(),records=records,
        gradient=gradient.tolist(),gradient_modes=modes,costs=costs,
        exact_next_decision_identity=True,baseline_absolute_loss=baseline_loss)


def evaluate_network(job):
    tag,metadata=job
    policy=load_credit_policy(metadata)
    folder=WORK['stage']/tag
    folder.mkdir()
    records=[]
    costs=dict(actor_calls=0,physics_ticks=0)
    for old in WORK['baseline']['records']:
        with np.load(WORK['source']/old['trace']) as z:ref=z['reference'].copy()
        result,rows,actions,observations=rollout(policy,WORK['factory'],ref,costs,WORK['deadline'])
        result.update(path=old['path'],trace=f"{tag}/path{old['path']}.npz")
        if result['success']:result['smooth_cruise99_s']=smooth_cruise(rows)
        np.savez_compressed(WORK['stage']/result['trace'],reference=ref,rows=rows,actions=actions,observations=observations)
        records.append(result)
    summary=aggregate(records)
    summary['smooth_curved_cruise99_s']=(sum(r['smooth_cruise99_s'] for r in records if r['path'] not in (48,57))
        if summary['successes']==20 else None)
    result=dict(tag=tag,records=records,summary=summary,costs=costs)
    result['absolute_objective']=absolute_actual(result,WORK['retained']) if summary['successes']==20 else None
    result['promotion']=promotion(records,WORK['retained'])
    result['primary_preserved']=all(result['promotion'][k] for k in ('all_complete','inward_first_turns','entry_nonworse'))
    return result


def main():
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_fast_absolute_shape_joint_credit_20260919'
    data=read(source/'results.json')
    baseline=next(r for r in data['candidates'] if r['tag']==data['best'])
    parent=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    stage=register('canonical_gate_band_stratified_suffix_joint_credit_20260919',
        [__file__,BASE/'deep_joint_credit_policy.py',BASE/'learn_absolute_shape_joint_credit.py',
         BASE/'analyze_nonlinear_joint_response.py',BASE/'probe_coordinated_suffix.py',
         BASE/'learn_nonlinear_joint_feedback.py',source/'research_candidate.pt',source/'results.json',source/'receipt.json'],
        dict(paths=20,queries_per_path=6,maximum_queries=120,branches_per_query=6,
             new_complete_policy_candidates=4,workers=4,wall_s=420,actor_optimizer_steps_per_arm=200,
             action_perturbation_latent=AMPLITUDE.tolist(),saved_state_function_limits=LIMITS.tolist(),seed=20260919),
        'Parameter-response learning preserved actual curved cruise but retained '
        'large reversal and individual wide-corner defects. Change credit and '
        'function learning: sample one ordinary decision uniformly within each '
        'of6 temporal strata on each of20 current actual complete trajectories. '
        'Weight each sampled decision by its stratum size. At each exact saved '
        'state perturb one of3 actions +/- once for20ms, then use the same current '
        'policy on every actual future state through complete coverage/braking/rest. '
        'Check exact nominal next-decision identity. Finite complete-consequence '
        'cost differences supply action derivatives; do not fit desired actions '
        'or feed labels, and do not call this PPO or an unbiased estimator proof. '
        'Use central derivatives only when both sides complete and preserve '
        'first-inside/incoming constraints; otherwise use a complete improving '
        'feasible side, or no derivative. Failed prefixes get no time/cruise credit. '
        'Train a shared reflection-equivariant64x64 residual network by the '
        'weighted derivative-times-action surrogate, with a training-only '
        'saved-state function trust projection. Zero residual exactly preserves '
        'the current unadopted fast candidate. Compare joint versus feed-only '
        'output masks with the SAME data, initialization, steps and optimizer. '
        'Evaluate joint scales0.25/0.5/1 and feed-only scale1 by full20 paths. '
        'The simultaneous policy update is accepted only from actual complete '
        'outcomes, never the summed local derivative prediction. Preserve86cd '
        'and all original per-corner, hard99, full64, fair and frozen generalization '
        'requirements. This is complete-suffix physical sampling followed by '
        'deterministic actor-parameter surrogate optimization, not imitation, '
        'a runtime speed rule, trajectory template or PPO interaction training.')
    started=time.perf_counter()
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
        '--active',stage.name,'--kind','complete_closed_loop_parameter_optimization',
        '--next','采集分层完整后缀三动作信用，再优化共享网络并执行完整联合/仅进给对照；不是PPO。'],
        cwd=BASE,check=True,stdout=subprocess.DEVNULL)
    rng=np.random.default_rng(20260919)
    support,queries=[],[]
    for r in baseline['records']:
        with np.load(source/r['trace']) as z:
            rows,obs=z['rows'].copy(),z['observations'].copy()
        support.append(obs)
        ordinary=np.flatnonzero(rows[np.minimum(np.arange(len(obs))*20,len(rows)-1),12]==0)
        for stratum,indices in enumerate(np.array_split(ordinary,6)):
            if len(indices):
                decision=int(rng.choice(indices))
                queries.append(dict(query=len(queries),path=r['path'],frame=decision*20,
                    stratum=stratum,stratum_size=len(indices),ordinary_decisions=len(ordinary)))
    support=np.vstack(support)
    mean,std=credit_normalization(support)
    torch.manual_seed(20260919)
    check=DeepJointCreditPolicy(parent,mean,std)
    identity=max(float(abs(check.act(x,True)[0]-check.base.act(x,True)[0]).max()) for x in support[::13])
    assert identity==0.
    write_json(stage/'initial_identity.json',dict(zero_residual_latent_error=identity,parent_sha256=sha(source/'research_candidate.pt')))
    write_json(stage/'query_plan.json',dict(queries=queries))
    pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,baseline,retained,source,started+410))
    collected,outcomes=[],[]
    try:
        for result in pool.imap_unordered(collect_query,queries):
            collected.append(result)
            write_json(stage/'query_results.json',dict(queries=collected,adopted=False,goal_complete=False))
            if len(collected)%12==0:
                print('FULL_SUFFIX_CREDIT',len(collected),'queries','complete',sum(r['success'] for q in collected for r in q['records']),
                    'primary_feasible',sum(r['credit_feasible'] for q in collected for r in q['records']),flush=True)
        collected.sort(key=lambda r:r['query'])
        qobs=torch.tensor([q['observation'] for q in collected],dtype=torch.float32)
        gradients=torch.tensor([q['gradient'] for q in collected],dtype=torch.float32)
        weights=torch.tensor([q['stratum_size'] for q in collected],dtype=torch.float32)[:,None]
        anchors=torch.as_tensor(support,dtype=torch.float32)
        limits=torch.as_tensor(LIMITS,dtype=torch.float32)
        normalization=float((abs(gradients)*weights*limits).sum())
        assert normalization>1e-8,'No usable complete-consequence gradient'
        np.savez_compressed(stage/'action_credit.npz',observations=qobs.numpy(),gradients=gradients.numpy(),
            weights=weights.numpy(),limits=LIMITS,amplitude=AMPLITUDE)
        jobs=[]
        training={}
        baseline_loss=absolute_actual(baseline,retained)
        for arm in ('joint','feed_only'):
            torch.manual_seed(20260919)
            p=DeepJointCreditPolicy(parent,mean,std)
            with torch.no_grad():p.output_mask.copy_(torch.tensor([1.,1.,1.] if arm=='joint' else [0.,1.,0.]))
            optimizer=torch.optim.Adam(p.residual.parameters(),lr=.003)
            history=[]
            for step_index in range(200):
                correction=p.correction(qobs)
                anchor=p.correction(anchors)/limits
                linear=(correction*gradients*weights).sum()/normalization
                loss=linear+.02*anchor.square().mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
                with torch.no_grad():
                    maximum=float(abs(p.correction(anchors)/limits).max())
                    if maximum>1.:
                        p.residual[-1].weight.div_(maximum);p.residual[-1].bias.div_(maximum)
                if step_index%50==0 or step_index==199:
                    history.append(dict(step=step_index,surrogate=float(linear.detach()),
                        maximum_normalized_saved_state_change=float(abs(p.correction(anchors)/limits).max().detach())))
            network=copy.deepcopy(p.residual.state_dict())
            with torch.no_grad():
                pred=(p.correction(qobs)*gradients*weights).sum().item()
                parity=float(abs(p.correction(reflect_observation(anchors))-p.correction(anchors)*torch.tensor([-1.,1.,1.])).max())
                max_change=float(abs(p.correction(anchors)/limits).max())
            assert parity<1e-7 and max_change<=1.+1e-5
            training[arm]=dict(history=history,predicted_cost_change=pred,reflection_error=parity,
                               maximum_normalized_saved_state_change=max_change)
            for scale in ((.25,.5,1.) if arm=='joint' else (1.,)):
                metadata=dict(architecture='deep_joint_credit',parent=parent,mean=mean,std=std,residual=network,
                    output_mask=p.output_mask.numpy().copy(),step_scale=scale,environment_architecture='gate_band_canonical_kernel',
                    learning_method='stratified_complete_suffix_action_credit_parameter_optimization')
                tag=f'{arm}_{scale}'
                torch.save(metadata,stage/f'{tag}.pt')
                jobs.append((tag,metadata))
        write_json(stage/'actor_optimization.json',dict(arms=training,normalization=normalization,baseline_absolute_loss=baseline_loss))
        for result in pool.imap_unordered(evaluate_network,jobs):
            outcomes.append(result)
            write_json(stage/'results.json',dict(candidates=outcomes,baseline_absolute_objective=baseline_loss,
                training=training,adopted=False,goal_complete=False))
            print('SUFFIX_LEARNED_ACTUAL',result['tag'],result['absolute_objective'],result['summary'],result['promotion'],flush=True)
        pool.close();pool.join();pool=None
        usable=[r for r in outcomes if r['primary_preserved'] and r['absolute_objective']<baseline_loss-1e-5]
        best=min(usable,key=lambda r:r['absolute_objective']) if usable else None
        if best:
            chosen=dict(next(md for tag,md in jobs if tag==best['tag']))
            torch.save(chosen,stage/'research_candidate.pt')
            write_json(stage/'results.json',dict(candidates=outcomes,best=best['tag'],baseline_absolute_objective=baseline_loss,
                training=training,adopted=False,goal_complete=False))
        receipt(stage,dict(status='completed_stratified_suffix_joint_parameter_learning',queries=len(collected),
            branches=sum(len(q['records']) for q in collected),new_candidates=len(outcomes),
            best=None if best is None else best['tag'],wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_stratified_suffix_joint_parameter_learning',error=str(exc),
            queries=len(collected),new_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
            '--kind','complete_closed_loop_parameter_optimization_finished',
            '--next','读取分层完整后缀信用的共享网络联合/仅进给真实结果；持续修正三动作学习直到原目标通过。'],
            cwd=BASE,check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
