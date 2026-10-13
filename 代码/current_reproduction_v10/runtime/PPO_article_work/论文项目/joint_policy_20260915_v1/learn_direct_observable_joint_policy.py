"""Optimize a compact observable-state three-action policy in the real full loop."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import multiprocessing as mp
import time
import numpy as np
import torch
from scipy.optimize import minimize
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import JointTeacherStatePolicy,state_pair,features
from learn_shared_temporal_joint_policy import initialize,inequalities,evaluate_complete_parameters
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_full_pool_joint_credit import aggregate
from localize_joint_shape_residual import characterize
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


WORK={}


def worker_initialize(metadata,records,folder,deadline):
    torch.set_num_threads(1)
    WORK.update(policy=JointTeacherStatePolicy(metadata),records={r['path']:r for r in records},
                folder=folder,deadline=deadline)


def worker_evaluate(job):
    path,coefficients,tag=job
    return evaluate_complete_parameters(WORK['policy'],coefficients,[WORK['records'][path]],
        WORK['folder'],tag,WORK['deadline'])[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--continue-stage')
    parser.add_argument('--output-stage',default='canonical_gate_band_direct_observable_joint_learning_20260920')
    parser.add_argument('--first-block',type=int,default=0)
    parser.add_argument('--wall-s',type=float,default=480.)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_signed_slack_credit_learning_20260920'
    parent_file=BASE/'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920/research_candidate.pt'
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    assert sha(parent_file)==read(source/'receipt.json')['source_model_sha256']
    previous=BASE/args.continue_stage if args.continue_stage else None
    if previous:
        assert read(previous/'receipt.json')['source_model_sha256']==sha(parent_file)
    stage=register(args.output_stage,
        [__file__,BASE/'joint_teacher_state_policy.py',BASE/'consolidate_joint_teacher_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_complete_joint_teachers.py',
         BASE/'learned_action_pulse_policy.py',BASE/'localize_joint_shape_residual.py',
         parent_file,source/'baseline.json',source/'receipt.json',
         BASE/'canonical_gate_band_secant_restored_joint_credit_learning_20260920/receipt.json']+
         ([previous/'receipt.json',previous/'replayed_states.npz',previous/'support_audit.json',
           previous/'state_joint/results.json'] if previous else []),
        dict(wall_s=args.wall_s,maximum_candidates_per_block=90,blocks=list(range(args.first_block,2)),coefficient_bounds=[-2.,2.],
             kernel_width_choices=[.3,.6],maximum_preferred_affected_paths=12,
             centers=4,outputs=3,maximum_new_full_pool_validations=1,continue_stage=args.continue_stage,
             exact_zero_support_reuse=True,workers=args.workers),
        ('Continue only pending coefficient blocks from the saved actual outcomes and '
         'exact state replay; no prior parameter vector is rerun. Before each new vector '
         'identify paths where EVERY active kernel is exactly zero on ALL saved parent '
         'states. Determinism then proves unchanged actions/trajectory inductively; reuse '
         'those records and audit the support mask instead of rerunning zero controls. '
         if previous else '')+
        'Repeated global compressed-feature steps fail actual nondegradation. Change the '
        'learned function, not the metric thresholds: four compact centers in the FULL '
        'observable geometry/physics/previous-action/innovation state independently '
        'control all three latent actions. Choose center states from the latest two '
        'unresolved training traces at target jump and post-apex reverse onset. These '
        'indices only select TRAINING observations and are never policy inputs. Replay '
        'saved states with exact filtered heading and decoded-action assertions. Choose '
        'the broader of two registered widths with at most12 affected saved paths. '
        'Optimize two six-dimensional coefficient blocks by actual full closed-loop '
        'COBYLA, using original physical/shape/time/cruise guards and smooth objective. '
        'All trajectories outside compact support are reusable by deterministic zero '
        'residual induction, with an explicit support audit. Evaluate a useful final '
        'policy on every affected route and preserve raw1ms data. This directly learns '
        'a deployable state feedback function, not teacher schedules or feed labels; '
        'still research-only parameter optimization, not PPO or generalization proof.')
    started=time.perf_counter();folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);history=[];blocks=[];pool=None
    try:
        if previous:
            with np.load(previous/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
        else:training=collect_data(parent,parent,baseline)
        assert not np.any(training['labels'])
        np.savez_compressed(stage/'replayed_states.npz',**training)
        a,b=state_pair(training['observations'],training['innovations'])
        both=np.vstack([a,b]);mean=both.mean(0);std=np.maximum(both.std(0),.05)
        a,b=(a-mean)/std,(b-mean)/std
        indices=[];anchors=[]
        for path in (206,304):
            r=next(r for r in baseline if r['path']==path);loc=characterize(r['trace'])
            recovery=max((i for i in loc['negative_turn_intervals'] if i['start_ms']>loc['apex_ms']),
                         key=lambda i:i['reverse_deg'])
            for kind,tick in (('target_change',loc['target_jumps'][0]['time_ms']),
                              ('reverse_onset',recovery['start_ms'])):
                decision=int(tick)//20
                index=np.flatnonzero((training['path_ids']==path)&(training['decisions']==decision)).item()
                indices.append(index);anchors.append(dict(path=path,kind=kind,decision=decision))
        md=dict(architecture='joint_teacher_state_radial',parent=parent,mean=mean,std=std,
                centers=a[indices],coefficients=np.zeros((4,3)),accepted=False,
                learning_method='direct_observable_state_complete_return_optimization',
                action_interval_ms=20,physical_tick_ms=1)
        choices=[]
        for width in (.3,.6):
            md['radii']=np.full(4,width)
            f=features(md,training['observations'],training['innovations'])
            active=np.any(f!=0.,axis=(1,2));affected=sorted(np.unique(training['path_ids'][active]).tolist())
            choices.append(dict(width=width,paths=affected,decisions=int(active.sum())))
        eligible=[c for c in choices if len(c['paths'])<=12]
        choice=max(eligible or choices[:1],key=lambda c:c['width']);md['radii']=np.full(4,choice['width'])
        touched=choice['paths'];unaffected=[r['path'] for r in baseline if r['path'] not in touched]
        write_json(stage/'support_audit.json',dict(anchors_for_training_only=anchors,width_choices=choices,
            selected=choice,unaffected_paths=unaffected,decoded_actions_exact=True,
            zero_label_replay=True,policy_has_no_time_path_or_angle_table=True))
        print('DIRECT_STATE_SUPPORT',choice,'unchanged_paths',len(unaffected),flush=True)
        assert all(p in touched for p in (206,304))
        # This stage handles isolated corners; support elsewhere requires a broader evaluator.
        selected=[r for r in baseline if r['path'] in touched]
        assert all('shape' in r for r in selected),'Compact support reached a non-corner route'
        retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==r['path']) for r in selected]
        deadline=started+args.wall_s-20
        initialize(stage,parent,selected,retained,deadline,True,None,None,True,{'direct_state':True},[104,204,206,304])
        policy=JointTeacherStatePolicy(md)
        saved_features=features(md,training['observations'],training['innovations'])
        def score(records):return float(sum(teacher_score(r,b,o,barrier=False) for r,b,o in zip(records,selected,retained)))
        zero=dict(tag='reused_identity',coefficients=[0.]*12,records=selected,score=score(selected),
                  measured_inequalities=inequalities(selected).tolist())
        history.append(zero)
        if previous:
            old=read(previous/'state_joint/results.json')['candidates']
            assert old[0]['score']==zero['score']
            history.extend(copy.deepcopy(old[1:]))
            raw_cache={}
            def raw_trace(file):
                if file not in raw_cache:
                    with np.load(file) as z:raw_cache[file]=(z['rows'].copy(),z['actions'].copy())
                return raw_cache[file]
            checked=0
            for c in old[1:]:
                active=np.any(np.asarray(c['coefficients']).reshape(-1,3)!=0.,axis=1)
                for r,b in zip(c['records'],selected):
                    if not np.any(saved_features[training['path_ids']==r['path']][:,active,:]):
                        actual,reference=raw_trace(r['trace']),raw_trace(b['trace'])
                        assert all(np.array_equal(x,y) for x,y in zip(actual,reference)),('zero_support_identity',c['tag'],r['path'])
                        checked+=1
            write_json(stage/'zero_support_reuse_audit.json',dict(previous_vectors=len(old)-1,
                previously_executed_zero_support_traces=checked,all_rows_actions_bitwise_identical=True,
                new_physical_runs_for_audit=0))
            print('ZERO_SUPPORT_REUSE_AUDIT',checked,'previously executed traces identical',flush=True)
        reused_count=len(history)
        md['reused_candidates']=reused_count
        cache={tuple(np.round(c['coefficients'],12)):c for c in history}
        if args.workers>1:
            pool=mp.get_context('spawn').Pool(args.workers,worker_initialize,(md,selected,folder,deadline))
        def feasible(c):return parameters_in_bounds(c['coefficients'],-2.,2.) and min(c['measured_inequalities'])>=-1e-8
        def evaluate(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'parameters_{len(history)}'
                active_centers=np.any(np.asarray(u).reshape(-1,3)!=0.,axis=1)
                zero_paths=[r['path'] for r in selected if not np.any(
                    saved_features[training['path_ids']==r['path']][:,active_centers,:])]
                if pool is None:
                    records=evaluate_complete_parameters(policy,u,selected,folder,tag,deadline,zero_support_paths=zero_paths)
                else:
                    measured=pool.map(worker_evaluate,[(r['path'],u,tag) for r in selected if r['path'] not in zero_paths])
                    changed={r['path']:r for r in measured}
                    records=[changed.get(r['path'],dict(copy.deepcopy(r),reused_exact_zero_support=True)) for r in selected]
                c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,
                       score=score(records),measured_inequalities=inequalities(records).tolist(),
                       exact_zero_support_paths=zero_paths,actual_rerun_paths=[r['path'] for r in selected if r['path'] not in zero_paths])
                history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=min(filter(feasible,history),key=lambda c:c['score'])['tag']))
                if len(history)%12==0:
                    best=min(filter(feasible,history),key=lambda c:c['score'])
                    print('DIRECT_STATE_EVALUATED',len(history)-1,'best',best['score'],'latest_min',min(c['measured_inequalities']),flush=True)
            return cache[key]
        incumbent=min(filter(feasible,history),key=lambda c:c['score'])
        for block in range(args.first_block,2):
            if time.perf_counter()>deadline-10:break
            anchor=np.asarray(incumbent['coefficients']).copy();indices=np.arange(6*block,6*(block+1))
            def expand(v):
                value=anchor.copy();value[indices]=v;return value
            before=len(history)
            solved=minimize(lambda v:evaluate(expand(v))['score'],anchor[indices],method='COBYLA',
                bounds=[(-2.,2.)]*6,constraints=[dict(type='ineq',fun=lambda v:np.array(evaluate(expand(v))['measured_inequalities']))],
                options=dict(maxiter=90,rhobeg=.15,tol=1e-5,catol=1e-8))
            incumbent=min(filter(feasible,history),key=lambda c:c['score'])
            blocks.append(dict(block=block,new_candidates=len(history)-before,best=incumbent['tag'],
                score=incumbent['score'],solver=str(solved.message)))
            print('DIRECT_STATE_BLOCK',blocks[-1],flush=True)
        finish(stage,md,baseline,data,incumbent,zero,history,blocks,choice,parent_file,started)
        if pool is not None:pool.close();pool.join();pool=None
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        if not history:raise
        incumbent=min((c for c in history if parameters_in_bounds(c['coefficients'],-2.,2.)
                       and min(c['measured_inequalities'])>=-1e-8),key=lambda c:c['score'])
        finish(stage,md,baseline,data,incumbent,zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join();pool=None
        receipt(stage,dict(status='failed_direct_observable_joint_learning',error=str(exc),
            candidates=len(history),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


def finish(stage,md,baseline,data,best,zero,history,blocks,choice,parent_file,started,budget_exhausted=False):
    improved=best['score']<zero['score']-.001
    md['coefficients']=np.array(best['coefficients']).reshape(-1,3)
    torch.save(md,stage/'state_joint/candidate.pt')
    write_json(stage/'state_joint/results.json',dict(candidates=history,best=best['tag']))
    write_json(stage/'optimization_blocks.json',blocks)
    by_path={r['path']:r for r in best['records']}
    records=[copy.deepcopy(by_path.get(r['path'],r)) for r in baseline]
    summary=aggregate(records);old=data['baseline']['summary']
    accepted=improved and summary['complete_time_s']<=old['complete_time_s']+.02 and summary['curved_cruise99_s']>=old['curved_cruise99_s']-.05
    accepted=bool(accepted and summary['reverse_sum_deg']<old['reverse_sum_deg'] and summary['exit_sum_mm']<=old['exit_sum_mm'] and summary['energy_sum']<=old['energy_sum'])
    c=dict(tag='state_joint',records=records,summary=summary,
        original64_summary=aggregate([r for r in records if r['path']<64]),
        added_training_summary=aggregate([r for r in records if r['path']>=64]),
        actual_rerun_paths=best.get('actual_rerun_paths',choice['paths']) if improved else [],
        zero_support_reused_paths=[r['path'] for r in baseline if r['path'] not in choice['paths']]+best.get('exact_zero_support_paths',[]),
        research_acceptable=accepted,adopted=False)
    result=dict(candidates=[c],selected_parameter=best['tag'],improved=improved,adopted=False,goal_complete=False)
    if accepted:
        torch.save(md,stage/'research_candidate.pt');result['best']='state_joint'
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_direct_observable_joint_learning',new_candidates=len(history)-md.get('reused_candidates',1),
        total_nonzero_candidates=len(history)-1,
        selected_parameter=best['tag'],improved=improved,research_acceptable=accepted,
        source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
        reused_candidates=md.get('reused_candidates',1),actual_new_candidates=len(history)-md.get('reused_candidates',1),
        budget_exhausted=budget_exhausted,adopted=False,goal_complete=False))
    print('DIRECT_STATE_COMPLETE',best['tag'],best['score'],accepted,summary,flush=True)


if __name__=='__main__':mp.freeze_support();main()
