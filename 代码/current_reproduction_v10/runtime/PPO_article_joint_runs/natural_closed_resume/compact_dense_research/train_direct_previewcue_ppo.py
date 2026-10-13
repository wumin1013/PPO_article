"""Actual PPO using14input preview cue, exact offline initialization and all34causal contexts."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from evaluate_historical_dense import ROOT, BASE, HIST, CASE, sha, write, verify_bound_inputs

STAGE=ROOT/'direct14_previewcue_ppo_seed922302_round13'
INITIAL=ROOT/'direct14_previewcue_fit_seed922302_round11/fit_12000.pt'
CASES={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
BASELINE={'original':444.356,'dense':15.056}
SEED=922302
CASE_BATCHES={'original':1536,'dense':512}
POTENTIAL_SPEED=100.


def worker():
    import numpy as np
    import torch
    from collections import Counter
    from compact_fine_policy import FineCompactPolicy as CompactPolicy
    from previewcue_learning_runtime import PreviewCueLearningRuntime as DirectCompactRuntime
    from learning_update_guarded_mc import optimizers, update
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    from analyze_dense_trace import analyze
    started=time.perf_counter();deadline=started+1380.
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    references={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in CASES.items()}
    geometries={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in CASES.items()}
    policy=CompactPolicy(14,3,SEED,feed_gain=1.)
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    policy.load_state_dict(initial['policy'])
    envs={k:DirectCompactRuntime(r,potential_speed=POTENTIAL_SPEED) for k,r in references.items()}
    sample=torch.as_tensor(np.stack([e.observation() for e in envs.values()]))
    with torch.no_grad():
        before=policy.actor(sample).clone()
        policy.log_std.copy_(torch.log(torch.tensor([.0017,.015,.02])))
        assert torch.equal(before,policy.actor(sample))
    assert all(torch.equal(value,initial['policy'][key]) for key,value in policy.state_dict().items() if key!='log_std')
    optimizer=optimizers(policy)
    class Collector(DirectCompactRuntime):
        def __init__(self,reference):
            super().__init__(reference,potential_speed=POTENTIAL_SPEED);self.contexts=[]
        def observation(self):
            x=super().observation();corners=geometries['original']['sharp_corners']
            while len(self.contexts)<34 and self.env.progress()[0]>=corners[len(self.contexts)]['reference_s_mm']-35.:
                c=corners[len(self.contexts)]
                self.contexts.append(dict(id=c['id'],progress_mm=self.env.progress()[0],state=self.state(),observation=x.copy()))
            return x
    collector=Collector(references['original'])
    collection=evaluate(policy,collector,STAGE/'context_collection.npz',deadline)
    with np.load(STAGE/'context_collection.npz') as d:collection_rows=d['rows']
    with np.load(ROOT/'direct14_previewcue_fit_seed922302_round11/eval_12000_original.npz') as d:expected=d['rows']
    assert collection['success'] and np.array_equal(collection_rows,expected),'Context collection must reproduce the complete incumbent trajectory'
    collection['analytic']=analytic_command_audit(references['original'][0],collection_rows,geometries['original'])
    assert collection['analytic']['analytic_corridor_pass'] and len(collector.contexts)==34
    contexts=collector.contexts;context_cursor=0;current_corner=''
    for context in contexts:
        envs['original'].restore(context['state'])
        assert np.array_equal(envs['original'].observation(),context['observation'])
    torch.save(contexts,STAGE/'all34_contexts.pt')
    collection.update(context_count=34,bitwise_full_trace_reproduction=True,exact_observation_restore=True,
        contexts_sha256=sha(STAGE/'all34_contexts.pt'),training_only=True)
    write(STAGE/'context_collection_audit.json',collection)
    print('CONTEXTS_READY',len(contexts),'wall',time.perf_counter()-started,flush=True)
    updates=[];evaluations=[];episodes=[];steps=0;update_count=0
    episode_counts={k:0 for k in CASES};episode_reward={k:0. for k in CASES}

    def save(label):
        path=STAGE/(label+'.pt')
        torch.save(dict(policy=policy.state_dict(),optimizers={k:v.state_dict() for k,v in optimizer.items()},
            environments={k:v.state() for k,v in envs.items()},steps=steps,updates=update_count,context_cursor=context_cursor,
            episode_counts=episode_counts.copy(),episode_reward=episode_reward.copy(),
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            interface=[14,3,SEED],observation_features=envs['original'].feature_names(),policy_memory_scalars=0,feed_gain=1.,policy_class='FineCompactPolicy',
            initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL)),path)
        return path

    def full_check(case,path):
        label=f'eval_{update_count:04d}_{case}';trace=STAGE/(label+'.npz')
        env=DirectCompactRuntime(references[case],potential_speed=POTENTIAL_SPEED)
        result=evaluate(policy,env,trace,deadline)
        with np.load(trace) as data:rows=data['rows']
        if len(rows):
            result['analytic']=analytic_command_audit(references[case][0],rows,geometries[case])
            if not result['analytic']['analytic_corridor_pass']:
                result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        result.update(checkpoint=str(path),checkpoint_sha256=sha(path),ppo_updates=update_count,
            observation_dimension=14,policy_memory_scalars=0,inherited_controller_at_execution=False,
            goal_complete=False,method_frozen=False,feed_gain=1.,
            observation_features=env.feature_names(),global_remaining_distance_input=False,far_turn_cue_mm=30)
        write(trace.with_suffix('.json'),result);write(STAGE/(label+'_gates.json'),env.tracker.events)
        if result['success']:
            m=metrics(references[case],rows,geometries[case]) if case=='original' else analyze(trace,trace.with_suffix('.json'))
            write(STAGE/(label+'_metrics.json'),m)
        print('FULL_CHECK',json.dumps(dict(update=update_count,case=case,success=result['success'],
            time_s=result['complete_time_s'],prefix_s=result['elapsed_prefix_or_complete_s'],reason=result['reason'])),flush=True)
        return result

    save('initial')
    while update_count<24 and time.perf_counter()<deadline-40.:
        keys=['observation','next_observation','latent','log_probability','value','reward','next_value','terminal','segment_end']
        batch={k:[] for k in keys};held=[];progress=[];times=[];case_ids=[];corner_ids=[]
        collected=time.perf_counter();interrupted=False
        for case in CASES:
            for j in range(CASE_BATCHES[case]):
                if time.perf_counter()>=deadline-40.:interrupted=True;break
                if case=='original' and (j%32==0 or envs[case].done):
                    context=contexts[context_cursor%34];context_cursor+=1;envs[case].restore(context['state'])
                    current_corner=context['id']
                env=envs[case];x=env.observation();z,logp,value=policy.act(x)
                x2,reward,done,info=env.step(z)
                with torch.no_grad():next_value=0. if done else float(policy.value(torch.as_tensor(x2)))
                values=[x,x2,z,logp,value,reward,next_value,done,(j==CASE_BATCHES[case]-1 or (case=='original' and j%32==31))]
                for k,v in zip(keys,values):batch[k].append(v)
                held.append(info['held_action']);progress.append(env.progress()[0]);times.append(env.ticks*.001);case_ids.append(case);corner_ids.append(current_corner if case=='original' else '')
                episode_reward[case]+=reward
                if done:
                    record=dict(case=case,episode=episode_counts[case],step=steps+len(batch['reward']),
                        success=env.success,reason=env.reason,prefix_s=env.ticks*.001,
                        complete_time_s=env.ticks*.001 if env.success else None,gates=env.tracker.frontier,
                        progress_mm=env.progress()[0],reward=episode_reward[case])
                    episodes.append(record)
                    with (STAGE/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                    episode_counts[case]+=1;episode_reward[case]=0.
                    if case=='dense':envs[case]=DirectCompactRuntime(references[case],potential_speed=POTENTIAL_SPEED)
            if interrupted:break
        if interrupted:
            if batch['reward']:
                batch['segment_end'][-1]=True
                np.savez_compressed(STAGE/'unapplied_partial_batch.npz',**{k:np.asarray(v) for k,v in batch.items()},case=np.asarray(case_ids))
            break
        assert len(batch['reward'])==2048 and Counter(case_ids)==CASE_BATCHES
        np.savez_compressed(STAGE/f'batch_{update_count+1:04d}.npz',**{k:np.asarray(v) for k,v in batch.items()},
            held_action=np.asarray(held),progress_mm=np.asarray(progress),episode_time_s=np.asarray(times),case=np.asarray(case_ids),corner_id=np.asarray(corner_ids))
        collected_wall=time.perf_counter()-collected;updating=time.perf_counter()
        statistics=update(policy,optimizer,batch)
        steps+=2048;update_count+=1
        stats=dict(update=update_count,steps=steps,collect_wall_s=collected_wall,update_wall_s=time.perf_counter()-updating,
            completed_episodes=episode_counts.copy(),latent_std=policy.distribution(sample).scale[0].detach().tolist(),**statistics)
        updates.append(stats);write(STAGE/'updates.json',updates)
        path=save(f'checkpoint_{update_count:04d}')
        print('UPDATE',update_count,'steps',steps,'KL',stats['post_update_exact_kl'],
            'episodes',episode_counts,'wall',time.perf_counter()-started,flush=True)
        if update_count in [12,24]:
            results={case:full_check(case,path) for case in ['dense','original']}
            eligible=all(results[k]['success'] and results[k]['complete_time_s']<=BASELINE[k]+1e-9 for k in CASES)
            eligible=bool(eligible and any(BASELINE[k]-results[k]['complete_time_s']>.020+1e-9 for k in CASES))
            corner_screen={}
            if results['original']['success']:
                current=json.loads((STAGE/f'eval_{update_count:04d}_original_metrics.json').read_text(encoding='utf-8'))
                protocol=json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
                for name,entry in protocol['original']['comparators'].items():
                    old=json.loads(Path(entry['path']).read_text(encoding='utf-8'))
                    delta=[a['speed_deficit_mm']-b['speed_deficit_mm'] for a,b in zip(current['corners'],old['corners'])]
                    corner_screen[name]=dict(not_worse=sum(d<=1e-7 for d in delta),total_change_mm=sum(delta))
                eligible=bool(eligible and corner_screen['historical_main']['not_worse']>=18
                    and corner_screen['historical_main']['total_change_mm']<=1.0879718582439666+1e-7
                    and corner_screen['S2']['not_worse']>=28 and corner_screen['historical_strong_fixed']['not_worse']>=22)
            evaluations.append(dict(update=update_count,checkpoint=str(path),results=results,eligible_for_half_ms=eligible,corner_screen=corner_screen))
            write(STAGE/'evaluations.json',evaluations)
        write(STAGE/'live.json',dict(status='training',steps=steps,updates=update_count,episodes=episode_counts.copy(),
            wall_s=time.perf_counter()-started,goal_complete=False))
    save('final')
    eligible=[r for r in evaluations if r['eligible_for_half_ms']]
    best=min(eligible,key=lambda r:(r['results']['dense']['complete_time_s'],r['results']['original']['complete_time_s'])) if eligible else None
    result=dict(status='step_budget_completed' if update_count==24 else 'wall_budget_completed',
        training_kind='Fresh on-policy PPOwith new14input preview-cue encoding after explicit offline initialization; no hidden teacher',
        initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),baseline_complete_times_s=BASELINE,
        observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,inherited_controller_at_execution=False,
        ppo_updates_this_round=update_count,new_decisions=steps,new_decisions_per_case={k:n*update_count for k,n in CASE_BATCHES.items()},potential_speed=POTENTIAL_SPEED,
        episodes=episode_counts,episode_reasons=dict(Counter(r['reason'] for r in episodes)),
        scheduled_evaluations=evaluations,selected_for_half_ms=best,adopted=False,
        retained_checkpoint=str(INITIAL),feed_gain=1.,policy_class='FineCompactPolicy',
        inherited_ppo_updates=40,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False)
    write(STAGE/'result.json',result)
    print('ROUND_FINISHED',update_count,'eligible',len(eligible),'wall',result['wall_s'],flush=True)


def main():
    if '--worker' in sys.argv:worker();return
    from verify_multicase_returns import verify
    checks=verify();manifest=verify_bound_inputs()
    assert sha(INITIAL)=='a8760c9baaac2b519e3a4f5f8f019903f1d5f386ecc97707fc63b0a288f68c88'
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'compact_fine_policy.py',ROOT/'compact_previewcue_runtime.py',ROOT/'previewcue_learning_runtime.py',ROOT/'compact_bounded_failure_runtime.py',ROOT/'compact_speed_deficit_runtime.py',ROOT/'verify_multicase_returns.py',ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',
        ROOT/'learning_update_compact_mc.py',ROOT/'learning_update_guarded_mc.py',ROOT/'evaluate_historical_dense.py',
        ROOT/'analyze_dense_trace.py',ROOT/'refine_previewcue_compact_policy.py',BASE/'learning_update_v4.py',
        BASE/'joint_policy.py',BASE/'paper_complete_evaluation.py',BASE/'natural_case_audit.py',BASE/'natural_case_metrics.py']
    hashes={str(p):sha(p) for p in files}
    for i,p in enumerate(files):shutil.copy2(p,STAGE/'source'/f'{i:02d}_{p.name}')
    write(STAGE/'numerical_checks.json',checks)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Actual PPO on14input preview-cue encoding after offline fit',
        seed=SEED,wall_budget_s=1800,training_and_checks_child_timeout_s=1450,worker_deadline_s=1380,
        optional_half_ms_timeout_s=290,initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
        source_sha256=hashes,historical_binding_manifest=manifest,
        case_sha256={k:{'reference':sha(p/'reference.csv'),'geometry':sha(p/'geometry.json')} for k,p in CASES.items()},
        initial_complete_trace_sha256=sha(ROOT/'direct14_previewcue_fit_seed922302_round11/eval_12000_original.npz'),feed_gain=1.,
        baseline_complete_times_s=BASELINE,ppo_updates=24,new_decisions=49152,decisions_per_case_per_batch=CASE_BATCHES,potential_speed=POTENTIAL_SPEED,
        scheduled_checks=[12,24],exploration_std=[.0017,.015,.02],actor_mean_initialization='Exact round11fit12000deterministic actor on14input preview-cue encoding; identity output activation. Current-encoder initialization is offline fitted, not yetPPOtrained; earlier ancestry40PPOupdates used the previous encoder. New complete original replay verifies exact trajectory and collects true34corner states. Set exploration std0.0017/0.015/0.02without mean change.',
        observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,case_id_observation=False,
        global_remaining_distance_input=False,far_turn_cue_mm=30,
        sampler='Raw Gaussian actions/likelihoods. Dense full episodes persist; original training restarts from all34 witnessed approach states in cyclic order,32decisions per segment. These are bootstrapped training fragments, never complete-task timing results; full original and dense evaluations unchanged.',
        returns='Separate unfinished rollouts; real terminals; gamma1/GAE0.98. Phi=(progress-total)/100. Extra reward=-integral(max(100-speed,0))/100 with coefficient1 on BOTH cases, including complete terminal motion. This changes the training objective to favor smaller speed deficit as well as time; final acceptance remains full physical time and all34. On failure replace max_task_time+1+remaining/20 by5+remaining/100; the remaining term cancels the zero-terminal-potential bonus. Successful rewards unchanged. Physical failures remain failures.',
        optimizer='Separate Adam actor3e-4/critic1e-3; critic32epochs then actor6epochs; minibatch256;clip0.2;full-batch post-step KL<=0.03 with parameter and optimizer rollback.',
        evidence_gap='New14input preview-cue model is complete at both dense resolutions:15.056/15.0955s, both>5percent faster than the strong two-action reference. Original444.356s and18of34historical corners fail joint acceptance. This round performs actualPPOwith the new observations, targeting the original path while preserving dense efficiency; no teacher at inference and no global remaining-distance policy input.',
        adoption='Both full times<=444.356/15.056s and one improvement>20ms; no degradation in initial18historical,28S2,22strongFixed nonworse counts or historical total deficit change1.087972mm. Then half-ms dense full success and<=15.0955s. This is an intermediate candidate screen; final all34historical/S2/Fixed, credible historical long-time gain, matched compact two-action training and repeats remain required.',
        stop='24updates or registered wall limit; no eligible checkpoint means retain initialization and no refined run; preserve failures and all checkpoints.',
        joint_acceptance=str(ROOT/'joint_lowdim_acceptance_20260922.json'),
        joint_acceptance_sha256=sha(ROOT/'joint_lowdim_acceptance_20260922.json'),
        goal_complete=False,method_frozen=False))
    start=time.perf_counter();code=None;refined=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1450)
            code=run.returncode;status='finished'
        except subprocess.TimeoutExpired:status='timeout'
    result_path=STAGE/'result.json'
    if code==0 and result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'));candidate=result['selected_for_half_ms']
        if candidate and time.perf_counter()-start<1480.:
            name=STAGE.name+'_half_ms';checkpoint=candidate['checkpoint']
            nominal=STAGE/f"eval_{candidate['update']:04d}_dense.json"
            command=[sys.executable,'-B',str(ROOT/'refine_previewcue_compact_policy.py'),'--name',name,
                '--checkpoint',checkpoint,'--nominal',str(nominal),'--feed-gain','1']
            with (STAGE/'refined_stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'refined_stderr.txt').open('w',encoding='utf-8') as err:
                try:refine_run=subprocess.run(command,stdout=out,stderr=err,timeout=290)
                except subprocess.TimeoutExpired:refine_run=None
            refined=ROOT/name/'result.json'
            if refine_run is not None and refine_run.returncode==0 and refined.exists():
                r=json.loads(refined.read_text(encoding='utf-8'))
                result['half_ms_result']=str(refined)
                result['adopted']=bool(r['success'] and r['complete_time_s']<=15.0955+1e-9)
                if result['adopted']:result['retained_checkpoint']=checkpoint
            write(result_path,result)
    unchanged=all(sha(p)==h for p,h in hashes.items())
    verify_bound_inputs()
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=unchanged,half_ms_result=str(refined) if refined else None,goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()
