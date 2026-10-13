"""Fresh PPO after validated14feature migration; retain the same corner objective."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from evaluate_historical_dense import ROOT, BASE, sha, write, verify_bound_inputs

STAGE = ROOT/'direct14_farvector_corner_ppo_seed922302_round20'
PREPARATION = ROOT/'direct14_farvector_feedback_fit_round19'
CONTEXT_SOURCE = ROOT/'direct14_guided_time_ppo_seed922302_round16'
TEACHER_INITIAL = ROOT/'direct14_previewcue_ppo_seed922302_round13/checkpoint_0024.pt'
TARGETS = ROOT/'direct14_joint_response_probe_round14/offline_target_feasibility.json'
SEED = 922302
CASE_BATCHES = {'original':1536, 'dense':512}


def initial_path():
    selected = json.loads((PREPARATION/'result.json').read_text(encoding='utf-8'))['selected_initialization']
    assert selected and selected['eligible_initialization']
    return Path(selected['checkpoint'])


def worker():
    import numpy as np
    import torch
    from collections import Counter
    from compact_fine_policy import FineCompactPolicy
    from compact_previewcue_runtime import PreviewCueRuntime
    from compact_joint_response import JointResponsePolicy
    from compact_farvector_runtime import FarVectorRuntime
    from farvector_time_runtime import FarVectorTimeRuntime
    from compact_research_eval import CompleteChecks
    from corner_deficit_objective import CornerDeficitCost
    from learning_update_guided_batchreset import optimizers, update
    started = time.perf_counter()
    deadline = started + 1570.
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    checks = CompleteChecks(STAGE, deadline, runtime_class=FarVectorRuntime)
    initial = initial_path()
    saved = torch.load(initial, map_location='cpu', weights_only=False)
    policy = FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(saved['policy'])
    with torch.no_grad():
        policy.log_std.copy_(torch.log(torch.tensor([.0017,.015,.02])))
    optimizer = optimizers(policy)
    envs = {k:FarVectorTimeRuntime(r,potential_speed=100.) for k,r in checks.references.items()}
    contexts = torch.load(CONTEXT_SOURCE/'all34_contexts.pt',map_location='cpu',weights_only=False)
    audit = json.loads((CONTEXT_SOURCE/'context_collection_audit.json').read_text(encoding='utf-8'))
    assert audit['bitwise_full_trace_reproduction'] and sha(CONTEXT_SOURCE/'all34_contexts.pt')==audit['contexts_sha256']
    for context in contexts:
        envs['original'].restore(context['state'])
        assert np.array_equal(PreviewCueRuntime.observation(envs['original']),context['observation'])
    corner_cost = CornerDeficitCost(checks.references['original'],checks.geometry['original'])
    teacher_saved = torch.load(TEACHER_INITIAL,map_location='cpu',weights_only=False)
    teachers = {}
    for target in json.loads(TARGETS.read_text(encoding='utf-8'))['selected']:
        teacher = JointResponsePolicy(feed_gain=target['feed_gain'],preview_gain=target['preview_gain'])
        teacher.load_state_dict(teacher_saved['policy'])
        teachers[target['id']] = teacher.eval().requires_grad_(False)
    dense_teacher = FineCompactPolicy(feed_gain=1.)
    dense_teacher.load_state_dict(teacher_saved['policy'])
    dense_teacher.eval().requires_grad_(False)
    write(STAGE/'context_reuse_audit.json',dict(source=str(CONTEXT_SOURCE/'all34_contexts.pt'),
        source_sha256=sha(CONTEXT_SOURCE/'all34_contexts.pt'),old_observation_exact_restore=True,
        scope='Frozen witnessed resets;new encoder has different feature5. No new context-generation simulation.'))
    updates, evaluations, episodes = [], [], []
    steps = count = cursor = 0
    episode_counts = dict.fromkeys(CASE_BATCHES,0)
    current_corner = ''

    def save(label):
        path = STAGE/(label+'.pt')
        torch.save(dict(policy=policy.state_dict(),optimizers={k:v.state_dict() for k,v in optimizer.items()},
            environments={k:v.state() for k,v in envs.items()},steps=steps,updates=count,context_cursor=cursor,
            episode_counts=episode_counts.copy(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),interface=[14,3,SEED],policy_class='FineCompactPolicy',feed_gain=1.,
            observation_features=envs['dense'].feature_names(),observation_dimension=14,policy_memory_scalars=0,
            actor_outputs=3,runtime_class='FarVectorRuntime',initial_checkpoint=str(initial),
            initial_checkpoint_sha256=sha(initial),inherited_ppo_updates=88,offline_initialization=str(PREPARATION)),path)
        return path

    save('initial')
    while count < 24 and time.perf_counter() < deadline-240.:
        keys=['physical_time_reward','corner_penalty_s','corner_deficit_mm','corner_budget_mm','teacher_target',
              'observation','next_observation','latent','log_probability','value','reward','next_value','terminal','segment_end']
        batch={k:[] for k in keys}
        case_ids, corner_ids, held = [], [], []
        collected=time.perf_counter()
        for case, size in CASE_BATCHES.items():
            for j in range(size):
                if case=='original' and (j%32==0 or envs[case].done):
                    context=contexts[cursor%len(contexts)]
                    cursor+=1
                    envs[case].restore(context['state'])
                    current_corner=context['id']
                    corner_cost.reset(current_corner)
                env=envs[case]
                x=env.observation()
                old_x=PreviewCueRuntime.observation(env)
                z,logp,value=policy.act(x)
                x2,reward,done,info=env.step(z)
                physical=reward
                penalty=0.
                constraint=dict(deficit_mm=0.,budget_mm=0.)
                if case=='original':
                    penalty,constraint=corner_cost.consume(info['rows'])
                    reward-=penalty
                teacher=teachers[current_corner] if case=='original' else dense_teacher
                with torch.no_grad():
                    next_value=0. if done else float(policy.value(torch.as_tensor(x2)))
                    target=np.clip(teacher.actor(torch.as_tensor(old_x)).numpy(),-1.,1.)
                    if target[1]>=1.-1e-6:
                        target[1]=1.04
                values=[physical,penalty,constraint['deficit_mm'],constraint['budget_mm'],target,x,x2,z,logp,value,
                        reward,next_value,done,j==size-1 or (case=='original' and j%32==31)]
                for key,value in zip(keys,values):
                    batch[key].append(value)
                case_ids.append(case)
                corner_ids.append(current_corner if case=='original' else '')
                held.append(info['held_action'])
                if done:
                    episode_counts[case]+=1
                    record=dict(case=case,episode=episode_counts[case],step=steps+len(case_ids),
                        success=env.success,reason=env.reason,prefix_s=env.ticks*.001,
                        complete_time_s=env.ticks*.001 if env.success else None,gates=env.tracker.frontier)
                    episodes.append(record)
                    with (STAGE/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(record)+'\n')
                    if case=='dense':
                        envs[case]=FarVectorTimeRuntime(checks.references[case],potential_speed=100.)
        assert Counter(case_ids)==CASE_BATCHES
        assert np.allclose(np.asarray(batch['reward'])+np.asarray(batch['corner_penalty_s']),batch['physical_time_reward'],atol=1e-12,rtol=0)
        np.savez_compressed(STAGE/f'batch_{count+1:04d}.npz',**{k:np.asarray(v) for k,v in batch.items()},
            case=np.asarray(case_ids),corner_id=np.asarray(corner_ids),held_action=np.asarray(held))
        collected_wall=time.perf_counter()-collected
        stats=update(policy,optimizer,batch)
        count+=1
        steps+=2048
        stats.update(update=count,steps=steps,collect_wall_s=collected_wall,episodes=episode_counts.copy(),
                     corner_penalty_total_s=float(np.sum(batch['corner_penalty_s'])))
        updates.append(stats)
        write(STAGE/'updates.json',updates)
        checkpoint=save(f'checkpoint_{count:04d}')
        print('UPDATE',count,'KL',stats['post_update_exact_kl'],'episodes',episode_counts,'wall',time.perf_counter()-started,flush=True)
        if count in [8,16,24]:
            record=checks.run(policy,checkpoint,count,new_ppo_updates=count)
            results=record['results']
            eligible=all(r['success'] for r in results.values())
            if eligible:
                comparisons=results['original']['corner_comparisons']
                eligible=bool(results['original']['complete_time_s']<=444.296+1e-9 and
                    results['dense']['complete_time_s']<=15.06985+1e-9 and
                    comparisons['historical_main']['not_worse']>=23 and
                    comparisons['S2']['not_worse']>=28 and comparisons['historical_strong_fixed']['not_worse']>=27 and
                    comparisons['historical_main']['total_change_mm']<=-1.18794470955+1e-7 and
                    (comparisons['historical_main']['not_worse']>=25 or
                     results['original']['complete_time_s']<444.276-1e-9 or results['dense']['complete_time_s']<14.996-1e-9))
            record['eligible_for_refinement']=eligible
            evaluations.append(record)
            write(STAGE/'evaluations.json',evaluations)
        write(STAGE/'live.json',dict(status='training',updates=count,steps=steps,
            wall_s=time.perf_counter()-started,goal_complete=False))
    save('final')
    eligible=[r for r in evaluations if r['eligible_for_refinement']]
    best=min(eligible,key=lambda r:(-r['results']['original']['corner_comparisons']['historical_main']['not_worse'],
        r['results']['original']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(status='step_budget_completed' if count==24 else 'wall_budget_completed',
        training_kind='Fresh on-policy PPO with the unchanged guided update and training-only individual corner cost,using new14feature encoder',
        ppo_updates_this_round=count,new_decisions=steps,inherited_ppo_updates=88,initial_checkpoint=str(initial),
        initial_checkpoint_sha256=sha(initial),evaluations=evaluations,selected_for_refinement=best,
        adopted=False,retained_checkpoint=str(ROOT/'direct14_guided_batchreset_ppo_seed922302_round17/checkpoint_0024.pt'),
        episode_counts=episode_counts,episode_reasons=dict(Counter(r['reason'] for r in episodes)),
        wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
    print('ROUND_FINISHED',count,'eligible',len(eligible),flush=True)


def main():
    if '--worker' in sys.argv:
        worker()
        return
    manifest=verify_bound_inputs()
    assert json.loads((PREPARATION/'receipt.json').read_text(encoding='utf-8'))['exit_code']==0
    initial=initial_path()
    STAGE.mkdir(exist_ok=False)
    (STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/name for name in ['farvector_time_runtime.py','compact_farvector_runtime.py',
        'compact_research_eval.py','learning_update_guided_batchreset.py','learning_update_guided_mc.py',
        'corner_deficit_objective.py','compact_fine_policy.py','compact_joint_response.py',
        'previewcue_time_runtime.py','previewcue_learning_runtime.py','compact_bounded_failure_runtime.py',
        'compact_speed_deficit_runtime.py','compact_previewcue_runtime.py','direct_compact_runtime.py','compact_control.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        wall_budget_s=1800,worker_deadline_s=1570,child_timeout_s=1700,source_sha256=hashes,
        bound_manifest=manifest,initial_checkpoint=str(initial),initial_checkpoint_sha256=sha(initial),seed=SEED,
        observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,ppo_updates=24,new_decisions=49152,
        case_batch_sizes=CASE_BATCHES,full_checks=[8,16,24],exploration_std=[.0017,.015,.02],
        evidence_gap='Old encoder corner-cost PPO increased original corner count but lost dense completion. New encoder has passed preservation checks; test fresh PPO without changing the objective or auxiliary teachers.',
        procedure='Same round18 training-only20times corner-excess deficit/100,physical-time reward,potential speed100,KLguard0.03,batch proposal rate reset3e-4,and auxiliary action loss0.5. Teachers receive old observations on current training states only;student and complete checks use new14features only.',
        teacher_checkpoint=str(TEACHER_INITIAL),teacher_sha256=sha(TEACHER_INITIAL),
        target_sha256=sha(TARGETS),context_sha256=sha(CONTEXT_SOURCE/'all34_contexts.pt'),
        adoption='Intermediate screen equals round18: both successes,times<=444.296/15.06985,corner counts>=23/28/27,total historical deficit change<=-1.18794471,and either>=25historical corners or>20ms time improvement. Refined resolution and final all34,time,repeats still required. No automatic final adoption.',
        stop='24updates or wall deadline. Preserve all failed evaluations. Any qualifying result needs a separately budgeted fine-resolution verification.',
        goal_complete=False,method_frozen=False))
    started=time.perf_counter()
    code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1700)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:
            status='timeout'
    verify_bound_inputs()
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:
        print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':
    main()
