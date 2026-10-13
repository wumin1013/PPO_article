"""PPO refinement preserving the incumbent's learned steering feedback exactly."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs

STAGE=ROOT/'direct14_coherent_head_ppo_round28'
SOURCE=ROOT/'direct14_farvector_local_feedback_fit_round21'
INITIAL=SOURCE/'fit_08000.pt'
SEED=922302
CASE_BATCHES={'original':1536,'dense':512}


def worker():
    import numpy as np
    import torch
    from collections import Counter
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from farvector_time_runtime import FarVectorTimeRuntime
    from compact_research_eval import CompleteChecks
    from corner_deficit_objective import CornerDeficitCost
    from saved_local_contexts import from_complete_trace
    from learning_update_coherent_anchor import optimizers,update
    started=time.perf_counter();deadline=started+1640.
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    checks=CompleteChecks(STAGE,deadline,runtime_class=FarVectorRuntime)
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False)
    policy=FineCompactPolicy(feed_gain=1.);policy.load_state_dict(saved['policy'])
    teacher=FineCompactPolicy(feed_gain=1.);teacher.load_state_dict(saved['policy'])
    teacher.eval().requires_grad_(False)
    # This stage changes only the feed and preview output rows. The complete
    # dynamic steering function and shared learned features are preserved.
    for parameter in policy.actor.parameters():parameter.requires_grad_(False)
    for parameter in policy.actor[4].parameters():parameter.requires_grad_(True)
    weight_mask=torch.ones_like(policy.actor[4].weight);weight_mask[0]=0.
    bias_mask=torch.tensor([0.,1.,1.])
    policy.actor[4].weight.register_hook(lambda gradient:gradient*weight_mask)
    policy.actor[4].bias.register_hook(lambda gradient:gradient*bias_mask)
    with torch.no_grad():policy.log_std.copy_(torch.log(torch.tensor([.0007,.008,.015])))
    policy.log_std.requires_grad_(False)
    optimizer=optimizers(policy)
    ref,expected,contexts,audit=from_complete_trace(SOURCE/'eval_8000_original.npz',
        SOURCE/'eval_8000_original_gates.json',checks.geometry['original'])
    local_x=[];replay_errors=[];context_ticks=0
    for context in contexts:
        env=FarVectorRuntime(ref);env.restore(context['state']);chunks=[]
        for _ in range(64):
            x=env.observation();local_x.append(x)
            _,_,_,info=env.step(teacher.act(x,True)[0]);chunks.append(info['rows'])
            assert not env.done
        rows=np.vstack(chunks);context_ticks+=len(rows)
        error=float(np.max(abs(rows[:,:4]-expected[context['tick']:context['tick']+len(rows),:4])))
        assert error<2e-5,(context['id'],error)
        replay_errors.append(error)
    torch.save(contexts,STAGE/'all34_contexts.pt')
    audit.update(maximum_replay_xyv_error=max(replay_errors),new_validation_ticks=context_ticks,
        contexts_sha256=sha(STAGE/'all34_contexts.pt'),all34_unmodified_local_checks_pass=True)
    write(STAGE/'context_audit.json',audit)
    retained=[np.asarray(local_x)]
    anchor_sources={}
    for path in [SOURCE/'eval_8000_dense.npz',ROOT/'direct14_farvector_local_feedback_fit_round21_dense_half_ms/trace.npz']:
        anchor_sources[str(path)]=sha(path)
        with np.load(path) as data:retained.append(data['observations'].copy())
    anchor_x=torch.tensor(np.vstack(retained),dtype=torch.float32)
    with torch.no_grad():anchor_y=teacher.actor(anchor_x)
    anchors=(anchor_x,anchor_y)
    np.savez_compressed(STAGE/'auxiliary_anchors.npz',observations=anchor_x.numpy(),targets=anchor_y.numpy())
    write(STAGE/'anchor_audit.json',dict(samples=len(anchor_x),source_sha256=anchor_sources,
        teacher_checkpoint=str(INITIAL),teacher_sha256=sha(INITIAL),
        scope='Auxiliary regularization only;not fresh PPO samples or independent experiments'))
    envs={k:FarVectorTimeRuntime(r,potential_speed=100.) for k,r in checks.references.items()}
    cost=CornerDeficitCost(ref,checks.geometry['original'])
    count=steps=cursor=0;current_corner='';updates=[];evaluations=[];episodes=[]

    def save(label):
        path=STAGE/(label+'.pt')
        torch.save(dict(policy=policy.state_dict(),optimizers={k:v.state_dict() for k,v in optimizer.items()},
            environments={k:v.state() for k,v in envs.items()},steps=steps,updates=count,context_cursor=cursor,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            interface=[14,3,SEED],feed_gain=1.,policy_class='FineCompactPolicy',runtime_class='FarVectorRuntime',
            observation_dimension=14,actor_outputs=3,policy_memory_scalars=0,
            observation_features=envs['dense'].feature_names(),initial_checkpoint=str(INITIAL),
            initial_checkpoint_sha256=sha(INITIAL),inherited_ppo_updates=88,new_ppo_updates=count,
            trainable_scope='Feed and preview rows of final actor layer plus separate critic;learned steering and shared features frozen'),path)
        return path

    save('initial')
    while count<12 and time.perf_counter()<deadline-260.:
        batch={k:[] for k in ['observation','next_observation','latent','log_probability','value','reward',
            'next_value','terminal','segment_end','teacher_target','physical_reward','corner_penalty_s']}
        cases=[];corner_ids=[];collected=time.perf_counter()
        for case,size in CASE_BATCHES.items():
            for j in range(size):
                if case=='original' and (j%64==0 or envs[case].done):
                    context=contexts[cursor%len(contexts)];cursor+=1
                    envs[case].restore(context['state']);current_corner=context['id'];cost.reset(current_corner)
                env=envs[case];x=env.observation();z,logp,value=policy.act(x)
                x2,reward,done,info=env.step(z);physical=reward;penalty=0.
                if case=='original':penalty,_=cost.consume(info['rows']);reward-=penalty
                with torch.no_grad():
                    next_value=0. if done else float(policy.value(torch.as_tensor(x2)))
                    target=teacher.actor(torch.as_tensor(x)).numpy()
                values=[x,x2,z,logp,value,reward,next_value,done,j==size-1 or(case=='original' and j%64==63),target,physical,penalty]
                for key,item in zip(batch,values):batch[key].append(item)
                cases.append(case);corner_ids.append(current_corner if case=='original' else '')
                if done:
                    episodes.append(dict(case=case,update=count+1,success=env.success,reason=env.reason,
                        prefix_s=env.ticks*.001,complete_time_s=env.ticks*.001 if env.success else None))
                    if case=='dense':envs[case]=FarVectorTimeRuntime(checks.references[case],potential_speed=100.)
        assert Counter(cases)==CASE_BATCHES
        assert np.allclose(np.asarray(batch['reward'])+np.asarray(batch['corner_penalty_s']),batch['physical_reward'],atol=1e-12,rtol=0)
        np.savez_compressed(STAGE/f'batch_{count+1:04d}.npz',**{k:np.asarray(v) for k,v in batch.items()},
            case=np.asarray(cases),corner_id=np.asarray(corner_ids))
        collect_wall=time.perf_counter()-collected
        stats=update(policy,optimizer,batch,anchors,kl_limit=.01)
        for key,value in saved['policy'].items():
            if key.startswith('actor.') and key not in ['actor.4.weight','actor.4.bias']:
                assert torch.equal(policy.state_dict()[key],value),key
        assert torch.equal(policy.actor[4].weight[0],saved['policy']['actor.4.weight'][0])
        assert torch.equal(policy.actor[4].bias[0],saved['policy']['actor.4.bias'][0])
        count+=1;steps+=sum(CASE_BATCHES.values())
        stats.update(update=count,steps=steps,collect_wall_s=collect_wall,
            frozen_steering_and_features_bitwise_unchanged=True,corner_penalty_s=float(np.sum(batch['corner_penalty_s'])))
        updates.append(stats);write(STAGE/'updates.json',updates);write(STAGE/'episodes.json',episodes)
        checkpoint=save(f'checkpoint_{count:04d}')
        print('PPO_UPDATE',count,'KL',stats['post_update_exact_kl'],'wall',time.perf_counter()-started,flush=True)
        if count in [4,8,12]:
            record=checks.run(policy,checkpoint,count,new_ppo_updates=count)
            r=record['results'];eligible=all(v['success'] for v in r.values())
            if eligible:
                c=r['original']['corner_comparisons']
                eligible=bool(r['original']['complete_time_s']<=444.236+1e-9 and r['dense']['complete_time_s']<=15.06985+1e-9 and
                    c['historical_main']['not_worse']>=30 and c['S2']['not_worse']==34 and c['historical_strong_fixed']['not_worse']>=33 and
                    c['historical_main']['total_change_mm']<=-4.42375694+1e-7 and
                    (c['historical_main']['not_worse']>=31 or r['original']['complete_time_s']<444.216-1e-9))
            record['eligible_for_refinement']=eligible;evaluations.append(record);write(STAGE/'evaluations.json',evaluations)
    save('final')
    eligible=[r for r in evaluations if r['eligible_for_refinement']]
    best=min(eligible,key=lambda r:(-r['results']['original']['corner_comparisons']['historical_main']['not_worse'],
        r['results']['original']['complete_time_s'])) if eligible else None
    write(STAGE/'result.json',dict(ppo_updates_this_round=count,new_decisions=steps,inherited_ppo_updates=88,
        evaluations=evaluations,selected_for_refinement=best,wall_s=time.perf_counter()-started,
        training_kind='Fresh on-policy PPO;single coherent incumbent anchors;frozen learned steering and features',
        adopted=False,retained_checkpoint=str(INITIAL),goal_complete=False,method_frozen=False))
    print('ROUND_FINISHED',count,'eligible',len(eligible),flush=True)


def main():
    if '--worker' in sys.argv:
        worker();return
    verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['learning_update_coherent_anchor.py','learning_update_compact_mc.py',
        'learning_update_guarded_mc.py','saved_local_contexts.py','corner_deficit_objective.py',
        'compact_fine_policy.py','compact_farvector_runtime.py','farvector_time_runtime.py','compact_research_eval.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Actual PPO refinement from the verified compact incumbent',wall_budget_s=1800,worker_deadline_s=1640,child_timeout_s=1740,
        source_sha256=hashes,checkpoint=str(INITIAL),checkpoint_sha256=sha(INITIAL),seed=SEED,
        updates=12,new_decisions=24576,batch_sizes=CASE_BATCHES,local_horizon_decisions=64,
        exploration_std=[.0007,.008,.015],post_step_kl_limit=.01,full_checks=[4,8,12],
        evidence_gap='Local teacher fitting twice failed closed-loop adoption;feedback conflicts and unintended steering drift must be avoided.',
        procedure='Preserve verified learned steering and shared features bitwise. Fresh PPO optimizes130feed/preview output parameters and the critic. Same time reward and34corner-excess cost;only one incumbent teacher supplies explicit action-equivalent auxiliary targets on visited states and both dense-resolution anchors.',
        adoption='Both complete nominal tasks;no loss of existing30/34/33corner counts or total deficit gain;at least one extra historical corner or>20ms original gain. Then both independent0.5ms checks. Final all34,credible time,matched two-action and repeats remain required.',
        stop='Twelve updates or registered deadline;retain failures and verified round21if no paired improvement.',
        goal_complete=False,method_frozen=False))
    started=time.perf_counter();code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1740)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status='timeout'
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':main()
