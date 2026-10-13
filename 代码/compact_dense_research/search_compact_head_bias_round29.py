"""Finite two-parameter optimization of one14input network,without new rules."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,CASE,write,sha,verify_bound_inputs

STAGE=ROOT/'direct14_head_bias_search_round29'
SOURCE=ROOT/'direct14_farvector_local_feedback_fit_round21'
INITIAL=SOURCE/'fit_08000.pt'
BIAS_PAIRS=list(itertools.product([.01,.02,.04],[0.,.02,.05,.10]))


def worker():
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection,metrics
    from saved_local_contexts import from_complete_trace
    started=time.perf_counter();deadline=started+1640.;torch.set_num_threads(1)
    folders={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
    refs={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in folders.items()}
    geometry={k:json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in folders.items()}
    ref,expected,contexts,context_audit=from_complete_trace(SOURCE/'eval_8000_original.npz',SOURCE/'eval_8000_original_gates.json',geometry['original'])
    replay_path=ROOT/'direct14_coherent_head_ppo_round28/context_audit.json'
    replay=json.loads(replay_path.read_text(encoding='utf-8'))
    assert replay['trace_sha256']==context_audit['trace_sha256'] and replay['all34_unmodified_local_checks_pass']
    assert replay['maximum_replay_xyv_error']<2e-5
    context_audit['validated_unmodified_replay']=dict(path=str(replay_path),sha256=sha(replay_path))
    write(STAGE/'context_audit.json',context_audit)
    protocol=json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
    old={k:json.loads(Path(v['path']).read_text(encoding='utf-8')) for k,v in protocol['original']['comparators'].items()}
    budgets={c['id']:min(m['corners'][i]['speed_deficit_mm'] for m in old.values()) for i,c in enumerate(geometry['original']['sharp_corners'])}
    baseline_metric=json.loads((SOURCE/'eval_8000_original_metrics.json').read_text(encoding='utf-8'))
    baseline_deficits={c['id']:c['speed_deficit_mm'] for c in baseline_metric['corners']}
    original_saved=torch.load(INITIAL,map_location='cpu',weights_only=False)
    records=[];full=[]

    def check_case(policy,checkpoint,case,trace):
        env=FarVectorRuntime(refs[case])
        result=evaluate(policy,env,trace,deadline)
        with np.load(trace) as d:rows=d['rows']
        result['analytic']=analytic_command_audit(refs[case][0],rows,geometry[case])
        if not result['analytic']['analytic_corridor_pass']:
            result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,
            observation_features=env.feature_names(),new_ppo_updates=0,
            training_kind='Finite offline head-bias parameter optimization,not PPO',goal_complete=False,method_frozen=False)
        write(trace.with_suffix('.json'),result);write(trace.parent/(trace.stem+'_gates.json'),env.tracker.events)
        if case=='original' and result['success']:
            m=metrics(refs[case],rows,geometry[case]);write(trace.parent/(trace.stem+'_metrics.json'),m)
            result['corner_comparisons']={}
            for name,b in old.items():
                delta=np.array([a['speed_deficit_mm']-v['speed_deficit_mm'] for a,v in zip(m['corners'],b['corners'])])
                result['corner_comparisons'][name]=dict(not_worse=int((delta<=1e-7).sum()),total_change_mm=float(delta.sum()),
                    maximum_increase_mm=float(delta.max()),losing_corners=[a['id'] for a,v in zip(m['corners'],delta) if v>1e-7])
            write(trace.with_suffix('.json'),result)
        return result

    for index,(feed,preview) in enumerate(BIAS_PAIRS):
        if time.perf_counter()>deadline-800.:break
        folder=STAGE/f'candidate_{index:02d}';folder.mkdir()
        policy=FineCompactPolicy(feed_gain=1.);policy.load_state_dict(original_saved['policy'])
        with torch.no_grad():
            policy.actor[4].bias[1]+=feed;policy.actor[4].bias[2]+=preview
        checkpoint=folder/'model.pt'
        torch.save(dict(policy=policy.state_dict(),interface=[14,3,922302],feed_gain=1.,
            runtime_class='FarVectorRuntime',policy_class='FineCompactPolicy',
            observation_features=FarVectorRuntime(refs['dense']).feature_names(),
            observation_dimension=14,actor_outputs=3,policy_memory_scalars=0,
            new_ppo_updates=0,inherited_ppo_updates=88,head_bias_delta=[0.,feed,preview],
            source_checkpoint=str(INITIAL),source_checkpoint_sha256=sha(INITIAL),
            training_kind='Finite offline parameter search'),checkpoint)
        dense=check_case(policy,checkpoint,'dense',folder/'dense.npz')
        record=dict(index=index,feed_bias=feed,preview_bias=preview,checkpoint=str(checkpoint),dense=dense,
            eligible_for_full_original=False)
        records.append(record)
        if not dense['success'] or dense['complete_time_s']>15.06985+1e-9:
            write(STAGE/'screening.json',records);print('BIAS_SCREEN',index,'dense_rejected',flush=True);continue
        half_name=f'{STAGE.name}_dense_half_{index:02d}'
        with (folder/'half_launch.txt').open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B',str(ROOT/'refine_farvector_candidate.py'),
                '--name',half_name,'--case','dense','--checkpoint',str(checkpoint),'--nominal',str(folder/'dense.json')],
                stdout=stream,stderr=subprocess.STDOUT,timeout=min(310.,deadline-time.perf_counter()))
        assert result.returncode==0
        half=json.loads((ROOT/half_name/'result.json').read_text(encoding='utf-8'));record['dense_half']=half
        if not half['success'] or half['complete_time_s']>15.1069+1e-9:
            write(STAGE/'screening.json',records);print('BIAS_SCREEN',index,'dense_half_rejected',flush=True);continue
        local=[]
        for context,corner in zip(contexts,geometry['original']['sharp_corners']):
            assert context['id']==corner['id']
            env=FarVectorRuntime(ref);env.restore(context['state']);start=env.env.state();chunks=[]
            for _ in range(64):
                _,_,_,info=env.step(policy.act(env.observation(),True)[0]);chunks.append(info['rows'])
                if env.done:break
            rows=np.vstack(chunks);p=ordered_projection(ref,rows[:,:2]);speed=np.linalg.norm(rows[:,2:4],axis=1)
            lo=corner['reference_s_mm']-10.;hi=corner['reference_s_mm']+10.;mask=(p>=lo)&(p<=hi)
            complete=bool(p.min()<lo and p.max()>hi)
            raw_a=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/.001
            raw_j=np.diff(np.vstack([start['acceleration'],raw_a]),axis=0)/.001
            peaks=[float(speed.max()),float(np.linalg.norm(raw_a,axis=1).max()),float(np.linalg.norm(raw_j,axis=1).max())]
            ga=analytic_command_audit(start['position'],rows,geometry['original'])
            valid=bool(not env.done and complete and ga['analytic_corridor_pass'] and np.all(np.asarray(peaks)<=[100.000001,2000.001,62500.02]))
            deficit=float(np.maximum(100-speed[mask],0).sum()*.001)
            local.append(dict(id=context['id'],valid=valid,deficit_mm=deficit,budget_mm=budgets[context['id']],
                deficit_change_mm=deficit-baseline_deficits[context['id']],budget_pass=deficit<=budgets[context['id']]+1e-7,
                maximum_distance_mm=ga['maximum_evaluated_distance_mm'],peaks=peaks))
            np.savez_compressed(folder/(context['id']+'.npz'),rows=rows,start_position=start['position'],
                initial_velocity=start['velocity'],initial_acceleration=start['acceleration'])
        record['local']=local;record['local_valid']=all(c['valid'] for c in local)
        record['local_budget_count']=sum(c['budget_pass'] for c in local)
        record['local_total_deficit_change_mm']=sum(c['deficit_change_mm'] for c in local)
        record['eligible_for_full_original']=bool(record['local_valid'] and record['local_budget_count']>=30 and record['local_total_deficit_change_mm']<0.)
        write(STAGE/'screening.json',records)
        print('BIAS_SCREEN',index,'both_dense_pass','local_count',record['local_budget_count'],'valid',record['local_valid'],flush=True)
    candidates=[r for r in records if r['eligible_for_full_original']]
    candidates.sort(key=lambda r:(-r['local_budget_count'],r['local_total_deficit_change_mm'],r['dense']['complete_time_s']))
    for record in candidates[:2]:
        if time.perf_counter()>deadline-260.:break
        checkpoint=Path(record['checkpoint']);policy=FineCompactPolicy(feed_gain=1.)
        policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'])
        result=check_case(policy,checkpoint,'original',checkpoint.parent/'original.npz')
        full.append(dict(index=record['index'],checkpoint=str(checkpoint),results=dict(original=result,dense=record['dense'],dense_half=record['dense_half'])))
        write(STAGE/'full_evaluations.json',full)
        print('BIAS_FULL_ORIGINAL',record['index'],result['success'],result['complete_time_s'],flush=True)
    write(STAGE/'result.json',dict(screened_candidates=len(records),screening=records,full_evaluations=full,
        new_ppo_updates=0,inherited_ppo_updates=88,wall_s=time.perf_counter()-started,
        adopted=False,retained_checkpoint=str(INITIAL),goal_complete=False,method_frozen=False,
        remaining='Any improved full candidate still needs independent original0.5ms and final joint audit;two-actionPPO and repeats also remain.'))


def main():
    if '--worker' in sys.argv:
        worker();return
    verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['compact_fine_policy.py','compact_farvector_runtime.py',
        'saved_local_contexts.py','refine_farvector_candidate.py']]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Finite offline two-parameter head optimization,not PPO',wall_budget_s=1800,worker_deadline_s=1640,child_timeout_s=1740,
        source_sha256=hashes,initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
        head_bias_pairs=BIAS_PAIRS,max_original_full_checks=2,new_ppo_updates=0,
        evidence_gap='Verified corner23requests can fall to99.34mm/s,disabling exact-cruise turn projection;test a shared feed/preview calibration without changing learned steering or observations.',
        procedure='Change only two biases inside the same14x64x64x3network. Both dense resolutions must pass first. Screen the same candidate on all34actual-arrival local resets,then execute up to two full original tasks. Local results never replace full checks.',
        adoption='No automatic adoption;both dense successes plus full original improvement must be followed by independent original0.5ms. Keep34corner and credible-time requirements unchanged.',
        stop='Twelve registered candidates,up to two full original checks,or wall deadline;preserve all failures and retain verified round21.',
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
