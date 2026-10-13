"""Finite output-response diagnosis from witnessed states and complete cases."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from evaluate_historical_dense import ROOT,BASE,CASE,HIST,sha,write,verify_bound_inputs

STAGE=ROOT/'direct14_feed_response_probe_seed922302_round8'
MODEL=ROOT/'direct14_speed_deficit_ppo_seed922302_round5/checkpoint_0024.pt'
CONTEXTS=ROOT/'direct14_all34_context_ppo_seed922302_round6/all34_contexts.pt'
GAINS=[1.,1.25,1.5,2.]


def worker():
    import numpy as np
    import torch
    from compact_feed_response import CalibratedFeedPolicy
    from direct_compact_runtime import DirectCompactRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics,ordered_projection
    from analyze_dense_trace import analyze
    torch.set_num_threads(1)
    start=time.perf_counter();deadline=start+1650
    cases={'original':BASE/'paper_evidence_review/natural_closed_case','dense':CASE}
    refs={k:np.loadtxt(p/'reference.csv',delimiter=',',skiprows=1) for k,p in cases.items()}
    geom={k:json.loads((p/'geometry.json').read_text(encoding='utf8')) for k,p in cases.items()}
    corners={c['id']:c for c in geom['original']['sharp_corners']}
    contexts=torch.load(CONTEXTS,weights_only=False)
    saved=torch.load(MODEL,weights_only=False)
    records=[];baseline=None;evaluations=[]

    def full(policy,label,case):
        trace=STAGE/f'{label}_{case}.npz'
        env=DirectCompactRuntime(refs[case]);r=evaluate(policy,env,trace,deadline)
        with np.load(trace) as data:rows=data['rows']
        if len(rows):
            r['analytic']=analytic_command_audit(refs[case][0],rows,geom[case])
            if not r['analytic']['analytic_corridor_pass']:
                r.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        r.update(feed_gain=policy.feed_gain,source_checkpoint=str(MODEL),source_checkpoint_sha256=sha(MODEL),
            actor_outputs=3,observation_dimension=14,policy_memory_scalars=0,
            training_kind='Finite monotone output calibration, zero new PPO or fitting',goal_complete=False,method_frozen=False)
        write(trace.with_suffix('.json'),r);write(STAGE/f'{label}_{case}_gates.json',env.tracker.events)
        if r['success']:
            m=metrics(refs[case],rows,geom[case]) if case=='original' else analyze(trace,trace.with_suffix('.json'))
            write(STAGE/f'{label}_{case}_metrics.json',m)
        print('FULL_CHECK',label,case,r['success'],r['complete_time_s'],r['reason'],flush=True)
        return r

    for gain in GAINS:
        if time.perf_counter()>deadline-300:break
        policy=CalibratedFeedPolicy(gain);policy.load_state_dict(saved['policy'])
        label=f'gain_{gain:g}';folder=STAGE/label;folder.mkdir()
        values=[]
        for context in contexts:
            if time.perf_counter()>deadline-300:raise TimeoutError('Registered local screening deadline')
            env=DirectCompactRuntime(refs['original']);env.restore(context['state'])
            assert np.array_equal(env.observation(),context['observation'])
            chunks=[];actions=[];observations=[]
            for _ in range(48):
                x=env.observation();z=policy.act(x,True)[0]
                _,_,_,info=env.step(z);chunks.append(info['rows']);actions.append(info['held_action']);observations.append(x)
                if env.done:break
            rows=np.vstack(chunks);c=corners[context['id']]
            p=ordered_projection(refs['original'],rows[:,:2]);sv=np.linalg.norm(rows[:,2:4],axis=1)
            mask=(p>=c['reference_s_mm']-c['window_before_mm'])&(p<=c['reference_s_mm']+c['window_after_mm'])
            complete=bool(p.min()<c['reference_s_mm']-c['window_before_mm'] and p.max()>c['reference_s_mm']+c['window_after_mm'])
            analytic=analytic_command_audit(context['state']['physical']['position'],rows,geom['original'])
            valid=bool(not env.done and complete and analytic['analytic_corridor_pass'])
            np.savez_compressed(folder/(context['id']+'.npz'),rows=rows,actions=np.array(actions),observations=np.array(observations))
            values.append(dict(id=context['id'],local_valid=valid,reason=env.reason,
                window_complete=complete,deficit_mm=float(np.maximum(100-sv[mask],0).sum()*.001),
                maximum_distance_mm=analytic['maximum_evaluated_distance_mm'],
                scope='Witnessed-state local probe, not a complete-task result'))
        if gain==1.:
            baseline=values
            expected=json.loads((ROOT/'direct14_speed_deficit_ppo_seed922302_round5/eval_0024_original_metrics.json').read_text(encoding='utf8'))
            assert all(v['local_valid'] for v in values)
            error=max(abs(a['deficit_mm']-b['speed_deficit_mm']) for a,b in zip(values,expected['corners']))
            assert error<1e-9,('Nominal replay must match full incumbent windows',error)
        delta=[v['deficit_mm']-b['deficit_mm'] for v,b in zip(values,baseline)]
        record=dict(gain=gain,corners=values,local_all34_valid=all(v['local_valid'] for v in values),
            local_nonworse_vs_initial=sum(d<=1e-7 for d in delta),local_total_deficit_change_mm=sum(delta),
            local_max_deficit_increase_mm=max(delta))
        write(folder/'local_result.json',record)
        if gain>1 and record['local_all34_valid'] and sum(delta)<-1e-4:
            record['dense']=full(policy,label,'dense')
        records.append(record);write(STAGE/'screening.json',records)
        print('LOCAL_SCREEN',gain,record['local_all34_valid'],record['local_nonworse_vs_initial'],sum(delta),flush=True)
    eligible=[r for r in records if r.get('dense',{}).get('success') and r['dense']['complete_time_s']<=15.076]
    best=min(eligible,key=lambda r:(r['local_total_deficit_change_mm'],r['dense']['complete_time_s'])) if eligible else None
    if best:
        gain=best['gain'];policy=CalibratedFeedPolicy(gain);policy.load_state_dict(saved['policy'])
        original=full(policy,f'gain_{gain:g}','original')
        best['original']=original
        write(STAGE/'selected_policy.json',dict(source_checkpoint=str(MODEL),source_checkpoint_sha256=sha(MODEL),feed_gain=gain,
            observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,new_ppo_updates=0,
            output_transform='z_feed + (gain-1)*max(z_feed-0.6,0), then existing common action clipping'))
    write(STAGE/'result.json',dict(status='registered_probe_completed',candidates=records,selected_full_candidate=best,
        new_ppo_updates=0,adopted=False,retained_checkpoint=str(MODEL),goal_complete=False,method_frozen=False,
        wall_s=time.perf_counter()-start,remaining='Full joint protocol and refined resolution still required before adoption'))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs()
    assert sha(MODEL)=='42f758ed83ccde810adbb31592c2c9ce5996883f0a51d066c2cd379625220ec5'
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'compact_feed_response.py',ROOT/'compact_control.py',ROOT/'direct_compact_runtime.py',
        BASE/'paper_complete_evaluation.py',BASE/'natural_case_metrics.py',BASE/'natural_case_audit.py',ROOT/'analyze_dense_trace.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Finite feed response diagnostic parameter optimization, not PPO',wall_budget_s=1800,child_timeout_s=1740,
        worker_deadline_s=1650,gains=GAINS,feed_response_threshold_mm_s=80,
        original_contexts=str(CONTEXTS),original_contexts_sha256=sha(CONTEXTS),source_sha256=hashes,source_model=str(MODEL),
        source_model_sha256=sha(MODEL),source_manifest=manifest,ppo_updates=0,
        evidence_gap='Repeated compact PPO has not improved complete joint objectives. Saved-data audit finds broad-angle feed under-request and changes in path length. Isolate the high-feed response, preserving learned low-feed, steering and preview outputs.',
        procedure='Four registered gains including exact incumbent replay on all34 witnessed approach states,48decisions per window. For valid reduced-total-deficit gains only, full dense runs; select one by local deficit then dense time for full original evaluation.',
        adoption='No automatic adoption from local probes; complete original/dense physical audits and final all34/time protocol plus finer resolution are required. Current incumbent retained until then.',
        stop='Registered finite candidate list or wall budget; failures and prefixes preserved. No baseline/window/physical-limit changes.',
        policy_inputs=14,policy_memory=0,goal_complete=False,method_frozen=False))
    start=time.perf_counter();code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf8') as out,(STAGE/'stderr.txt').open('w',encoding='utf8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1740)
            code=run.returncode;status='finished'
        except subprocess.TimeoutExpired:status='timeout'
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf8')[-5000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf8')[-3000:])


if __name__=='__main__':main()
