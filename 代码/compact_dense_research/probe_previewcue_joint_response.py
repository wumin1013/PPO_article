"""Bounded parameter diagnosis with paired local starts and full verification."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from evaluate_historical_dense import ROOT, BASE, CASE, sha, write, verify_bound_inputs

RESPONSES = [(1.,1.), (1.25,1.), (1.5,1.), (1.,1.05), (1.25,1.05),
             (1.25,1.10), (1.25,1.15), (1.5,1.05), (1.,1.10)]
CONTEXTS = ROOT/'direct14_previewcue_ppo_seed922302_round13/all34_contexts.pt'


def worker(args):
    import numpy as np
    import torch
    from compact_joint_response import JointResponsePolicy
    from compact_previewcue_runtime import PreviewCueRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics, ordered_projection
    from analyze_dense_trace import analyze
    torch.set_num_threads(1)
    stage = ROOT/args.name
    model = Path(args.checkpoint)
    started = time.perf_counter()
    deadline = started + 1650.
    cases = {'original': BASE/'paper_evidence_review/natural_closed_case', 'dense': CASE}
    refs = {k: np.loadtxt(p/'reference.csv', delimiter=',', skiprows=1) for k,p in cases.items()}
    geom = {k: json.loads((p/'geometry.json').read_text(encoding='utf-8')) for k,p in cases.items()}
    corners = {c['id']:c for c in geom['original']['sharp_corners']}
    protocol = json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
    historical = {k:json.loads(Path(v['path']).read_text(encoding='utf-8')) for k,v in protocol['original']['comparators'].items()}
    contexts = torch.load(CONTEXTS, map_location='cpu', weights_only=False)
    saved = torch.load(model, map_location='cpu', weights_only=False)
    screens = []
    evaluations = []

    def policy_for(feed, preview):
        policy = JointResponsePolicy(feed_gain=feed, preview_gain=preview)
        policy.load_state_dict(saved['policy'])
        return policy

    def full(policy, label, case):
        trace = stage/f'{label}_{case}.npz'
        env = PreviewCueRuntime(refs[case])
        result = evaluate(policy, env, trace, deadline)
        with np.load(trace) as data:
            rows = data['rows']
        if len(rows):
            result['analytic'] = analytic_command_audit(refs[case][0], rows, geom[case])
            if not result['analytic']['analytic_corridor_pass']:
                result.update(success=False, complete_time_s=None, motion_scope='failure_prefix_only')
        result.update(source_checkpoint=str(model), source_checkpoint_sha256=sha(model),
                      feed_gain=policy.feed_gain, preview_gain=policy.preview_gain,
                      actor_outputs=3, observation_dimension=14, policy_memory_scalars=0,
                      new_ppo_updates=0, goal_complete=False, method_frozen=False)
        write(trace.with_suffix('.json'), result)
        write(stage/f'{label}_{case}_gates.json', env.tracker.events)
        if result['success']:
            value = metrics(refs[case],rows,geom[case]) if case=='original' else analyze(trace,trace.with_suffix('.json'))
            write(stage/f'{label}_{case}_metrics.json', value)
            if case=='original':
                result['corner_comparisons'] = comparisons(value['corners'], 'speed_deficit_mm')
                write(trace.with_suffix('.json'),result)
        print('FULL_CHECK',label,case,result['success'],result['complete_time_s'],result['reason'],flush=True)
        return result

    def comparisons(values, key):
        result = {}
        for name,old in historical.items():
            delta = [a[key]-b['speed_deficit_mm'] for a,b in zip(values,old['corners'])]
            result[name] = dict(not_worse=sum(d<=1e-7 for d in delta), total_change_mm=sum(delta), maximum_increase_mm=max(delta))
        return result

    for idx,(feed,preview) in enumerate(RESPONSES):
        if time.perf_counter()>deadline-620.:
            break
        policy = policy_for(feed, preview)
        label = f'response_{idx:02d}'
        folder = stage/label
        folder.mkdir()
        values = []
        for context in contexts:
            env = PreviewCueRuntime(refs['original'])
            env.restore(context['state'])
            assert np.array_equal(env.observation(),context['observation'])
            chunks=[]
            actions=[]
            for decision in range(48):
                action = policy.act(env.observation(),True)[0]
                _,_,_,info = env.step(action)
                chunks.append(info['rows'])
                actions.append(info['held_action'])
                if env.done:
                    break
            rows = np.vstack(chunks)
            c = corners[context['id']]
            p = ordered_projection(refs['original'],rows[:,:2])
            speed = np.linalg.norm(rows[:,2:4],axis=1)
            low,high = c['reference_s_mm']-c['window_before_mm'],c['reference_s_mm']+c['window_after_mm']
            mask = (p>=low)&(p<=high)
            complete = bool(p.min()<low and p.max()>high)
            audit = analytic_command_audit(context['state']['physical']['position'],rows,geom['original'])
            valid = bool(not env.done and complete and audit['analytic_corridor_pass'])
            pmono = np.maximum.accumulate(p)
            crossing = np.interp([low,high],pmono,np.arange(1,len(rows)+1)*.001)
            np.savez_compressed(folder/(context['id']+'.npz'),rows=rows,actions=np.asarray(actions))
            values.append(dict(id=context['id'],local_valid=valid,reason=env.reason,
                window_complete=complete,deficit_mm=float(np.maximum(100-speed[mask],0).sum()*.001),
                window_time_s=float(crossing[1]-crossing[0]) if complete else None,
                maximum_distance_mm=audit['maximum_evaluated_distance_mm']))
        record=dict(index=idx,label=label,feed_gain=feed,preview_gain=preview,corners=values,
            local_all34_valid=all(v['local_valid'] for v in values),
            local_total_deficit_mm=sum(v['deficit_mm'] for v in values),
            local_total_window_time_s=sum(v['window_time_s'] or 0. for v in values),
            local_comparisons=comparisons(values,'deficit_mm'),
            scope='Paired witnessed-state probes; these are not complete-task results')
        write(folder/'local_result.json',record)
        if record['local_all34_valid']:
            record['dense'] = full(policy,label,'dense')
        screens.append(record)
        write(stage/'screening.json',screens)
        print('LOCAL_SCREEN',idx,feed,preview,record['local_all34_valid'],
              record['local_comparisons']['historical_main'],flush=True)
    eligible=[r for r in screens if r.get('dense',{}).get('success') and r['dense']['complete_time_s']<=15.06985]
    eligible.sort(key=lambda r:(-r['local_comparisons']['historical_main']['not_worse'],r['local_total_window_time_s']))
    for record in eligible[:3]:
        if time.perf_counter()>deadline-250.:
            break
        policy=policy_for(record['feed_gain'],record['preview_gain'])
        result=full(policy,record['label'],'original')
        pair=dict(response={k:record[k] for k in ['index','label','feed_gain','preview_gain']},
                  original=result,dense=record['dense'])
        evaluations.append(pair)
        write(stage/'full_evaluations.json',evaluations)
    completed=[r for r in evaluations if r['original']['success']]
    chosen=min(completed,key=lambda r:(-r['original']['corner_comparisons']['historical_main']['not_worse'],
               r['original']['complete_time_s'])) if completed else None
    if chosen:
        response=chosen['response']
        policy=policy_for(response['feed_gain'],response['preview_gain'])
        path=stage/'selected.pt'
        torch.save(dict(policy=policy.state_dict(),interface=[14,3,922302],policy_class='JointResponsePolicy',
            feed_gain=response['feed_gain'],preview_gain=response['preview_gain'],
            source_checkpoint=str(model),source_checkpoint_sha256=sha(model),new_ppo_updates=0,
            observation_features=PreviewCueRuntime(refs['original']).feature_names(),policy_memory_scalars=0),path)
        chosen['checkpoint']=str(path)
        chosen['checkpoint_sha256']=sha(path)
    write(stage/'result.json',dict(status='finite_search_completed',new_ppo_updates=0,
        training_kind='Finite parameter optimization, not PPO or offline fitting',
        evaluated_responses=len(screens),full_evaluations=evaluations,selected_candidate=chosen,
        adopted=False,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False,
        remaining='Paired numerical resolution and final joint requirements remain mandatory'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    assert Path(args.name).name==args.name
    if args.worker:
        worker(args)
        return
    manifest=verify_bound_inputs()
    stage=ROOT/args.name
    stage.mkdir(exist_ok=False)
    (stage/'source').mkdir()
    sources=[Path(__file__),ROOT/'compact_joint_response.py',ROOT/'compact_fine_policy.py',
             ROOT/'compact_previewcue_runtime.py',ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',
             BASE/'natural_case_metrics.py',BASE/'natural_case_audit.py',BASE/'paper_complete_evaluation.py']
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p,stage/'source'/p.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        wall_budget_s=1800,child_timeout_s=1740,worker_deadline_s=1650,
        responses=RESPONSES,source_checkpoint=args.checkpoint,source_checkpoint_sha256=sha(args.checkpoint),
        contexts=str(CONTEXTS),contexts_sha256=sha(CONTEXTS),source_sha256=hashes,
        bound_historical_manifest=manifest,goal_complete=False,method_frozen=False,
        evidence_gap='New14input PPO preserves dense5percent advantage but falls short on long time and all34. Saved actions show broad-angle feed requests briefly below100 and extra path length. Probe feed response and modest preview scaling jointly.',
        procedure='Nine finite response pairs; all34 paired witnessed starts with48decisions each. Valid variants receive full dense checks. Up to3dense-qualified variants ranked by local historical corner count then total crossing time receive full original checks. Local probes cannot establish full success.',
        adoption='No automatic adoption. Same candidate must pass final all34 and credible full time criteria plus refined resolution. Save best intermediate candidate for subsequent PPO.',
        stop='Registered list or wall limit; retain failures; no changes to baseline or physical executor',
        new_ppo_updates=0,observation_dimension=14,policy_memory_scalars=0))
    started=time.perf_counter()
    code=None
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            child=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--name',args.name,
                '--checkpoint',args.checkpoint,'--worker'],stdout=out,stderr=err,timeout=1740)
            code=child.returncode
            status='finished'
        except subprocess.TimeoutExpired:
            status='timeout'
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    verify_bound_inputs()
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:
        print((stage/'stderr.txt').read_text(encoding='utf-8')[-4000:])


if __name__=='__main__':
    main()
