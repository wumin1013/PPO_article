"""Complete64-path training-pool screening, reusing exact same-model traces."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from canonical_gate_state import checkpoint_environment_factory
from learn_nonlinear_joint_feedback import rollout
from learn_retained_joint_growth import smooth_cruise
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}


def curved(reference):
    segment=np.diff(reference,axis=0)
    length=np.linalg.norm(segment,axis=1)
    direction=segment[length>1e-10]/length[length>1e-10,None]
    return bool(len(direction)>1 and np.max(np.linalg.norm(np.diff(direction,axis=0),axis=1))>1e-8)


def initialize(stage,metadata,deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,policy=load_credit_policy(metadata),
                factory=checkpoint_environment_factory('gate_band_canonical_kernel'),deadline=deadline)


def evaluate(job):
    path,reference=job
    costs=dict(actor_calls=0,physics_ticks=0)
    r,rows,actions,observations=rollout(WORK['policy'],WORK['factory'],reference,costs,WORK['deadline'])
    file=WORK['stage']/f'path{path}.npz'
    np.savez_compressed(file,reference=reference,rows=rows,actions=actions,observations=observations)
    r.update(path=path,trace=file.relative_to(BASE).as_posix(),trace_sha256=sha(file),
             curved_path=curved(reference),reused=False,costs=costs)
    if r['success']:r['smooth_cruise99_s']=smooth_cruise(rows)
    write_json(WORK['stage']/f'path{path}.json',r)
    return r


def summarize(records):
    complete=len(records)==64 and all(r['success'] for r in records)
    shapes=[r for r in records if 'shape' in r and r['success']]
    return dict(paths=len(records),successes=sum(r['success'] for r in records),
        all_complete=complete,failed_paths=[r['path'] for r in records if not r['success']],
        full_time_s=sum(r['complete_time_s'] for r in records) if complete else None,
        cruise99_s=sum(r['motion']['cruise_ge99_s'] for r in records) if complete else None,
        curved_cruise99_s=sum(r['motion']['cruise_ge99_s'] for r in records if r['curved_path']) if complete else None,
        corner_reverse_sum_deg=sum(r['shape']['full_physical_reverse_deg'] for r in shapes),
        corner_exit_sum_mm=sum(r['shape']['out_edge_outward_mm'] for r in shapes),
        corner_energy_sum=sum(r['shape']['full_normalized_energy'] for r in shapes),
        inward_first_turns=sum(r['target_motion']['first_turn_inside'] is True for r in shapes),
        physical_peaks=np.max(np.array([r['audit']['peaks'] for r in records]),axis=0).tolist(),
        all_rest_pass=all(r['audit']['rest_pass'] for r in records),
        original_constraints_unchanged=True,training_pool_only=True,
        final_shape_accepted=False,independent_seeds_done=False,fair_comparison_done=False,generalization_done=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-stage',required=True)
    parser.add_argument('--stage',required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    source=(BASE/args.source_stage).resolve()
    assert source.parent==BASE
    data=read(source/'results.json');selected=next(c for c in data['candidates'] if c['tag']==data['best'])
    model_file=source/'research_candidate.pt';metadata=torch.load(model_file,map_location='cpu',weights_only=False)
    retained=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918'
    assert sha(retained/'candidate_0.5.pt')=='86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
    pool_data=read(retained/'results.json')['full_candidate']
    refs={}
    for r,trace in zip(pool_data['results'],pool_data['traces']):
        with np.load(BASE/trace) as d:refs[r['original_path']]=d['reference'].copy()
    assert len(refs)==64 and set(refs)==set(range(64))
    stage=register(args.stage,[__file__,BASE/'learn_nonlinear_joint_feedback.py',
        BASE/'analyze_nonlinear_joint_response.py',BASE/'complete_corner_metrics.py',
        BASE/'complete_motion_metrics.py',BASE/'validate_actions.py',
        model_file,source/'results.json',source/'receipt.json',retained/'results.json',retained/'receipt.json'],
        dict(wall_s=300,workers=4,training_paths=64,reused_complete_paths=len(selected['records']),
             new_paths=64-len(selected['records'])),
        'Extend this exact measured research checkpoint to the original64 TRAINING '
        'paths before further efficiency claims. Reuse its20 previously completed '
        'same-model traces after checking exact reference identity and record hashes; '
        'execute only the missing44 paths with the unchanged20ms/1ms runtime, '
        '1.25mm single-sided Euclidean corridor, vectorV/A/J, ordered coverage, '
        'complete braking and20ms rest. Report complete total time/cruise only if '
        'ALL64 complete. A failed prefix has no full-route time benefit. This is '
        'training-pool screening, not independent seeds, fair fixed-preview training '
        'comparison, long-path generalization, sealed testing or final adoption.')
    started=time.perf_counter();pool=None;records=[]
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_training_pool_evaluation','--next',
        '复用同一模型的20条完成轨迹，只补齐44条训练路径，核实整池成功率与完整时间。'],check=True,stdout=subprocess.DEVNULL)
    try:
        for old in selected['records']:
            r=copy.deepcopy(old)
            assert r['success']
            file=source/r['trace']
            with np.load(file) as d:reference=d['reference'].copy()
            assert np.array_equal(reference,refs[r['path']])
            r.update(trace=file.relative_to(BASE).as_posix(),trace_sha256=sha(file),
                     curved_path=curved(reference),reused=True)
            records.append(r)
        write_json(stage/'reuse_receipt.json',dict(model_sha256=sha(model_file),
            records=[dict(path=r['path'],trace=r['trace'],sha256=r['trace_sha256']) for r in records]))
        done={r['path'] for r in records};pending=[(p,ref) for p,ref in refs.items() if p not in done]
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,started+290))
        for r in pool.imap_unordered(evaluate,pending):
            records.append(r)
            write_json(stage/'results.json',dict(records=sorted(records,key=lambda r:r['path']),summary=summarize(records),
                source_model_sha256=sha(model_file),adopted=False,goal_complete=False))
            print('TRAINING_POOL',len(records),'of64',r['path'],r['success'],r['reason'],flush=True)
        pool.close();pool.join();pool=None
        summary=summarize(records)
        summary['retained_full_time_s']=89.735
        summary['full_time_change_s']=summary['full_time_s']-89.735 if summary['all_complete'] else None
        write_json(stage/'results.json',dict(records=sorted(records,key=lambda r:r['path']),summary=summary,
            source_model_sha256=sha(model_file),adopted=False,goal_complete=False))
        receipt(stage,dict(status='completed_learned_training_pool_check',summary=summary,
            reused_paths=len(selected['records']),new_paths=len(pending),wall_s=time.perf_counter()-started,
            source_model_sha256=sha(model_file),adopted=False,goal_complete=False))
        print('TRAINING_POOL_COMPLETE',summary,flush=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        write_json(stage/'results.json',dict(records=sorted(records,key=lambda r:r['path']),adopted=False,goal_complete=False))
        receipt(stage,dict(status='failed_learned_training_pool_check',error=str(exc),paths=len(records),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_training_pool_evaluation_finished','--next',
            '根据完整64训练池的真实成功率、形状和效率修正学习分布与三动作机制；不打开封存测试。'],
            check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
