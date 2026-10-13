"""Bounded complete-suffix probes at the newly exposed training-pool failure."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from canonical_gate_state import checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from analyze_nonlinear_joint_response import branch
from learn_nonlinear_joint_feedback import rollout
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}
AMPLITUDE=np.array([.01,.03,.03])


def initialize(stage,metadata,reference,rows,actions,deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,policy=load_credit_policy(metadata),reference=reference,rows=rows,
                actions=actions,factory=checkpoint_environment_factory('gate_band_canonical_kernel'),deadline=deadline)


def collect(frame):
    costs=dict(actor_calls=0,physics_ticks=0,saved_prefix_geometry_ticks=0)
    ref,rows,actions=[WORK[k] for k in ('reference','rows','actions')]
    env=reconstruct(ref,rows,actions,frame,costs,env_factory=WORK['factory'])
    x=env.observation();initial=env.state();latent=WORK['policy'].act(x,True)[0]
    assert np.array_equal(env.decode_action(latent),actions[frame//20])
    _,_,_,info=env.step(latent)
    assert np.array_equal(info['rows'],rows[frame:frame+len(info['rows'])])
    folder=WORK['stage']/f'frame{frame}';folder.mkdir();records=[]
    for head in range(3):
        for sign in (-1,1):
            offset=sign*AMPLITUDE[head]*np.eye(3)[head]
            r,full,held,observations=branch(WORK['policy'],WORK['factory'],ref,rows,actions,
                                          initial,offset,costs,WORK['deadline'])
            file=folder/f'head{head}_{sign:+d}.npz'
            np.savez_compressed(file,reference=ref,rows=full,actions=held,suffix_observations=observations,
                                frame=frame,offset=offset)
            r.update(path=40,frame=frame,head=head,sign=sign,offset=offset.tolist(),
                     trace=file.relative_to(WORK['stage']).as_posix(),recorded_ticks=len(full))
            # Constraint credit can inspect a common observed physical prefix;
            # it is NOT a complete-time or complete-shape reward for a failure.
            records.append(r)
    result=dict(frame=frame,observation=x.tolist(),records=records,costs=costs,exact_nominal_decision_identity=True)
    write_json(folder/'result.json',result)
    return result


def evaluate_ancestor(job):
    tag,file=job
    md=torch.load(file,map_location='cpu',weights_only=False);policy=load_credit_policy(md)
    costs=dict(actor_calls=0,physics_ticks=0)
    r,rows,actions,obs=rollout(policy,WORK['factory'],WORK['reference'],costs,WORK['deadline'])
    filename=WORK['stage']/(tag+'.npz')
    np.savez_compressed(filename,reference=WORK['reference'],rows=rows,actions=actions,observations=obs)
    r.update(tag=tag,path=40,checkpoint_sha256=sha(file),trace=filename.name,costs=costs,recorded_ticks=len(rows))
    return r


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_recentered_fine_trace_learning_20260919'
    full=BASE/'canonical_gate_band_recentered_training_pool_check_20260919'
    metadata=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    with np.load(full/'path40.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
    assert read(full/'receipt.json')['summary']['failed_paths']==[40]
    ancestors=[('ancestor_secant',BASE/'canonical_gate_band_secant_calibrated_joint_learning_20260919/research_candidate.pt'),
               ('ancestor_fast',BASE/'canonical_gate_band_fast_absolute_shape_joint_credit_20260919/research_candidate.pt')]
    stage=register('canonical_gate_band_training_pool_failure_credit_20260919',
        [__file__,BASE/'analyze_nonlinear_joint_response.py',BASE/'probe_coordinated_suffix.py',
         source/'research_candidate.pt',full/'path40.npz',full/'receipt.json',full/'path40_representation_audit.json']+
        [p for _,p in ancestors],
        dict(wall_s=180,workers=4,path=40,frames_ms=[220,280,340,380,420],
             amplitude=AMPLITUDE.tolist(),complete_suffix_branches=30,ancestor_paths=2),
        'The latest learned actor completed63/64 training paths; closed polygon40 '
        'violated1.25mm at451ms with3/13 gates, so no full64 time/cruise benefit is '
        'claimed. Its state features are nonzero; do not assume missing capacity '
        'before measuring credit. At five existing decision states execute single '
        '20ms +/- steering/feed/preview interventions and then the unchanged actor '
        'through actual completion or failure. Audit exact nominal next-step identity. '
        'Two earlier research actors are tested on this previously untested path '
        'only to locate whether the defect predates the latest local update; these '
        'are not matched causal mechanism controls. Retained86cd successful path '
        'is cached, not rerun. Preserve raw failed prefixes for constraint diagnosis '
        'only, with no complete-time/shape credit. Use measured complete rescue or '
        'constraint responses to learn a state-based joint correction next; never '
        'deploy the pulse, path/time condition or a hand speed limit.')
    started=time.perf_counter();pool=None;queries=[];controls=[]
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','physical_response_collection_for_parameter_learning','--next',
        '在新暴露的训练路径40上验证完整三动作救援后果，随后学习状态函数修复。'],check=True,stdout=subprocess.DEVNULL)
    try:
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,ref,rows,actions,started+172))
        pending=[pool.apply_async(collect,(frame,)) for frame in (220,280,340,380,420)]
        ancestor_jobs=[pool.apply_async(evaluate_ancestor,(job,)) for job in ancestors]
        for future in pending:
            q=future.get();queries.append(q)
            write_json(stage/'results.json',dict(queries=queries,ancestors=controls,probe_only=True,adopted=False,goal_complete=False))
            print('PATH40_SUFFIX',q['frame'],[(r['head'],r['sign'],r['success'],r['recorded_ticks'],r['geometry']['gates_passed']) for r in q['records']],flush=True)
        for future in ancestor_jobs:
            r=future.get();controls.append(r);print('PATH40_ANCESTOR',r['tag'],r['success'],r['recorded_ticks'],r['geometry']['gates_passed'],flush=True)
        pool.close();pool.join();pool=None
        write_json(stage/'results.json',dict(queries=queries,ancestors=controls,probe_only=True,adopted=False,goal_complete=False))
        receipt(stage,dict(status='completed_training_pool_failure_credit',branches=30,
            complete_rescues=sum(r['success'] for q in queries for r in q['records']),
            ancestor_paths=len(controls),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_training_pool_failure_credit',error=str(exc),queries=len(queries),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'physical_response_collection_finished','--next',
            '用路径40真实完整后果或共同物理前缀约束信用学习修复，保持已有形状和巡航收益。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
