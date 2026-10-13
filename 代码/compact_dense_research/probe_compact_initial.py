"""Bounded interface and initial control check, no parameter updates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
from compact_control import CompactRuntime,CompactPolicy,feature_names,BASE
from paper_complete_evaluation import evaluate

def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def worker(stage):
    torch.set_num_threads(1)
    reference=np.loadtxt(ROOT/'dense_closed_case/reference.csv',delimiter=',',skiprows=1)
    started=time.perf_counter();records=[]
    for dimension in [14,12,10]:
        for action_dim in [2,3]:
            env=CompactRuntime(reference,dimension=dimension,action_dim=action_dim)
            policy=CompactPolicy(dimension,action_dim)
            x=env.observation();a=policy.act(x,True)[0]
            assert len(x)==len(feature_names(dimension))==dimension and a.shape==(action_dim,)
            assert env.decode_action(a).shape==(3,) and policy.actor[-1].out_features==action_dim
            assert 'update' not in env.state()['tracker']
    for action_dim,allow_swing in [(3,True),(2,True),(3,False)]:
        name=f'action{action_dim}_swing{int(allow_swing)}'
        env=CompactRuntime(reference,action_dim=action_dim,allow_swing=allow_swing)
        policy=CompactPolicy(action_dim=action_dim)
        result=evaluate(policy,env,stage/(name+'.npz'),started+400)
        records.append(dict(name=name,result=result,parameter_count=sum(p.numel() for p in policy.parameters()),
            observation_dimension=14,actor_output_dimension=action_dim,policy_memory_scalars=0))
        write(stage/'results.json',records)
        print(name,result['success'],result['reason'],result['complete_time_s'],result['elapsed_prefix_or_complete_s'],flush=True)
    if len(records)==3:
        a=np.load(stage/'action3_swing1.npz')['rows'];b=np.load(stage/'action2_swing1.npz')['rows']
        assert a.shape==b.shape and np.max(abs(a-b))<1e-6, 'Matched constant initial decoded commands must agree.'
        write(stage/'matched_initial_check.json',dict(passed=True,maximum_difference=float(np.max(abs(a-b)))))

def main():
    stage=ROOT/'compact_initial_probe'
    if '--worker' in sys.argv:worker(stage);return
    stage.mkdir(exist_ok=False)
    write(stage/'registration.json',dict(kind='Interface and initial-policy diagnostic',wall_budget_s=480,child_timeout_s=450,
        ppo_updates=0,dimensions=[14,12,10],initial_full_checks=['three action allow','two action allow','three action no-swing'],
        no_swing='After each witnessed sharp-corner gate until min50mm or midpoint to next sharp corner, outgoing inward signed displacement must be >=-1e-7mm; extra prior only, no command override.',
        sources={str(p):sha(p) for p in [Path(__file__),ROOT/'compact_control.py',ROOT/'dense_closed_case/registration.json']},
        adoption='Check interface correctness and initial completion only; not a trained-policy comparison.',stop='Stop on each physical/prior failure or global deadline.'))
    started=time.perf_counter()
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=450)
            status='finished';code=result.returncode
        except subprocess.TimeoutExpired:status='timeout';code=None
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,ppo_updates=0,goal_complete=False))
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-8000:])
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf-8')[-8000:])

if __name__=='__main__':main()
