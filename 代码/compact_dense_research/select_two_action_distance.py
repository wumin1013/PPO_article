"""Registered deterministic parameter selection for the independently trained2-output actor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import sys,time,json,subprocess,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from compact_control import CompactPolicy
from compact_training_environment import TrainingRuntime
from paper_complete_evaluation import evaluate

ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'two_action_distance_selection_after80'
MODEL=ROOT/'ppo14_action2_allow_seed922301_mc_continue/checkpoint_0080.pt'
DISTANCES=[.8,1.25,1.5,2.,3.,4.,6.,9.,12.]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def worker():
    torch.set_num_threads(1);started=time.perf_counter()
    reference=np.loadtxt(ROOT/'dense_closed_case/reference.csv',delimiter=',',skiprows=1)
    saved=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert saved['interface'][:3]==[14,2,922301]
    policy=CompactPolicy(14,2,922301);policy.load_state_dict(saved['policy'])
    records=[]
    for distance in DISTANCES:
        if time.perf_counter()>started+255.:break
        env=TrainingRuntime(reference,dimension=14,action_dim=2,preview_rule_mm=distance,potential_speed=30.,remaining_encoding='log')
        file=STAGE/f'distance_{distance:g}.npz'
        result=evaluate(policy,env,file,started+265.)
        record=dict(distance_mm=distance,result=result,trace=str(file));records.append(record)
        write(STAGE/'evaluations.json',records)
        print('DISTANCE',distance,result['success'],result['complete_time_s'],result['reason'],flush=True)
    feasible=[r for r in records if r['result']['success']]
    selected=min(feasible,key=lambda r:r['result']['complete_time_s']) if feasible else None
    reproduction=next((r for r in records if r['distance_mm']==1.5),None)
    if reproduction:assert reproduction['result']['success'] and abs(reproduction['result']['complete_time_s']-28.273)<1e-8
    write(STAGE/'result.json',dict(kind='Parameter selection only; no PPO or offline model fitting',ppo_updates=0,
        evaluated_distances=len(records),selected=selected,baseline_reproduction_passed=reproduction is not None,
        model=str(MODEL),model_sha256=sha(MODEL),wall_s=time.perf_counter()-started,goal_complete=False,
        next='Continue independent two-output PPO training at the selected global distance; this probe alone is not the final trained ablation.'))

def main():
    if '--worker' in sys.argv:worker();return
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__),ROOT/'compact_control.py',ROOT/'compact_training_environment.py']
    manifest={}
    for p in sources:
        q=STAGE/'source'/p.name;shutil.copy2(p,q);manifest[str(p)]={'sha256':sha(p),'snapshot':str(q)}
    write(STAGE/'registration.json',dict(kind='Global parameter search on an independently trained2-output actor',
        distances_mm=DISTANCES,wall_budget_s=300,child_timeout_s=285,ppo_updates=0,model=str(MODEL),model_sha256=sha(MODEL),
        reference_sha256=sha(ROOT/'dense_closed_case/reference.csv'),sources=manifest,
        adoption='Select shortest complete successful trajectory, retain every failure; subsequently retrain two-output policy at selected distance.',
        stop='Stop at global wall deadline or completion of the nine predeclared distances.',goal_complete=False))
    started=time.perf_counter();code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=285)
            status='finished';code=result.returncode
        except subprocess.TimeoutExpired:status='timeout'
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,goal_complete=False,
        sources_unchanged=all(sha(p)==v['sha256'] for p,v in manifest.items())))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8'))
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8'))

if __name__=='__main__':main()
