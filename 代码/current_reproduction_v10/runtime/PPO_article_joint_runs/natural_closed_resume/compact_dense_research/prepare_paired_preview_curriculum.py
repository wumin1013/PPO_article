"""Transparent shared two-action PPO pretraining, then matched2/3-output branches."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json,sys,time,subprocess,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from compact_control import CompactPolicy
from compact_training_environment import TrainingRuntime
from learning_update_v4 import optimizers
from paper_complete_evaluation import evaluate

ROOT=Path(__file__).resolve().parent
STAGE=ROOT/'paired_preview_curriculum_from_two_action80'
SOURCE=ROOT/'ppo14_action2_allow_seed922301_mc_continue/checkpoint_0080.pt'
SELECTION=ROOT/'two_action_distance_selection_after80/result.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def worker():
    torch.set_num_threads(1);start=time.perf_counter()
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
    distance=json.loads(SELECTION.read_text(encoding='utf-8'))['selected']['distance_mm']
    latent=torch.tensor((distance-6.4)/5.6,dtype=torch.float32)
    actual_distance=.8+5.6*(1.+float(latent))
    reference=np.loadtxt(ROOT/'dense_closed_case/reference.csv',delimiter=',',skiprows=1)
    parent=CompactPolicy(14,2,922301);parent.load_state_dict(saved['policy'])
    results=[]
    for action_dim in (2,3):
        policy=CompactPolicy(14,action_dim,922301)
        for layer in (0,2):policy.actor[layer].load_state_dict(parent.actor[layer].state_dict())
        policy.critic.load_state_dict(parent.critic.state_dict())
        with torch.no_grad():
            policy.actor[-1].weight[:2].copy_(parent.actor[-1].weight)
            policy.actor[-1].bias[:2].copy_(parent.actor[-1].bias)
            policy.log_std[:2].copy_(parent.log_std)
            if action_dim==3:
                policy.actor[-1].weight[2].zero_();policy.actor[-1].bias[2].copy_(latent)
        def environment():return TrainingRuntime(reference,dimension=14,action_dim=action_dim,
            preview_rule_mm=actual_distance,potential_speed=30.,remaining_encoding='log')
        optimizer=optimizers(policy);env=environment()
        checkpoint=STAGE/f'action{action_dim}_start.pt'
        torch.save(dict(policy=policy.state_dict(),optimizers={k:v.state_dict() for k,v in optimizer.items()},
            environment=env.state(),torch_rng=saved['torch_rng'],numpy_rng=saved['numpy_rng'],python_rng=saved['python_rng'],
            steps=saved['steps'],updates=saved['updates'],episodes=saved['episodes'],episode_reward=0.,best=None,
            interface=[14,action_dim,922301,False,actual_distance,30.,'log'],observation_features=env.feature_names(),
            policy_memory_scalars=0,shared_pretraining=dict(source=str(SOURCE),source_sha256=sha(SOURCE),
                steps=saved['steps'],updates=saved['updates'],kind='Independent two-output PPO, no three-action weights used',
                optimizer_reset_for_both_branches=True,training_environment_reset_for_both_branches=True,
                previously_unfinished_training_prefix_ticks=saved['environment']['ticks'],
                distance_selection=str(SELECTION),distance_selection_sha256=sha(SELECTION))),checkpoint)
        result=evaluate(policy,env,STAGE/f'action{action_dim}_initial.npz',start+260.)
        results.append(dict(action_dim=action_dim,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),result=result))
    a=np.load(STAGE/'action2_initial.npz');b=np.load(STAGE/'action3_initial.npz')
    equal_shape=a['rows'].shape==b['rows'].shape
    differences={key:float(np.max(abs(a[key]-b[key]))) if a[key].shape==b[key].shape else None for key in ['rows','actions','observations']}
    matched=bool(all(r['result']['success'] for r in results) and equal_shape and differences['rows']<1e-4 and differences['actions']<1e-4)
    write(STAGE/'result.json',dict(kind='Shared PPO pretraining and output-extension initialization; zero new optimizer updates',
        new_ppo_updates=0,shared_pretraining_decisions=saved['steps'],shared_pretraining_updates=saved['updates'],
        nominal_distance_mm=distance,decoded_common_distance_mm=actual_distance,initial_executions_matched=matched,
        maximum_differences=differences,results=results,wall_s=time.perf_counter()-start,goal_complete=False,
        limitation='Branches share the same independently trained two-output parent. This is a paired curriculum comparison, not two independent from-scratch training histories. Future independent seeds must repeat the complete pipeline.'))
    print(json.dumps(dict(matched=matched,distance_mm=actual_distance,differences=differences,
        complete_times=[r['result']['complete_time_s'] for r in results]),indent=2))
    if not matched:raise RuntimeError('Do not start paired training: initial execution match failed')

def main():
    if '--worker' in sys.argv:worker();return
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    sources=[Path(__file__),ROOT/'compact_control.py',ROOT/'compact_training_environment.py']
    for p in sources:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(wall_budget_s=300,child_timeout_s=285,new_ppo_updates=0,
        source_checkpoint=str(SOURCE),source_checkpoint_sha256=sha(SOURCE),distance_selection=str(SELECTION),
        distance_selection_sha256=sha(SELECTION),sources={str(p):sha(p) for p in sources},
        design='Copy two-output PPO parent trunk, steering/feed head and critic to both branches. Add a constant preview head only to3-output branch. Same decoded distance and optimizer reset. Full evaluations from rest.',
        adoption='Both full successful executions and maximum row/action differences below1e-4 at identical tick count. Report actual differences, never assert exact identity without checking.',
        stop='Registered global wall deadline or failed execution match.',goal_complete=False))
    start=time.perf_counter();code=None
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            completed=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=285)
            status='finished';code=completed.returncode
        except subprocess.TimeoutExpired:status='timeout'
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8'))
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8'))

if __name__=='__main__':main()
