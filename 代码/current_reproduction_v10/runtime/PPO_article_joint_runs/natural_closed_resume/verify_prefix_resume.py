"""A bounded non-learning equality check for uninterrupted versus resumed motion."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_complete_evaluation import evaluate
from arc_cruise_policy import build_policy
from study_io import sha,write_json


def main():
    torch.set_num_threads(1);start=time.perf_counter();stage=OUT/'prefix_resume_verification';stage.mkdir(exist_ok=False)
    selected=json.loads((OUT/'joint_actual_context_feedback_training/selected_policy.json').read_text(encoding='utf-8'))
    case=BASE/'paper_evidence_review/natural_closed_case';ref=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'));c=geometry['sharp_corners'][2];k=c['reference_vertex_index'];v=ref[k]
    u=v-ref[k-1];u/=np.linalg.norm(u);w=ref[k+1]-v;w/=np.linalg.norm(w)
    reference=np.array([v-50*u,v-25*u,v,v+25*w,v+50*w]);md=torch.load(selected['model'],map_location='cpu',weights_only=False)
    write_json(stage/'registration.json',dict(kind='software_motion_continuation_verification',wall_budget_s=120,ppo_updates=0,
        hypothesis='Uninterrupted and exactly replayed continuation yield matching complete motion with nonzero causal corner memory.',source_sha256=sha(Path(__file__))))
    p=build_policy(load_credit_policy(md),selected['spec']);full=evaluate(p,environment_factory(reference),stage/'uninterrupted.npz',start+100)
    assert full['success'],full
    z=dict(np.load(stage/'uninterrupted.npz'));n=30;prefix=dict(reference=reference,rows=z['rows'][:20*n],
        observations=z['observations'][:n],actions=z['actions'][:n],latents=z['latents'][:n],executed_actions_1ms=z['executed_actions_1ms'][:20*n])
    p=build_policy(load_credit_policy(md),selected['spec']);e=environment_factory(reference);p.bind(e);error=0.
    for i in range(n+1):
        if i:
            for row in prefix['rows'][(i-1)*20:i*20]:e.tracker.update(row[:2])
            row=prefix['rows'][i*20-1];e.position=row[:2].copy();e.velocity=row[2:4].copy();e.acceleration=row[4:6].copy()
            e.heading=float(np.arctan2(row[9],row[8]));e.held=prefix['actions'][i-1].copy()
        e.ticks=20*i;e.decisions=i
        if i<n:error=max(error,float(np.max(abs(e.decode_action(p.act(prefix['observations'][i],True)[0])-prefix['actions'][i]))))
    resumed=evaluate(p,e,stage/'resumed.npz',start+100,prefix=prefix,bind_policy=False)
    zz=dict(np.load(stage/'resumed.npz'));assert resumed['success'] and len(zz['rows'])==len(z['rows'])
    maxima=np.max(abs(z['rows']-zz['rows']),axis=0)
    assert error<1e-7 and maxima[:2].max()<1e-8 and maxima[2:8].max()<1e-3,(error,maxima)
    assert full['complete_time_s']==resumed['complete_time_s']
    write_json(stage/'receipt.json',dict(status='passed',wall_s=time.perf_counter()-start,
        complete_time_s=full['complete_time_s'],replay_action_max_error=error,raw_row_column_max_errors=maxima.tolist(),
        complete_time_equal=True,ppo_updates=0,physics_ticks_executed=full['physics_ticks']+resumed['physics_ticks']-20*n,goal_complete=False))
    print('PREFIX_RESUME_VERIFIED',full['complete_time_s'],error,maxima.tolist(),time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
