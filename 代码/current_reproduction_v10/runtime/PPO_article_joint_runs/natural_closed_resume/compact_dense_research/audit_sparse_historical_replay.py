"""Replay every recorded original-case decision with reduced memory; no simulation."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from evaluate_historical_dense import ROOT, BASE, OUT, HIST, sha, write, verify_bound_inputs
sys.path[:0] = [str(BASE),str(OUT),str(ROOT)]
from candidate_policy_factory import candidate_policy
from turn_aware_projector import environment_factory
from sparse_historical_memory import sparsify_memory, SparseObservableMemoryPolicy


def main():
    stage = ROOT / 'historical_sparse_memory_full_recorded_replay'
    stage.mkdir(exist_ok=False)
    verify_bound_inputs()
    write(stage/'registration.json',dict(kind='All recorded original-case decisions with exact zero-coefficient memory removal',
        wall_budget_s=600,physics_ticks=0,ppo_updates=0,
        source_sha256={str(Path(__file__)):sha(Path(__file__)),str(ROOT/'sparse_historical_memory.py'):sha(ROOT/'sparse_historical_memory.py')},
        original_trace_sha256=sha(HIST/'full_trace.npz'),
        adoption='Every recorded decision must match within1e-7 decoded physical-action units. This checks the recorded original trajectory, not new closed-loop performance.',
        stop='On mismatch or600s deadline; preserve actual checked count.'))
    started=time.perf_counter();torch.set_num_threads(1)
    selection=json.loads((HIST/'selected_policy.json').read_text(encoding='utf-8'))
    policy=sparsify_memory(candidate_policy(torch.load(selection['model'],map_location='cpu',weights_only=False),selection))
    data=np.load(HIST/'full_trace.npz');rows=data['rows'];actions=data['actions'];observations=data['observations']
    env=environment_factory(data['reference']);policy.bind(env)
    counts=[];node=policy
    while node is not None:
        if isinstance(node,SparseObservableMemoryPolicy):
            counts.append(dict(original_memory_scalars=int(np.prod(node.original_memory_shape)),memory_scalars=int(node.memory.size),
                original_previous_feature_scalars=int(np.prod(node.original_previous_shape)),previous_feature_scalars=int(node.previous_features.size)))
        node=getattr(node,'base',None)
    maximum=0.;checked=0;last=started;status='complete'
    for i,observation in enumerate(observations):
        if time.perf_counter()-started>=590:
            status='deadline';break
        if i:
            for row in rows[(i-1)*20:i*20]:
                env.tracker.update(row[:2])
            row=rows[i*20-1]
            env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
            env.heading=float(np.arctan2(row[9],row[8]));env.held=actions[i-1].copy()
        env.ticks=i*20;env.decisions=i
        latent=policy.act(observation,True)[0]
        decoded=env.decode_action(latent,env.fixed_d)
        error=float(np.max(abs(decoded-actions[i])))
        maximum=max(maximum,error);checked+=1
        if error>1e-7:
            status='mismatch';break
        if time.perf_counter()-last>=40:
            last=time.perf_counter();print('REPLAY',checked,len(actions),maximum,flush=True)
    report=dict(status=status,all_recorded_decisions_checked=checked==len(actions),checked_decisions=checked,
        recorded_decisions=len(actions),maximum_decoded_action_error=maximum,passed=checked==len(actions) and maximum<=1e-7,
        layers=counts,original_memory_scalars=sum(x['original_memory_scalars'] for x in counts),
        memory_scalars=sum(x['memory_scalars'] for x in counts),
        original_previous_feature_scalars=sum(x['original_previous_feature_scalars'] for x in counts),
        previous_feature_scalars=sum(x['previous_feature_scalars'] for x in counts),
        physics_ticks=0,ppo_updates=0,wall_s=time.perf_counter()-started,goal_complete=False,
        observation_dimension=72,scope='Exact structural memory reduction audited on every recorded original-case decision. A new complete closed-loop rollout and dense-case checks remain required before adopting the implementation.')
    write(stage/'result.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
