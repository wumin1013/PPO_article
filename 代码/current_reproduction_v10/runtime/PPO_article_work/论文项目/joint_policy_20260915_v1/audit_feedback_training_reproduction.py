"""Distinguish training reproduction failure from development geometry gaps."""
import time
import numpy as np
import torch
from dynamic_residual_policy import PreviewResidualPolicy
from search_complete_return import read
from joint_runtime_v5 import environment_factory
from train_pilot import evaluate
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from study_io import BASE,register,receipt,write_json,sha


def main():
    source=BASE/'consistent_teacher_feedback_learning_20260916';indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    stage=register('feedback_training_reproduction_audit_20260916',[__file__,source/'receipt.json',source/'epoch_50.pt',source/'epoch_200.pt',
        source/'training_sources.json',BASE/'learning_preparation_v4/paths.json',BASE/'dynamic_residual_policy.py'],
        dict(training_updates=0,new_training_path_evaluations=32,evaluation_decisions=8000,physics_ticks=200000,wall_s=180),
        'Run the frozen feedback student independently on the exact sixteen training paths at epochs50/200. Reuse saved teacher results. Failure here is not solely an unseen-geometry generalization problem; no intervention or teacher actions at evaluation.')
    torch.set_num_threads(1);started=time.perf_counter();costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];refs=[np.asarray(paths[i]) for i in indices];outputs=[]
    for epoch in (50,200):
        saved=torch.load(source/f'epoch_{epoch}.pt',map_location='cpu',weights_only=False)
        model=PreviewResidualPolicy(saved['observation_dim']);model.load_state_dict(saved['policy'])
        results=evaluate(model,refs,stage,f'epoch_{epoch}',None,costs,started+180,environment_factory)
        assert costs['evaluation_actor_calls']<=8000 and costs['evaluation_physics_ticks']<=200000
        records=[]
        for i,r in enumerate(results):
            with np.load(stage/f'epoch_{epoch}'/f'{i}.npz') as d:
                shape=corner_shape(d['reference'],d['rows'],d['actions']) if len(d['reference'])==3 else None
                spatial=geometric_turn_metrics(d['reference'],d['rows']) if shape else None
            records.append(dict(training_path=indices[i],result=r,shape=shape,spatial=spatial))
        outputs.append(dict(epoch=epoch,successes=sum(r['success'] for r in results),records=records))
        print('EPOCH',epoch,'training success',outputs[-1]['successes'],'failed',[r['training_path'] for r in records if not r['result']['success']],flush=True)
    write_json(stage/'results.json',dict(checkpoints=outputs,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
