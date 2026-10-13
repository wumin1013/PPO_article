"""Closed-loop replay of original training geometries after frame comparison."""
import time
import numpy as np
import torch
from frame_feedback_policy import FrameFeedbackPolicy
from train_feedback_from_optimized_rollouts import decoded
from train_pilot import evaluate
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    models=[BASE/'reference_frame_feedback_learning_20260916/epoch_1000.pt',BASE/'heading_frame_feedback_learning_20260916/epoch_2000.pt']
    source=BASE/'interior_teacher_feedback_learning_20260916'
    stage=register('frame_feedback_training_reproduction_20260916',[__file__]+models+[source/'training_data.npz',BASE/'frame_feedback_policy.py'],
        dict(training_decisions=0,evaluation_decisions=6000,physics_ticks=150000,wall_s=180),
        'Audit two retained complete-development frame candidates on the original16 training paths. Compare saved teacher-state fit errors with independently executed trajectories. No model update or sealed test.')
    indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53];paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    refs=[np.asarray(paths[i]) for i in indices];started=time.perf_counter()
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0);records=[]
    with np.load(source/'training_data.npz') as z:x=torch.tensor(z['observation']);target=z['actions'];path_id=z['path']
    for model in models:
        saved=torch.load(model,map_location='cpu',weights_only=False);p=FrameFeedbackPolicy(frame=saved['frame']);p.load_state_dict(saved['policy'])
        results=evaluate(p,refs,stage,saved['frame'],None,costs,started+180,environment_factory)
        with torch.no_grad():pred=decoded(p.distribution(x).mean).numpy()
        rows=[]
        for i,index in enumerate(indices):
            with np.load(stage/saved['frame']/f'{i}.npz') as z:
                shape=corner_shape(z['reference'],z['rows'],z['actions']) if len(z['reference'])==3 else None
                if shape:shape['spatial']=geometric_turn_metrics(z['reference'],z['rows'])
                speed=np.linalg.norm(z['rows'][:,2:4],axis=1)
            error=pred[path_id==index]-target[path_id==index]
            rows.append(dict(path=index,success=results[i]['success'],reason=results[i]['reason'],shape=shape,
                teacher_state_action_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),teacher_state_action_abs_q90=np.quantile(abs(error),.9,axis=0).tolist(),
                peak_speed=float(speed.max()),cruise_ge95_s=float((speed>=95).sum()*.001)))
        records.append(dict(model=str(model.relative_to(BASE)),model_sha256=sha(model),successes=sum(r['success'] for r in results),paths=rows))
        print(saved['frame'],records[-1]['successes'],'/16 failed',[r['path'] for r in rows if not r['success']],flush=True)
        print('CORNERS',[(r['path'],round(r['shape']['reverse_turning_total_deg'],2),round(r['shape']['spatial']['bending_energy_mm_inv'],2)) for r in rows if r['shape']],flush=True)
    assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=150000
    write_json(stage/'results.json',dict(records=records,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_training_reproduction_audit',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
