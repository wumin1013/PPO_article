"""Development review after identifying a threshold-dependent training rejection."""
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from isolated_corner_shape import first_run
from probe_constrained_parameter_step import summarize
from train_pilot import evaluate
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def onset(reference,rows):
    u,v=np.diff(reference,axis=0);u/=np.linalg.norm(u)
    normal=np.sign(u[0]*v[1]-u[1]*v[0])*np.array([-u[1],u[0]])
    inside=(rows[:,:2]-reference[1])@normal
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u)))
    inbound=np.arange(len(rows))<=np.argmin(np.linalg.norm(rows[:,:2]-reference[1],axis=1))
    def direction(pos,neg):
        found=[(int(x[0]),label) for x,label in [(np.flatnonzero(pos),'inside'),(np.flatnonzero(neg),'outside')] if len(x)]
        return min(found)[1] if found else None
    return dict(displacement_first={str(t):direction(inbound&(inside>t),inbound&(inside<-t)) for t in [.0001,.001,.005,.01,.02]},
        heading_first={str(t):direction(inbound&(speed>5)&(heading>t),inbound&(speed>5)&(heading<-t)) for t in [.05,.1,.2,.5,1.]})


def main():
    source=BASE/'matched_endpoint_representation_learning_20260916'
    models=['full_features_50.pt','mask_endpoint_bearing_50.pt']
    stage=register('endpoint_representation_shape_review_20260916',
        [__file__,source/'receipt.json',BASE/'turn_onset_label_audit_20260916/receipt.json',BASE/'turn_onset_label_audit_20260916/results.json']+[source/name for name in models],
        dict(new_training_decisions=0,evaluation_decisions=2200,physics_ticks=50000,wall_s=120),
        'Supplement one matched checkpoint per arm after saved-training XY audit found that the0.5deg '
        'first-turn indicator contradicts finer heading thresholds and actual first lateral displacement. '
        'Both checkpoints are the first50-update models with16/16 training completion. Selection is fixed '
        'here before any development run; keep both arms including weaker shape. Do not erase original '
        'labels or declare shape acceptance from an alternate threshold. Inspect true XY, onset sensitivity, '
        'outward amplitude/duration, reversal, curvature, cruise and complete physical completion. No sealed test.')
    write_json(stage/'selection.json',dict(models=models,training_only_selection=True,
        reason='Matched earliest50-update checkpoints; masked has reduced raw training excursions and total reversal, while discrete onset label caused artificial rejection.'))
    torch.set_num_threads(1);started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    paths=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['validation']];results=[]
    for name in models:
        saved=torch.load(source/name,map_location='cpu',weights_only=False)
        policy=LocalGeometryFeedbackPolicy(mask_endpoint=saved['mask_endpoint']);policy.load_state_dict(saved['policy'])
        tag=name[:-3];evaluated=evaluate(policy,paths,stage,tag,None,costs,started+120,environment_factory)
        row=summarize(stage,tag,evaluated,list(range(12)));row.update(model=name,model_sha256=sha(source/name),final_acceptance=False)
        for i in range(4):
            with np.load(stage/tag/f'{i}.npz') as z:row['paths'][i]['onset_sensitivity']=onset(z['reference'],z['rows'])
        results.append(row);write_json(stage/'results.json',dict(models=results,costs=costs,goal_complete=False))
        print('ENDPOINT_REVIEW',name,row['successes'],row['full_time_s'],flush=True)
        for r in row['paths'][:4]:
            s=r['shape'];print(r['path'],s['first_physical_turn'],s['outward_excursion_mm'],s['reverse_turning_total_deg'],s['spatial']['middle80_turn_arc_length_mm'],r['onset_sensitivity']['displacement_first'],flush=True)
        print('CRUISE',row['paths'][-1],flush=True)
    assert costs['evaluation_actor_calls']<=2200 and costs['evaluation_physics_ticks']<=50000
    receipt(stage,dict(status='completed_matched_development_shape_review',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
