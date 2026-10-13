"""Continuous future-geometry features; no requested action or speed rule."""
import math
import numpy as np


def anticipation_features(observation,innovation):
    x=np.asarray(observation)
    directions=x[42:62].reshape(10,2).astype(float)
    directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
    valid=np.clip(x[62:72],0.,1.)
    origin=directions[0]
    bend=.5*(1.-np.clip(directions@origin,-1.,1.))
    cross=origin[0]*directions[:,1]-origin[1]*directions[:,0]
    def average(values,start,stop):
        weights=valid[start:stop]
        return float(values[start:stop]@weights/max(weights.sum(),1e-12))
    return np.array([average(bend,0,6),average(bend,6,10),
                     average(cross,0,10)*math.tanh(float(innovation[1])/.5)])


def main():
    import torch
    from study_io import BASE,write_json,sha
    from search_complete_return import read
    from deep_joint_credit_policy import load_credit_policy
    from decision_innovation_policy import innovation_features
    from canonical_gate_state import checkpoint_environment_factory
    source=BASE/'canonical_gate_band_shared_temporal_joint_learning_20260919'
    parent_file=BASE/'canonical_gate_band_target_jump_teacher_state_distillation_20260919/research_candidate.pt'
    policy=load_credit_policy(torch.load(parent_file,map_location='cpu',weights_only=False))
    factory=checkpoint_environment_factory('gate_band_canonical_kernel')
    data=read(source/'baseline.json')['baseline']['records']
    results=[]
    for path in [14,104,204,206]:
        r=next(r for r in data if r['path']==path)
        with np.load(r['trace']) as z:ref,rows,actions,obs=[z[k].copy() for k in ('reference','rows','actions','observations')]
        apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
        end=max(20,(apex//20-6)*20)
        env=policy.bind(factory(ref));tick=0;old=[];future=[]
        for index,x in enumerate(obs):
            frame=index*20
            if frame>=end:break
            while tick<frame:
                env.position=rows[tick,:2].copy();env.velocity=rows[tick,2:4].copy()
                env.acceleration=rows[tick,4:6].copy();env.heading=math.atan2(rows[tick,9],rows[tick,8])
                env.tracker.update(env.position);tick+=1
            if index:env.held=actions[index-1].copy()
            env.ticks=frame;env.decisions=index
            latent=policy.act(x,True)[0]
            assert np.array_equal(env.decode_action(latent),actions[index])
            _,inputs=innovation_features(env,latent)
            old.append(np.tanh(inputs/.5)**2);future.append(anticipation_features(x,inputs))
        old,future=np.asarray(old),np.asarray(future)
        result=dict(path=path,window_ms=[0,end],decisions=len(old),
            median_existing_features=np.median(old,axis=0).tolist(),
            median_anticipation_features=np.median(future,axis=0).tolist(),
            weak_reactive_but_visible_future_decisions=int(((old.max(1)<.001)&(future[:,:2].max(1)>.05)).sum()))
        results.append(result);print('ANTICIPATION_AUDIT',result,flush=True)
    write_json(BASE/'canonical_gate_band_joint_expert_residual_learning_20260920/anticipation_feature_audit.json',
        dict(source_model_sha256=sha(parent_file),records=results,thresholds_are_diagnostic_only=True,
             physical_rollouts_repeated=0,not_a_causal_sufficiency_claim=True))


if __name__=='__main__':main()
