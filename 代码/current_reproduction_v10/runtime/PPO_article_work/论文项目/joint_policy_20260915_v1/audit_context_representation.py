"""Non-vacuous saved-state checks for both context arms, beyond the empty query audit."""
import numpy as np
import torch
from observable_memory_joint_policy import actor_features
from reflection_policy import reflect_observation
from study_io import BASE,write_json,sha


def main():
    torch.set_num_threads(1)
    stage=BASE/'canonical_gate_band_context_coherent_parameter_training_20260920'
    source=BASE/'canonical_gate_band_dual_coherent_parameter_training_20260920'
    with np.load(source/'adaptive_dual_actor_replay.npz') as z:obs,innov=z['observations'].copy(),z['innovations'].copy()
    reflected=reflect_observation(torch.as_tensor(obs)).numpy()
    rows=[]
    for mode in ('geometry','dynamic'):
        file=stage/(mode+'_initial_policy.pt');md=torch.load(file,map_location='cpu',weights_only=False)
        actual=np.array([actor_features(md,x,y) for x,y in zip(obs,innov)])
        no_innovation=np.array([actor_features(md,x,np.zeros(3)) for x in obs])
        mirror=np.array([actor_features(md,x,-y) for x,y in zip(reflected,innov)])
        assert np.array_equal(actual,no_innovation),('innovation_independence',mode)
        assert np.array_equal(mirror,actual*np.array([-1.,1.,1.])),('reflection',mode)
        rows.append(dict(mode=mode,model_sha256=sha(file),checked_states=len(obs),
            full_feature_reflection_bitwise=True,innovation_independence_bitwise=True))
    write_json(stage/'context_representation_audit.json',dict(arms=rows,new_physical_runs=0,
        earlier_helper_query_count=0,earlier_helper_query_assertions_were_vacuous=True,
        current_audit_checks_every_saved_state=True))
    print('CONTEXT_REPRESENTATION_EXACT',len(obs),'states per arm',flush=True)


if __name__=='__main__':main()
