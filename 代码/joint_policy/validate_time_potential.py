"""Finite algebra checks for time-scaled potential shaping, without execution."""
import math
import numpy as np
from study_io import BASE,register,write_json,receipt

if __name__=='__main__':
    stage=register('time_potential_validation_v3',[__file__,BASE/'joint_runtime_v3.py',BASE/'reward_conditioning_design_v3.md'],
        dict(algebra_cases=6,distance_roundtrips=7,physics_ticks=0,models=0),
        'Validate success-time ordering, partial-trajectory telescoping and invertible remaining distance; no model or physical evaluation.')
    results=[]
    for length,progress,durations,success in [
        (40.,[0,10,20,30,39.2],[.3,.3,.3,.4],True),
        (40.,[0,10,9,28,39.1],[.3,.1,.7,.4],True),
        (34922.97,[0,120,23420,34922.2],[3.,230.,118.],True),
        (40.,[0,10,20],[.3,.3],False),
        (40.,[0,10,20,30,39.2],[.2,.2,.2,.3],True),
        (40.,[0,10,20,30,40],[.2,.2,.2,.4],True)]:
        phi=(np.asarray(progress)-length)/100.
        reward=-np.asarray(durations)+np.diff(np.r_[phi[:-1],0.])
        penalty=0. if success else 6.
        reward[-1]-=penalty
        expected=-sum(durations)+length/100.-penalty
        assert abs(float(reward.sum())-expected)<1e-9
        if len(durations)>=3:
            split=2
            partial=float((-np.asarray(durations[:split])+np.diff(phi[:split+1])).sum())
            rest=float((-np.asarray(durations[split:])+np.diff(np.r_[phi[split:-1],0.])).sum())-penalty
            assert abs(partial+rest-expected)<1e-9
        results.append(dict(length=length,success=success,time_s=sum(durations),return_sum=float(reward.sum()),expected=expected))
    assert results[4]['return_sum']>results[0]['return_sum']
    distances=[0.,.8,3.,30.,55.,350.,40000.]
    for distance in distances:
        observation=math.log1p(distance/30.)/8.
        recovered=30.*math.expm1(8.*observation)
        assert abs(recovered-distance)<=1e-8 and 0<=observation<1.
    write_json(stage/'results.json',dict(cases=results,remaining_distances_mm=distances,
        success_time_ordering_preserved=True,partial_bootstrap_identity=True,
        policy_improvement_proven=False))
    receipt(stage,dict(status='finite_reward_algebra_checks_passed',algebra_cases=6,distance_roundtrips=7,
        physics_ticks=0,models=0,training_updates=0))
    print({'algebra_cases_passed':6,'distance_roundtrips':7,'training_updates':0})
