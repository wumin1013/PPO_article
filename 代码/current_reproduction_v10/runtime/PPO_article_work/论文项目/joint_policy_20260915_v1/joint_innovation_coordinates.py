"""Three-action coordinates normalized by the observable combined-target change."""
import math
import numpy as np
from joint_runtime import wrap
from probe_joint_target_coordinates import guide_angle


def anticipation_command_features(observation):
    directions=np.asarray(observation)[42:62].reshape(-1,2)
    groups=(directions[:3],directions[3:6],directions[6:])
    return np.array([math.atan2(g[:,1].mean(),g[:,0].mean())/math.pi for g in groups])


def innovation_coordinate_action(env,parent_latent,expert_latent,weights,independent_residual=None):
    weights=np.asarray(weights,dtype=float)
    extra=np.zeros(3) if independent_residual is None else np.asarray(independent_residual)
    if not np.any(weights) and not np.any(extra):return parent_latent.copy()
    action=parent_latent.astype(float).copy()
    action[1:]+=weights[1:]*(expert_latent-parent_latent)[1:]
    if np.any(extra[1:]):action[1:]+=extra[1:]
    if weights[0] or extra[0]:
        requested=env.decode_action(action,env.fixed_d)
        previous=env.held.copy()
        jump=wrap(guide_angle(env,requested)+requested[0]-guide_angle(env,previous)-previous[0])
        unit=(requested[0]-weights[0]*jump+extra[0])/(math.pi/2)
        # The existing steering actuator range is unchanged.
        action[0]=np.arctanh(np.clip(unit,-1.+1e-9,1.-1e-9))
    return action.astype(parent_latent.dtype)
