"""Static per-center support radii; no additional observations or temporal state."""
import torch
from localized_two_action_policy import policy_from as legacy_policy_from
from localized_joint_correction import LocalizedJointPolicy


def policy_from(saved, seed):
    if saved.get('policy_class') != 'LocalizedJointVariableRadiusPolicy':
        return legacy_policy_from(saved,seed)
    policy=LocalizedJointPolicy(seed,count=saved['joint_basis_count'],base_count=saved['base_local_basis_count'])
    radius=saved['policy']['actor.radius']
    assert radius.shape==(saved['joint_basis_count'],) and torch.isfinite(radius).all() and (radius>0).all()
    policy.actor.radius=radius.clone()
    policy.load_state_dict(saved['policy'])
    return policy


def physics_worker(connection,dt,args,stage,deadline):
    import train_mixed_resolution_ppo as common
    common.policy_from=policy_from
    common.physics_worker(connection,dt,args,stage,deadline)
