"""Load equal static support widths for both arms without altering old loaders."""
import torch
from localized_two_action_policy import LocalizedTwoPolicy
from variable_radius_recovery_policy import policy_from as previous_policy_from


def policy_from(saved, seed):
    if saved.get('policy_class') != 'LocalizedTwoVariableRadiusPolicy':
        return previous_policy_from(saved, seed)
    policy = LocalizedTwoPolicy(seed, count=saved['local_basis_count'])
    radius = saved['policy']['actor.radius']
    assert radius.shape == (saved['local_basis_count'],)
    assert torch.isfinite(radius).all() and (radius > 0).all()
    policy.actor.radius = radius.clone()
    policy.load_state_dict(saved['policy'])
    return policy


def physics_worker(connection, dt, args, stage, deadline):
    import train_mixed_resolution_ppo as common
    common.policy_from = policy_from
    common.physics_worker(connection, dt, args, stage, deadline)
