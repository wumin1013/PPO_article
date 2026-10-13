"""Run the established PPO sampler with an explicit two-output actor factory."""
from localized_two_action_policy import policy_from
import train_mixed_resolution_ppo as common

_physical_worker=common.physics_worker


def physics_worker(connection,dt,args,stage,deadline):
    # Spawned processes install the same strict factory before any model loads.
    common.policy_from=policy_from
    _physical_worker(connection,dt,args,stage,deadline)


if __name__=='__main__':
    common.policy_from=policy_from
    common.physics_worker=physics_worker
    common.main()
