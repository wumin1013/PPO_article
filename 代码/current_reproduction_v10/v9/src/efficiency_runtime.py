"""Versioned tracking-bandwidth study; unchanged projection and completion."""
import numpy as np

VERSION = 'v9_tracking_bandwidth'


def install(*, omega, dt=.001, tolerance=1.25):
    from runtime_physical_time_v5 import install as base_install
    settings = base_install(dt=dt, tolerance=tolerance)
    import joint_runtime as joint
    tracking = joint.tracking
    if omega not in (40., 60., 80.):
        raise ValueError('Unregistered tracking bandwidth')
    original = tracking.compute_tracking_demand
    if omega != 40.:
        def demand(velocity, acceleration, desired, guidance_rate):
            velocity, acceleration, _, diagnostics = original(
                velocity, acceleration, desired, guidance_rate)
            requested_ff = float(guidance_rate)*np.array([-desired[1], desired[0]])
            applied_ff = tracking.ball(requested_ff, np.zeros(2), 1998.)
            jerk = omega**2*(desired-velocity)-2.*omega*(acceleration-applied_ff)
            diagnostics['tracking_desired_jerk_mm_s3'] = float(np.linalg.norm(jerk))
            return velocity, acceleration, jerk, diagnostics
        tracking.compute_tracking_demand = demand
    settings.update(version=VERSION, tracking_omega_s_inv=omega,
        tracking_damping_ratio=1., weights_unchanged=True,
        kcm_unchanged=True, terminal_brake_unchanged=True,
        shared_by=['JNN', 'FIXED'])
    return settings
