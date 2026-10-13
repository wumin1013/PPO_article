"""Preserve normal tracking; increase tangential response during straight acceleration."""
import numpy as np


def install(*, omega, dt=.001, tolerance=1.25):
    from runtime_physical_time_v5 import install as install_base
    settings = install_base(dt=dt, tolerance=tolerance)
    import joint_runtime as joint
    tracking = joint.tracking
    original = tracking.compute_tracking_demand
    if omega not in (40.,60.,80.):
        raise ValueError('Unregistered straight-tracking bandwidth')
    if omega != 40.:
        def demand(velocity, acceleration, desired, guidance_rate):
            velocity, acceleration, jerk, diagnostics = original(
                velocity, acceleration, desired, guidance_rate)
            speed = float(np.linalg.norm(velocity))
            target_speed = float(np.linalg.norm(desired))
            weight = 0.
            if target_speed > speed and target_speed > 1e-9:
                direction = desired/target_speed
                normal_speed = float(np.linalg.norm(velocity-np.dot(velocity,direction)*direction))
                # Smoothly remove the extra tangential demand during steering.
                rate_weight = max(0.,1.-(abs(guidance_rate)/.1)**2)
                angle_weight = max(0.,1.-(normal_speed/(max(speed,1.)*.01))**2)
                weight = rate_weight*angle_weight
                extra = (omega**2-1600.)*float(np.dot(desired-velocity,direction))
                extra -= (2.*omega-80.)*float(np.dot(acceleration,direction))
                jerk = jerk+weight*extra*direction
            diagnostics['tracking_desired_jerk_mm_s3'] = float(np.linalg.norm(jerk))
            diagnostics['straight_tangential_weight'] = weight
            return velocity,acceleration,jerk,diagnostics
        tracking.compute_tracking_demand = demand
    settings.update(version='v9_straight_tangential',tracking_omega_s_inv=omega,
        base_tracking_omega_s_inv=40.,normal_tracking_unchanged=True,
        straight_rate_cutoff_rad_s=.1,straight_heading_sine_cutoff=.01,
        kcm_unchanged=True,terminal_brake_unchanged=True,shared_by=['JNN','FIXED'])
    return settings
