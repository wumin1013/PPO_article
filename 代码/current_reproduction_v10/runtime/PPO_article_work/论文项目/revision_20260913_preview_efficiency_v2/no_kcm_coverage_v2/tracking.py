"""Common tracking demand and separately switchable kinematic constraints."""
import numpy as np


def ball(point, center, radius):
    displacement = point - center
    magnitude = float(np.linalg.norm(displacement))
    return point if magnitude <= radius else center + displacement * (radius / magnitude)


def compute_tracking_demand(velocity, acceleration, desired, guidance_rate):
    velocity, acceleration, desired = [np.asarray(x, float) for x in
                                       (velocity, acceleration, desired)]
    requested_ff = float(guidance_rate) * np.array([-desired[1], desired[0]])
    # Common target regularization, not a bound on actual no-KCM acceleration.
    applied_ff = ball(requested_ff, np.zeros(2), 1998.)
    desired_jerk = 1600. * (desired - velocity) - 80. * (acceleration - applied_ff)
    diagnostics = dict(normal_ff_requested_mm_s2=float(np.linalg.norm(requested_ff)),
        normal_ff_applied_mm_s2=float(np.linalg.norm(applied_ff)),
        filtered_guidance_rate_rad_s=float(guidance_rate),
        tracking_desired_jerk_mm_s3=float(np.linalg.norm(desired_jerk)))
    return velocity, acceleration, desired_jerk, diagnostics


def enforce_kinematic_constraints(velocity, acceleration, desired_jerk, dt):
    """Original FF candidate clipping and original three-ball intersection."""
    candidate_a = ball(acceleration + dt * ball(desired_jerk, np.zeros(2), 62437.5),
                       np.zeros(2), 1998.)
    candidate = velocity + dt * candidate_a
    corrections = np.zeros((3, 2))
    centers = (np.zeros(2), velocity, velocity + acceleration * dt)
    radii = (100., 1998. * dt, 62437.5 * dt * dt)
    for _ in range(32):
        previous = candidate.copy()
        for index, (center, radius) in enumerate(zip(centers, radii)):
            temporary = candidate + corrections[index]
            candidate = ball(temporary, center, radius)
            corrections[index] = temporary - candidate
        if np.linalg.norm(candidate - previous) < 1e-12:
            break
    next_a = (candidate - velocity) / dt
    next_j = (next_a - acceleration) / dt
    ratios = [np.linalg.norm(candidate) / 100., np.linalg.norm(next_a) / 2000.,
              np.linalg.norm(next_j) / 62500.]
    if not np.isfinite(ratios).all() or max(ratios) > 1.000001:
        raise ValueError('Planar KCM intersection infeasible: ' + str(ratios))
    return candidate, next_a, next_j


def tracking_step(velocity, acceleration, desired, guidance_rate, dt, *, kcm_enabled):
    velocity, acceleration, desired_j, diagnostics = compute_tracking_demand(
        velocity, acceleration, desired, guidance_rate)
    if kcm_enabled:
        candidate, next_a, next_j = enforce_kinematic_constraints(
            velocity, acceleration, desired_j, dt)
    else:
        # Remove both candidate a/j clipping and feasible-set projection only.
        candidate_a = acceleration + dt * desired_j
        candidate = velocity + dt * candidate_a
        next_a = (candidate - velocity) / dt
        next_j = (next_a - acceleration) / dt
        if not np.isfinite(np.r_[candidate, next_a, next_j]).all():
            raise ValueError('Nonfinite unconstrained tracking integration')
    diagnostics.update(pure_kcm_enabled=float(kcm_enabled),
        no_kcm=float(not kcm_enabled), common_tracking_and_ff_retained=1.)
    return candidate, next_a, next_j, diagnostics
