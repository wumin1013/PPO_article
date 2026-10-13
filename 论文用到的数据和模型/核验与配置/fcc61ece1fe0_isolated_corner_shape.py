"""Signed physical-XY diagnostics for isolated turns; no controller rules.

Thresholds only suppress near-zero numerical motion in the reported onset.
Raw extrema accompany thresholds. They are not a replacement for XY review.
"""
import numpy as np


def first_run(mask, count=20):
    indices = np.flatnonzero(np.convolve(np.asarray(mask, int), np.ones(count, int), 'valid') == count)
    return int(indices[0]) if len(indices) else None


def corner_shape(reference, rows, actions):
    ref, rows, actions = map(np.asarray, (reference, rows, actions))
    if len(ref) != 3:
        raise ValueError('Only isolated three-vertex paths have a single signed inside')
    tangent = ref[1]-ref[0]
    tangent = tangent/np.linalg.norm(tangent)
    outgoing = ref[2]-ref[1]
    outgoing = outgoing/np.linalg.norm(outgoing)
    turn = np.arctan2(np.linalg.det(np.stack([tangent, outgoing])), tangent@outgoing)
    sign = np.sign(turn)
    if not sign:
        raise ValueError('A nonzero corner is required')
    normal = sign*np.array([-tangent[1], tangent[0]])
    xy = rows[:, :2]-ref[1]
    x, inside = xy@tangent, xy@normal
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    vt, vn = rows[:, 2:4]@tangent, rows[:, 2:4]@normal
    heading = np.unwrap(np.arctan2(vn, vt))
    apex = int(np.argmin(np.linalg.norm(xy, axis=1)))
    inbound = np.arange(len(rows)) <= apex
    moving = speed > 5.
    positive = inbound & moving & (heading > np.deg2rad(.5))
    negative = inbound & moving & (heading < -np.deg2rad(.5))
    start_in, start_out = first_run(positive), first_run(negative)
    first = min([i for i in (start_in, start_out) if i is not None], default=None)
    reverse_y = inbound & (inside < -.02)
    reverse_index = np.flatnonzero(reverse_y)
    kappa = np.zeros(len(rows))
    cross = rows[:, 2]*rows[:, 5]-rows[:, 3]*rows[:, 4]
    kappa[moving] = sign*cross[moving]/speed[moving]**3
    # Physical turning is measured until 10 mm past closest approach, before stopping.
    distance = np.cumsum(speed*.001)
    window = moving & (distance >= max(0., distance[apex]-15.)) & (distance <= distance[apex]+10.)
    omega = kappa*speed
    signed_actions = sign*actions[:, 0]
    first_command = np.flatnonzero(np.abs(signed_actions) > np.deg2rad(.5))
    result = dict(turn_deg=float(np.rad2deg(turn)),
        first_physical_turn=('inside' if first == start_in else 'outside') if first is not None else None,
        first_physical_turn_s=(first+1)*.001 if first is not None else None,
        first_inward_heading_s=(start_in+1)*.001 if start_in is not None else None,
        first_outward_heading_s=(start_out+1)*.001 if start_out is not None else None,
        first_signed_steering_deg=float(np.rad2deg(signed_actions[first_command[0]])) if len(first_command) else None,
        outward_excursion_mm=float(max(0., -inside[inbound].min())),
        outward_position_duration_s=float(reverse_y.sum()*.001),
        outward_position_span_s=float((reverse_index[-1]-reverse_index[0]+1)*.001) if len(reverse_index) else 0.,
        reverse_heading_duration_s=float(negative.sum()*.001),
        min_signed_heading_deg=float(np.rad2deg(heading[inbound & moving].min())),
        signed_curvature_min_mm_inv=float(kappa[window].min()),
        signed_curvature_max_mm_inv=float(kappa[window].max()),
        reverse_turning_total_deg=float(np.rad2deg(-np.minimum(omega[window], 0.).sum()*.001)),
        total_abs_turning_deg=float(np.rad2deg(np.abs(omega[window]).sum()*.001)),
        apex_distance_mm=float(np.linalg.norm(xy[apex])),
        apex_speed_mm_s=float(speed[apex]),
        speed_peak_mm_s=float(speed.max()),
        cruise_ge95_duration_s=float((speed >= 95.).sum()*.001),
        approach_inside_max_mm=float(inside[inbound].max()),
        shape_screen_pass=bool(first == start_in and start_in is not None and max(0., -inside[inbound].min()) <= .05))
    return result
