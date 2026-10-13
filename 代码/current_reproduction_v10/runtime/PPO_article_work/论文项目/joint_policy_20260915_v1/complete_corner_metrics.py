"""Complete isolated-corner XY diagnostics, independent of control and rewards."""
import numpy as np
from geometric_turn_metrics import spatial_turn_profile


def complete_corner_metrics(reference, rows, spacing=.1, *, initial_xy=None):
    ref, rows = np.asarray(reference), np.asarray(rows)
    initial_xy=ref[0] if initial_xy is None else np.asarray(initial_xy)
    if len(ref) != 3:
        raise ValueError('A single nonzero corner is required')
    u, v = np.diff(ref, axis=0)
    u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
    angle = float(np.arctan2(u[0]*v[1]-u[1]*v[0], u@v))
    sign = np.sign(angle)
    if not sign:
        raise ValueError('A nonzero corner is required')
    ni, no = sign*np.array([-u[1],u[0]]), sign*np.array([-v[1],v[0]])
    xy = rows[:,:2]-ref[1]
    inside_in, inside_out = xy@ni, xy@no
    speed = np.linalg.norm(rows[:,2:4], axis=1)
    moving = speed > 5.
    full_heading = np.unwrap(np.arctan2(rows[:,2:4]@ni, rows[:,2:4]@u))
    # Velocity heading while stopped is undefined; do not interpret it as a turn.
    heading = full_heading[moving]
    cross = rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4]
    omega = sign*cross[moving]/speed[moving]**2
    apex = int(np.argmin(np.linalg.norm(xy,axis=1)))
    # Being inside the incoming half-plane does not exclude outward motion.
    lateral_step=np.diff(np.vstack([initial_xy,rows[:,:2]]),axis=0)@ni
    approach=(np.arange(len(rows))<apex)&moving&(rows[:,12]==0)
    significant=approach&(full_heading < -np.deg2rad(.05))
    spatial = spatial_turn_profile(np.vstack([initial_xy,rows[:,:2]]),spacing)
    turns = sign*spatial['turn']
    energy = float((turns**2).sum()/spacing)
    positive = np.maximum(turns,0.); cumulative = np.cumsum(positive)
    locations = spatial['grid'][1:-1]
    width = float(locations[np.searchsorted(cumulative,.9*cumulative[-1])]-
                  locations[np.searchsorted(cumulative,.1*cumulative[-1])]) if cumulative[-1]>1e-8 else None
    result = dict(turn_deg=float(np.rad2deg(angle)),spacing_mm=spacing,
        in_edge_outward_mm=float(max(0.,-inside_in.min())),
        reverse_lateral_total_mm=float(-np.minimum(lateral_step[approach],0.).sum()),
        reverse_lateral_significant_mm=float(-np.minimum(lateral_step[significant],0.).sum()),
        reverse_lateral_significant_duration_s=float(significant.sum()*.001),
        out_edge_outward_mm=float(max(0.,-inside_out.min())),
        in_edge_outward_gt02_duration_s=float((inside_in<-.02).sum()*.001),
        out_edge_outward_gt02_duration_s=float((inside_out<-.02).sum()*.001),
        in_edge_outward_gt005_duration_s=float((inside_in<-.005).sum()*.001),
        out_edge_outward_gt005_duration_s=float((inside_out<-.005).sum()*.001),
        heading_min_deg=float(np.rad2deg(heading.min())) if len(heading) else None,
        heading_max_deg=float(np.rad2deg(heading.max())) if len(heading) else None,
        heading_overshoot_deg=float(max(0.,np.rad2deg(heading.max()-abs(angle)))) if len(heading) else None,
        heading_overshoot_gt05_duration_s=float((heading>abs(angle)+np.deg2rad(.5)).sum()*.001),
        full_physical_reverse_deg=float(np.rad2deg(np.maximum(-omega,0).sum()*.001)),
        full_spatial_reverse_deg=float(np.rad2deg(np.maximum(-turns,0).sum())),
        full_normalized_energy=float(1.25*energy/angle**2),
        full_middle80_turn_arc_mm=width,
        full_spatial_peak_curvature_mm_inv=float(np.abs(turns).max()/spacing),
        apex_distance_mm=float(np.linalg.norm(xy[apex])),
        apex_speed_mm_s=float(speed[apex]),
        speed_peak_mm_s=float(speed.max()),
        cruise_ge95_s=float((speed>=95).sum()*.001),
        cruise_ge98_s=float((speed>=98).sum()*.001),
        cruise_ge99_s=float((speed>=99).sum()*.001),
        definition='Both signed edge half-planes across all recorded XY; no position smoothing. Heading and physical turning use speed>5mm/s; spatial turning includes full path at declared arc spacing.')
    return result


def coverage_costs(original, complete):
    """Diagnostic factorial only; same original weights, bounds and tolerances."""
    old_side = original['incoming']
    two_sides = max(max(complete['in_edge_outward_mm'],complete['out_edge_outward_mm'])-.05,0.)/1.25
    full_reverse = max(complete['full_physical_reverse_deg']-5.,0.)/abs(complete['turn_deg'])
    result={}
    for extent in ('window','full'):
        energy=original['normalized_energy'] if extent=='window' else complete['full_normalized_energy']
        reverse=original['reverse'] if extent=='window' else full_reverse
        for side in ('incoming','two_edges'):
            quality=energy+5*(old_side if side=='incoming' else two_sides)+2*reverse
            result[f'{extent}_{side}']=dict(quality=float(quality),cost=float(-np.expm1(-quality)))
    assert abs(result['window_incoming']['cost']-original['cost'])<1e-12
    return result
