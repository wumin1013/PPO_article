"""Independent analytic line/arc distance checks on raw commands, without XY filtering."""
import numpy as np


def analytic_distance(points, geometry):
    points = np.asarray(points, float)
    best = np.full(len(points), np.inf)
    for p in geometry['primitives']:
        if p['kind'] == 'line':
            a, b = np.asarray(p['start_mm']), np.asarray(p['end_mm'])
            edge = b-a
            u = np.clip(((points-a)*edge).sum(1)/(edge@edge), 0., 1.)
            distance = np.linalg.norm(points-a-u[:, None]*edge, axis=1)
        else:
            relative = points-np.asarray(p['center_mm'])
            angle = np.arctan2(relative[:, 1], relative[:, 0])
            direction = np.sign(p['sweep_rad'])
            phase = (direction*(angle-p['start_angle_rad'])) % (2*np.pi)
            radial = np.abs(np.linalg.norm(relative, axis=1)-p['radius_mm'])
            endpoints = np.minimum(np.linalg.norm(points-p['start_mm'], axis=1),
                                   np.linalg.norm(points-p['end_mm'], axis=1))
            distance = np.where(phase <= abs(p['sweep_rad'])+1e-12, radial, endpoints)
        best = np.minimum(best, distance)
    return best


def analytic_command_audit(initial, rows, geometry, epsilon=1.25):
    """Use the distance-to-set Lipschitz bound to certify segment interiors.

    This certifies the piecewise-linear position command geometry, not continuous
    acceleration/jerk of linear interpolation. Kinematics are audited separately.
    """
    xy = np.vstack([initial, rows[:, :2]])
    distance = analytic_distance(xy, geometry)
    a, b, da, db = xy[:-1], xy[1:], distance[:-1], distance[1:]
    maximum = float(distance.max())
    levels = 0; queries = len(xy)
    unresolved = 0
    for levels in range(15):
        # Any point along a segment is bounded from its nearer endpoint.
        upper = .5*(da+db+np.linalg.norm(b-a, axis=1))
        pending = (upper > epsilon) & (np.maximum(da, db) <= epsilon)
        if not np.any(pending):
            break
        a, b, da, db = a[pending], b[pending], da[pending], db[pending]
        middle = .5*(a+b); dm = analytic_distance(middle, geometry)
        maximum = max(maximum, float(dm.max())); queries += len(dm)
        a, b, da, db = np.vstack([a, middle]), np.vstack([middle, b]), np.r_[da, dm], np.r_[dm, db]
    else:
        unresolved = int(np.sum(.5*(da+db+np.linalg.norm(b-a, axis=1)) > epsilon))
    return dict(analytic_corridor_pass=maximum <= epsilon and unresolved == 0,
                maximum_evaluated_distance_mm=maximum, unresolved_segments=unresolved,
                subdivision_levels=levels, distance_queries=queries,
                geometry_scope='raw positions and complete straight command intervals; no added chord tolerance',
                kinematic_scope='not certified here; independent vector dynamics audit required')
