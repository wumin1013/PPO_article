"""Coalesce floating-point station duplicates without removing path geometry.

All derivative states within a coalesced station retain their velocity,
acceleration and jerk constraints. No tolerance or physical limit is enlarged.
"""
import inspect
import numpy as np

_INSTALLED = False


def grid_geometry(curve, source_geometry, settings):
    import copp_adapter_v5b as parent
    core = parent.core
    raw = []
    offset = 0.
    previous = None
    join_maxima = [0., 0.]
    counts = {}
    for index, piece in enumerate(core.flatten(curve)):
        local = parent.local_stations(piece, settings)
        d1, d2, d3 = parent.signed_derivatives(piece, local)
        if previous is not None:
            jumps = [float(np.linalg.norm(previous[0]-d1[0])),
                     float(np.linalg.norm(previous[1]-d2[0]))]
            join_maxima = np.maximum(join_maxima, jumps).tolist()
            if max(jumps) > 1e-7:
                raise ValueError('Original signed geometric C2 join acceptance failed')
        counts[piece.kind] = counts.get(piece.kind, 0)+len(local)-1
        for s, first, second, third in zip(local, d1, d2, d3):
            raw.append((float(offset+s), first, second, third, index))
        previous = (d1[-1], d2[-1])
        offset += piece.length
    groups = []
    for row in raw:
        threshold = max(1e-12, 16*np.spacing(max(1., abs(row[0]))))
        if groups and row[0]-groups[-1][0][0] <= threshold:
            if row[0] < groups[-1][0][0]-threshold:
                raise ValueError('Reversing station order')
            groups[-1].append(row)
        else:
            groups.append([row])
    sigma = np.asarray([group[0][0] for group in groups])
    if abs(sigma[-1]-curve.length) > max(1e-10, 32*np.spacing(curve.length)):
        raise ValueError('Station coverage differs from complete curve')
    sigma[0], sigma[-1] = 0., curve.length
    if np.any(np.diff(sigma) <= 1e-12):
        raise ValueError('Station coalescing did not resolve the original numerical spacing requirement')
    first, second, third = [], [], []
    boundaries, merged = [], []
    for index, group in enumerate(groups):
        initial = group[0]
        first.append(initial[1]); second.append(initial[2]); third.append(initial[3])
        for row in group[1:]:
            jumps = [float(np.linalg.norm(initial[1]-row[1])), float(np.linalg.norm(initial[2]-row[2]))]
            if max(jumps) > 1e-7:
                raise ValueError('Coalesced derivative state exceeds original C2 numerical allowance')
            boundaries.append(dict(index=index, first=row[1], second=row[2], third=row[3],
                velocity_jump=jumps[0], curvature_jump=jumps[1]))
        if len(group) > 1:
            merged.append(dict(index=index, states=len(group),
                original_station_span_mm=float(group[-1][0]-group[0][0]),
                all_derivative_constraints_retained=True))
    scale = settings['spatial_scale_mm']
    curve.metadata.update(grid_version='v7-floating-station-coalescing',
        station_coalescing=dict(rule='Group width <=max(1e-12mm,16ULP); preserve all derivative-state inequalities.',
            raw_station_count=len(raw), station_count=len(groups),
            original_geometric_pieces_removed=0,
            maximum_group_span_mm=max((x['original_station_span_mm'] for x in merged), default=0.),
            groups=merged, original_join_maxima=join_maxima),
        grid_intervals_by_kind=counts)
    return sigma, np.asarray(first), np.asarray(second)*scale, np.asarray(third)*scale**2, boundaries


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    import copp_adapter_v5b as parent
    core = parent.core
    parent.grid_geometry = grid_geometry
    source = inspect.getsource(core.setup_problem)
    marker = '    for row in boundaries:\n'
    if source.count(marker) != 1:
        raise RuntimeError('Underlying constraint constructor differs from reviewed source')
    additions = '''    for row in boundaries:
        con.add_constraint_1st(np.asarray([[vmax**2/float(row["first"]@row["first"])]]), row["index"])
        con.add_constraint_2nd((row["second"]*scale@normals.T)[None,:],
            (row["first"]@normals.T)[None,:], np.full((1,m),amax*factor), row["index"])
'''
    # The unchanged next statement adds each retained state's third-order row.
    namespace = {}
    exec(compile(source.replace(marker, additions), __file__, 'exec'), core.__dict__, namespace)
    core.setup_problem = namespace['setup_problem']
    _INSTALLED = True
