"""Stable source-line derivatives for numerical S2 trim remnants.

No position query, path length or control point is changed. For a line shorter
than1e-10mm, recover its intended tangent from the original untrimmed G01 source
instead of dividing a cancellation-dominated coordinate difference by its norm.
Both adjacent curve tangents must agree with the recovered direction.
"""
import numpy as np
import copp_station_coalescing_v7 as previous

THRESHOLD_MM = 1e-10
_INSTALLED = False


def grid_geometry(curve, source_geometry, settings):
    import copp_adapter_v5b as parent
    pieces = parent.core.flatten(curve)
    reference = np.asarray(curve.metadata['planning_reference'], float)
    vectors = np.diff(reference, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    arcs = np.r_[0., np.cumsum(lengths)]
    native = parent.signed_derivatives
    overrides, receipts = {}, []
    for index, piece in enumerate(pieces):
        if piece.kind != 'line' or piece.length > THRESHOLD_MM:
            continue
        middle = .5*(piece.source_start+piece.source_end)
        source_index = int(np.clip(np.searchsorted(arcs, middle, side='right')-1, 0, len(lengths)-1))
        tangent = vectors[source_index]/lengths[source_index]
        adjacent_errors = []
        for other_index, end in ((index-1, True), (index+1, False)):
            if 0 <= other_index < len(pieces):
                other = pieces[other_index]
                if other.kind == 'line' and other.length <= THRESHOLD_MM:
                    raise ValueError('Consecutive unresolved numerical remnants require separate diagnosis')
                d1, _, _ = native(other, np.asarray([other.length if end else 0.]))
                adjacent_errors.append(float(np.linalg.norm(d1[0]-tangent)))
        if max(adjacent_errors, default=0.) > 1e-7:
            raise ValueError('Recovered source direction does not agree with original adjacent curve tangents')
        overrides[id(piece)] = tangent
        receipts.append(dict(piece_index=index, source_segment_index=source_index,
            length_mm=float(piece.length), original_tangent=np.asarray(piece.tangent).tolist(),
            stable_source_tangent=tangent.tolist(), adjacent_tangent_errors=adjacent_errors,
            positions_and_length_changed=False))

    def stable(piece, local):
        tangent = overrides.get(id(piece))
        if tangent is None:
            return native(piece, local)
        shape = (len(np.asarray(local).reshape(-1)), 2)
        return np.broadcast_to(tangent, shape).copy(), np.zeros(shape), np.zeros(shape)
    parent.signed_derivatives = stable
    try:
        result = previous.grid_geometry(curve, source_geometry, settings)
    finally:
        parent.signed_derivatives = native
    curve.metadata.update(numerical_source_tangents=dict(version='v7b', threshold_mm=THRESHOLD_MM,
        positions_and_curve_length_unchanged=True, rows=receipts))
    return result


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    previous.install()
    import copp_adapter_v5b as parent
    parent.grid_geometry = grid_geometry
    _INSTALLED = True
