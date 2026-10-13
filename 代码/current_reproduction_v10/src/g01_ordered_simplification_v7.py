"""Bounded, order-preserving G01 chord merging; no reference gate is removed.

Monotone projection and bounded transverse deviation on each accepted source
interval certify a two-way geometric correspondence between the original
polyline and its replacement chord. The original complete program remains the
reference for command acceptance; this is planner geometry preprocessing only.
"""
import numpy as np

FRACTION = .1
_INSTALLED = False


def simplify(reference, tolerance):
    reference = np.asarray(reference, float)
    if (reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2
            or not np.isfinite(reference).all() or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Finite complete planar reference and positive tolerance required')
    delta = FRACTION * float(tolerance)
    accepted = []
    pending = [(0, len(reference)-1)]
    while pending:
        left, right = pending.pop()
        points = reference[left:right+1]
        vector = points[-1]-points[0]
        norm2 = float(vector @ vector)
        if norm2 <= 1e-20:
            if right-left < 2:
                raise ValueError('Zero-length original block is unsupported')
            split = left + int(np.argmax(np.linalg.norm(points-points[0], axis=1)))
            if split in (left, right):
                raise ValueError('Source interval has no nonzero chord')
            pending.extend(((split, right), (left, split)))
            continue
        projection = (points-points[0]) @ vector / norm2
        residual = points-points[0]-projection[:, None]*vector
        distance = np.linalg.norm(residual, axis=1)
        # Strict monotonicity is not required: a source segment may project to
        # one chord point, but the complete projected image must remain ordered.
        ordered = bool(np.all(np.diff(projection) >= 0.) and projection.min() >= 0.
                       and projection.max() <= 1.+2e-15)
        maximum = float(distance.max(initial=0.))
        if ordered and maximum <= delta:
            accepted.append(dict(start_reference_index=left, end_reference_index=right,
                maximum_transverse_deviation_mm=maximum,
                minimum_projection_increment=float(np.diff(projection).min()),
                projected_parameter_range=[float(projection.min()), float(projection.max())],
                two_way_ordered_correspondence=True))
            continue
        if right-left == 1:
            raise ValueError('Numerically invalid original source segment')
        offset = int(np.argmax(distance))
        if not ordered or offset in (0, right-left):
            offset = (right-left)//2
        split = left+offset
        pending.extend(((split, right), (left, split)))
    accepted.sort(key=lambda item:item['start_reference_index'])
    if (accepted[0]['start_reference_index'] != 0
            or accepted[-1]['end_reference_index'] != len(reference)-1
            or any(a['end_reference_index'] != b['start_reference_index']
                   for a, b in zip(accepted[:-1], accepted[1:]))):
        raise AssertionError('Simplification must cover every original source interval exactly once')
    indices = [accepted[0]['start_reference_index']] + [x['end_reference_index'] for x in accepted]
    reduced = reference[indices].copy()
    if not np.array_equal(reduced[[0, -1]], reference[[0, -1]]):
        raise AssertionError('Finite traversal endpoints changed')
    certificate = dict(version='v7-fixed-ten-percent-ordered-G01-merging',
        simplification_fraction=FRACTION, effective_original_tolerance_mm=float(tolerance),
        simplification_budget_mm=delta, smoothing_budget_mm=float(tolerance)-delta,
        input_vertices=len(reference), planner_vertices=len(reduced), retained_source_indices=indices,
        maximum_merged_deviation_mm=max(x['maximum_transverse_deviation_mm'] for x in accepted),
        complete_source_interval_partition=True, finite_traversal_endpoints_preserved=True,
        reference_closed=bool(np.linalg.norm(reference[0]-reference[-1]) < 1e-9),
        proof='On each source interval, chord projection is monotone and covers[0,1]. Linear interpolation preserves the transverse bound between source vertices. Every chord coordinate therefore has an ordered source point at distance no greater than delta, and every source point has its chord projection at that bound.',
        final_audit='Always use the complete original reference and every original directed gate; simplified gates never substitute for final acceptance.',
        intervals=accepted)
    return reduced, certificate


def install():
    """Change planner geometry only; outer original-reference audits stay intact."""
    global _INSTALLED
    if _INSTALLED:
        return
    from baselines import legacy_s2, ruckig_s2, motion_overlap
    import copp_adapter_v5b
    native_s2 = legacy_s2.prepare_geometry
    native_copp = copp_adapter_v5b.prepare_geometry
    native_scmo = motion_overlap.plan

    def prepare(native):
        def wrapped(reference, *, tolerance=1.25, source_geometry=None, **kwargs):
            reduced, certificate = simplify(reference, tolerance)
            curve, *rest = native(reduced, tolerance=certificate['smoothing_budget_mm'],
                                  source_geometry=None, **kwargs)
            curve.reference = np.asarray(reference, float)
            curve.metadata.update(g01_preprocessing=certificate,
                audit_reference='Complete original caller reference; every original gate retained',
                source_representation='Tolerance-budgeted ordered G01 chord merging; no fitted analytic arcs',
                planning_reference=reduced.tolist())
            return (curve, *rest)
        return wrapped

    s2_wrapper = prepare(native_s2)
    legacy_s2.prepare_geometry = s2_wrapper
    ruckig_s2.prepare_geometry = s2_wrapper
    copp_adapter_v5b.prepare_geometry = prepare(native_copp)

    def scmo(reference, *, tolerance=1.25, **kwargs):
        reduced, certificate = simplify(reference, tolerance)
        result = native_scmo(reduced, tolerance=certificate['smoothing_budget_mm'], **kwargs)
        result['metrics'].update(g01_preprocessing=certificate,
            audit_reference='Complete original caller reference; every original gate retained',
            planning_reference=reduced.tolist())
        return result
    motion_overlap.plan = scmo
    _INSTALLED = True
