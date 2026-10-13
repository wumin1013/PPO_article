"""Signed source-arc extension of the unchanged project S2 geometry builder.

Only the source provider, signed Arc and signed Hermite endpoints are adapted.
The original corner construction, geometric error bound and trim rules remain.
"""
import hashlib
import math
import numpy as np
import baselines.legacy_s2 as legacy


def geometry_container(module, pieces, item, reference, *, settings):
    merged = []
    for piece in pieces:
        if piece.length <= 0:
            continue
        if merged and isinstance(piece, module.Line) and isinstance(merged[-1], module.Line):
            previous = merged[-1]
            if (np.linalg.norm(previous.end-piece.start) > module.MATCH_TOLERANCE_MM or
                    np.linalg.norm(previous.tangent-piece.tangent) > 1e-10):
                raise ValueError('Noncollinear remnants cannot be merged')
            merged[-1] = module.Line(previous.start, piece.end, previous.source_start, piece.source_end)
        else:
            merged.append(piece)
    if not merged:
        raise ValueError('Complete nonempty geometry required')
    for left, right in zip(merged[:-1], merged[1:]):
        if (np.linalg.norm(left.position([left.length])[0]-right.position([0.])[0]) > module.MATCH_TOLERANCE_MM or
                abs(left.source_end-right.source_start) > module.MATCH_TOLERANCE_MM):
            raise ValueError('Geometric/source interval discontinuity')
    curve = module.Compound(merged, metadata=dict(path_id=item['id'], closed=item['closed'],
        source_reference_sha256=item['reference_sha256'], geometry_only=True,
        source_order_semantics='One monotone complete source traversal',
        unused_curve_constant_caps_computed=False, signed_arc_adapter_version='v5b'
        , entry_exit_straight_required=False,
        geometry_settings={key: settings[key] for key in ['corner_error_mm', 'fit', 'inversion_tolerance_mm']}))
    curve.reference = reference
    if (np.linalg.norm(curve.position([0.])[0]-reference[0]) > module.MATCH_TOLERANCE_MM or
            np.linalg.norm(curve.position([curve.length])[0]-reference[-1]) > module.MATCH_TOLERANCE_MM):
        raise ValueError('Original finite traversal endpoints changed')
    return curve


def install_signed_arc(module):
    class SignedArc(module.Arc):
        def __init__(self, center, radius, angle0, angle1, source_start, source_end):
            self.center, self.radius = np.asarray(center, float), float(radius)
            self.angle0, self.angle1 = float(angle0), float(angle1)
            if radius <= 0 or angle1 == angle0:
                raise ValueError('Positive radius and nonzero signed sweep required')
            self.direction = math.copysign(1., angle1-angle0)
            self.length = self.radius*abs(angle1-angle0)
            self.source_start, self.source_end = float(source_start), float(source_end)
            self.metadata = dict(kind=self.kind, center_mm=self.center.tolist(), radius_mm=self.radius,
                angles_rad=[self.angle0, self.angle1], source_interval_mm=[source_start, source_end],
                signed_curvature_mm_inv=self.direction/self.radius)

        def angle(self, sigma):
            return self.angle0+self.direction*np.asarray(sigma, float)/self.radius

        def tangent_at(self, sigma):
            angle = self.angle(np.asarray(sigma, float).reshape(-1))
            return self.direction*np.column_stack((-np.sin(angle), np.cos(angle)))

    module.Arc = SignedArc

    def arc_extension(before, arc, after, free_before, free_after, chord_error):
        if (np.linalg.norm(before.tangent-arc.tangent_at([0.])[0]) > 1e-10 or
                np.linalg.norm(after.tangent-arc.tangent_at([arc.length])[0]) > 1e-10):
            raise ValueError('Declared line-arc interfaces are not tangent joins')
        maximum = min(free_before, free_after, arc.length/2)
        if maximum <= 0:
            raise ValueError('No nonoverlapping source interval for the C2 arc extension')
        budget = module.CORNER_ERROR_MM-chord_error
        attempts = []
        kappa = arc.direction/arc.radius
        for reduction in range(module.MAX_ARC_BLEND_HALVINGS+1):
            h = maximum/(2**reduction)
            entry = module._quintic_hermite(before.position([before.length-h])[0], arc.position([h])[0],
                before.tangent, arc.tangent_at([h])[0], 0., kappa, 2*h,
                [float(before.source_s(before.length-h)), arc.source_start, float(arc.source_s(h))], 'line_to_arc_C2')
            exit_curve = module._quintic_hermite(arc.position([arc.length-h])[0], after.position([h])[0],
                arc.tangent_at([arc.length-h])[0], after.tangent, kappa, 0., 2*h,
                [float(arc.source_s(arc.length-h)), arc.source_end, float(after.source_s(h))], 'arc_to_line_C2')
            def entry_source(u):
                return np.where((u <= .5)[:, None], before.position(before.length-h+2*h*u), arc.position(2*h*u-h))
            def exit_source(u):
                return np.where((u <= .5)[:, None], arc.position(arc.length-h+2*h*u), after.position(2*h*u-h))
            checks = [module._blend_error_bound(entry, entry_source, h, budget),
                      module._blend_error_bound(exit_curve, exit_source, h, budget)]
            attempts.append(dict(h_mm=h, checks=checks))
            if all(check['accepted'] for check in checks):
                pieces = [entry]
                if arc.length > 2*h:
                    pieces.append(SignedArc(arc.center, arc.radius,
                        arc.angle0+arc.direction*h/arc.radius, arc.angle1-arc.direction*h/arc.radius,
                        float(arc.source_s(h)), float(arc.source_s(arc.length-h))))
                pieces.append(exit_curve)
                return h, module.Compound(pieces, metadata=dict(role='declared_arc_with_two_C2_interfaces',
                    original_arc=arc.metadata, source_chord_error_bound_mm=chord_error,
                    connector_error_budget_mm=budget, geometric_attempts=attempts, selected_h_mm=h,
                    selection='First passing original pointwise geometric bound under maximum nonoverlap trim and halving'))
        raise ValueError('Bounded line-arc connector geometry checks failed')
    module._arc_extension = arc_extension


def prepare_geometry(reference, *, tolerance=1.25, limits=(100., 2000., 62500.),
                     corner_scale=1., source_geometry=None, **unused):
    legacy.setup()
    reference = np.asarray(reference, float)
    lengths = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    if reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2 or not np.isfinite(reference).all() or np.any(lengths <= 1e-10):
        raise ValueError('Finite ordered reference without zero-length blocks required')
    source = legacy.WORK/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1'
    module = legacy._module('copp_signed_s2_geometry_v5', source/'geometry.py')
    install_signed_arc(module)
    module.GeometryPath = lambda pieces, item, ref, settings: geometry_container(module, pieces, item, ref, settings=settings)
    module.CORNER_ERROR_MM = tolerance*(1.2447/1.25)*corner_scale
    module.SOURCE_REGISTRATION_SHA = hashlib.sha256(reference.tobytes()).hexdigest()
    arcs = np.r_[0., np.cumsum(lengths)]
    raw = [module.Line(reference[i], reference[i+1], arcs[i], arcs[i+1]) for i in range(len(lengths))]
    chord = 0.
    if source_geometry is not None:
        raw = []
        previous_index = 0
        for p in source_geometry['primitives']:
            lo, hi = p['start_reference_index'], p['end_reference_index']
            if lo != previous_index or hi <= lo:
                raise ValueError('Declared primitive indices are not a complete ordered partition')
            previous_index = hi
            if p['kind'] == 'line':
                a, b = np.asarray(p['start_mm']), np.asarray(p['end_mm'])
                d = b-a
                f = np.sum((reference[lo:hi+1]-a)*d, axis=1)/float(d@d)
                if (np.max(np.linalg.norm(reference[lo:hi+1]-a-f[:, None]*d, axis=1)) > 2e-7 or
                        np.any(np.diff(f) < -1e-9)):
                    raise ValueError('Declared line differs from complete reference')
                raw.append(module.Line(a, b, p['start_s_mm'], p['end_s_mm']))
            elif p['kind'] == 'arc':
                theta = np.linspace(p['start_angle_rad'], p['start_angle_rad']+p['sweep_rad'], hi-lo+1)
                expected = np.asarray(p['center_mm'])+p['radius_mm']*np.column_stack((np.cos(theta), np.sin(theta)))
                if np.max(np.linalg.norm(expected-reference[lo:hi+1], axis=1)) > 2e-7:
                    raise ValueError('Declared arc differs from complete reference')
                raw.append(module.Arc(p['center_mm'], p['radius_mm'], p['start_angle_rad'],
                    p['start_angle_rad']+p['sweep_rad'], p['start_s_mm'], p['end_s_mm']))
                chord = max(chord, p.get('chord_error_bound_mm', 0.))
            else:
                raise ValueError('Unsupported declared primitive')
        if previous_index != len(reference)-1:
            raise ValueError('Incomplete declared primitive traversal')
    extrapolated = []
    original_corner = module._corner
    def corner(a, b):
        phi = math.atan2(abs(float(module.cross(a.tangent, b.tangent))), float(a.tangent@b.tangent))
        if phi <= 1e-12 or math.degrees(phi) >= 30.-1e-7:
            return original_corner(a, b)
        theta = math.pi-phi
        ratio = theta**.9927/2.0769
        d = 16*module.CORNER_ERROR_MM/((7*ratio+16)*math.sin(phi/2))
        extrapolated.append(math.degrees(phi))
        return dict(phi_rad=phi, theta_rad=theta, n_c_over_d=ratio, nominal_d_mm=d,
                    nominal_LT_mm=(2*ratio+1)*d, extrapolated_empirical_ratio=True)
    module._corner = corner
    item = dict(id='copp_v5_complete_source', closed=bool(np.linalg.norm(reference[0]-reference[-1]) < 1e-9),
        reference_file='caller_supplied_complete_reference', reference_sha256=module.SOURCE_REGISTRATION_SHA)
    module._source_geometry = lambda path_id: (item, reference, raw, chord)
    curve = module.build_geometry(item['id'], vmax=limits[0], amax=.999*limits[1], jmax=.999*limits[2])
    curve.metadata['source_representation'] = 'validated original analytic primitives with signed arcs' if source_geometry else 'caller polyline'
    return curve, module, None, None, source, extrapolated
