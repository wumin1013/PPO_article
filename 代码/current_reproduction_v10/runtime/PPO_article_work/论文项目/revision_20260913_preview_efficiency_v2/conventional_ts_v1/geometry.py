"""Draft Sencer five-control-interval geometry and explicit line-arc extension.

No construction, source read, optimization, schedule or evaluation on import.
build_geometry() uses frozen defining vertices and the original CSV audit
reference. Only geometry is considered; there is no time/PPO/J objective.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
I = HERE.parent
ROOT = I.parents[1]
SOURCE = I / 'geometry_design_v1'
SOURCE_REGISTRATION_SHA = '7879f69348ca580e4a209fad4a0025f407b44869b0c88acbb30d043769f50847'
PATH_IDS = ('mixed_lengths_large', 'dense_corners_large', 'closed_line_arc_large')
CORNER_ERROR_MM = 1.245
MATCH_TOLERANCE_MM = 2e-8
LENGTH_TOLERANCE_MM = 1e-9
INVERSION_TOLERANCE_MM = 1e-11
PEAK_RELATIVE_TOLERANCE = 2e-5
PEAK_ABSOLUTE_TOLERANCE = 1e-12
MAX_PEAK_LEAVES = 8192
MAX_ARC_BLEND_HALVINGS = 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def bernstein(points, u):
    u = np.asarray(u, float).reshape(-1)
    degree = len(points)-1
    weights = np.column_stack([math.comb(degree, k)*(1-u)**(degree-k)*u**k for k in range(degree+1)])
    return weights @ points


def split_controls(points):
    levels = [np.asarray(points, float)]
    while len(levels[-1]) > 1:
        levels.append(.5*(levels[-1][:-1]+levels[-1][1:]))
    return (np.asarray([level[0] for level in levels]),
            np.asarray([level[-1] for level in levels[::-1]]))


def _out(lo, hi):
    return math.nextafter(float(lo), -math.inf), math.nextafter(float(hi), math.inf)


def _add(a, b): return _out(a[0]+b[0], a[1]+b[1])
def _neg(a): return -a[1], -a[0]
def _sub(a, b): return _add(a, _neg(b))


def _mul(a, b):
    values = [a[i]*b[j] for i in (0, 1) for j in (0, 1)]
    return _out(min(values), max(values))


def _square(a):
    return _out(0. if a[0] <= 0 <= a[1] else min(a[0]*a[0], a[1]*a[1]), max(a[0]*a[0], a[1]*a[1]))


def _divide(a, positive):
    if positive[0] <= 0:
        raise ValueError('Regularity lower bound is not positive')
    return _mul(a, _out(1./positive[1], 1./positive[0]))


def _absolute_upper(a): return max(abs(a[0]), abs(a[1]))


def _peak_enclosure(derivatives, direction):
    """Bernstein convex-hull enclosure of kappa and sqrt(kappa^4+kappa_s^2).

Derivative control vectors retain derivatives in the ORIGINAL parameter.
Subdividing them never rescales a derivative by a tiny parameter interval.
Floating operations use outward endpoints; this remains a numerical enclosure,
not a claimed formal interval proof of all control-point construction roundoff.
"""
    vectors = [[_out(np.min(points[:, axis]), np.max(points[:, axis])) for axis in (0, 1)]
               for points in derivatives]
    v, a, j = vectors
    h = _add(_square(v[0]), _square(v[1]))
    projection = float(np.min(derivatives[0] @ direction))
    lower = max(h[0], math.nextafter(projection*projection, -math.inf) if projection > 0 else 0.)
    h = (lower, h[1])
    if h[0] <= 0:
        return math.inf, math.inf
    d = _sub(_mul(v[0], a[1]), _mul(v[1], a[0]))
    e = _sub(_mul(v[0], j[1]), _mul(v[1], j[0]))
    g = _add(_mul(v[0], a[0]), _mul(v[1], a[1]))
    h2, h3 = _mul(h, h), _mul(_mul(h, h), h)
    ks = _sub(_divide(e, h2), _divide(_mul((3., 3.), _mul(d, g)), h3))
    kappa = math.nextafter(_absolute_upper(d)/h[0]**1.5, math.inf)
    normal_jerk = math.nextafter(math.hypot(kappa*kappa, _absolute_upper(ks)), math.inf)
    return kappa, normal_jerk


class Line:
    kind = 'line'

    def __init__(self, start, end, source_start, source_end):
        self.start, self.end = np.asarray(start, float), np.asarray(end, float)
        self.length = float(np.linalg.norm(self.end-self.start))
        self.source_start, self.source_end = float(source_start), float(source_end)
        self.tangent = (self.end-self.start)/self.length if self.length else np.zeros(2)
        self.metadata = dict(kind=self.kind, start=self.start.tolist(), end=self.end.tolist(),
                             source_interval_mm=[self.source_start, self.source_end])

    def position(self, sigma):
        sigma = np.asarray(sigma, float).reshape(-1)
        return self.start + sigma[:, None]*self.tangent

    def source_s(self, sigma):
        fraction = np.asarray(sigma, float)/self.length if self.length else np.zeros_like(sigma, dtype=float)
        return self.source_start + fraction*(self.source_end-self.source_start)


class Arc:
    kind = 'arc'

    def __init__(self, center, radius, angle0, angle1, source_start, source_end):
        self.center, self.radius = np.asarray(center, float), float(radius)
        self.angle0, self.angle1 = float(angle0), float(angle1)
        if radius <= 0 or not angle1 > angle0:
            raise ValueError('This registered arc extension supports its original CCW arc only')
        self.length = self.radius*(self.angle1-self.angle0)
        self.source_start, self.source_end = float(source_start), float(source_end)
        self.metadata = dict(kind=self.kind, center_mm=self.center.tolist(), radius_mm=self.radius,
            angles_rad=[self.angle0, self.angle1], source_interval_mm=[source_start, source_end])

    def angle(self, sigma): return self.angle0 + np.asarray(sigma, float)/self.radius

    def position(self, sigma):
        angle = self.angle(np.asarray(sigma, float).reshape(-1))
        return self.center + self.radius*np.column_stack((np.cos(angle), np.sin(angle)))

    def tangent_at(self, sigma):
        angle = self.angle(np.asarray(sigma, float).reshape(-1))
        return np.column_stack((-np.sin(angle), np.cos(angle)))

    def source_s(self, sigma):
        return self.source_start + np.asarray(sigma, float)/self.length*(self.source_end-self.source_start)

    def peak_bounds(self):
        return dict(kappa_upper=1./self.radius, constant_speed_jerk_factor_upper=1./self.radius**2,
                    kappa_sample_lower=1./self.radius, constant_speed_jerk_factor_sample_lower=1./self.radius**2,
                    leaves=0, method='analytic circle interior')


class Bezier:
    kind = 'bezier5'

    def __init__(self, points, source_knots, *, metadata):
        points = np.asarray(points, float)
        if points.shape != (6, 2) or not np.isfinite(points).all():
            raise ValueError('Six finite planar control points required')
        self.origin = points[0].copy()
        self.points = points-self.origin
        self.derivatives = []
        derivative = self.points
        for degree in (5, 4, 3):
            derivative = degree*np.diff(derivative, axis=0)
            self.derivatives.append(derivative)
        chord = self.points[-1]-self.points[0]
        self.direction = chord/np.linalg.norm(chord)
        self.regularity_lower = float(np.min(self.derivatives[0] @ self.direction))
        if not self.regularity_lower > 0:
            raise ValueError('Bezier tangent lacks a positive source-direction certificate')
        self.source_knots = np.asarray(source_knots, float)
        if self.source_knots.shape != (3,) or np.any(np.diff(self.source_knots) < 0):
            raise ValueError('Ordered source start/mid/end required')
        self.source_start, self.source_end = float(self.source_knots[0]), float(self.source_knots[-1])
        self.metadata = dict(metadata, kind=self.kind, control_points_mm=points.tolist(),
            source_knots_mm=self.source_knots.tolist(), regularity_projection_lower=self.regularity_lower)
        self._prepare_length()
        self._peaks = None

    def position_u(self, u): return self.origin + bernstein(self.points, u)

    def geometry_u(self, u):
        v, a, j = [bernstein(points, u) for points in self.derivatives]
        h, d, e, g = np.sum(v*v, axis=1), cross(v, a), cross(v, j), np.sum(v*a, axis=1)
        if np.any(h <= 0):
            raise ValueError('Nonregular curve query')
        kappa = d/h**1.5
        kappa_s = e/h**2 - 3*d*g/h**3
        return v/np.sqrt(h)[:, None], kappa, kappa_s

    def _integral(self, left, right, order):
        left, right = np.broadcast_arrays(np.asarray(left, float), np.asarray(right, float))
        nodes, weights = leggauss(order)
        u = .5*(left.reshape(-1, 1)+right.reshape(-1, 1)) + .5*(right-left).reshape(-1, 1)*nodes
        speed = np.linalg.norm(bernstein(self.derivatives[0], u.ravel()), axis=1).reshape(u.shape)
        return .5*(right-left).ravel()*(speed @ weights)

    def _prepare_length(self):
        for cells in (16, 32, 64, 128, 256, 512):
            u = np.linspace(0., 1., cells+1)
            low, high = [self._integral(u[:-1], u[1:], order) for order in (16, 32)]
            disagreement = float(np.abs(high-low).sum())
            if disagreement <= LENGTH_TOLERANCE_MM:
                if np.any(high <= 0):
                    raise ValueError('Positive Bezier arc-length cells required')
                self.u_nodes, self.s_nodes = u, np.r_[0., np.cumsum(high)]
                self.length = float(self.s_nodes[-1])
                self.metadata.update(length_mm=self.length, quadrature_cells=cells,
                                     quadrature_order_pair=[16, 32], quadrature_disagreement_mm=disagreement)
                return
        raise ValueError('Bounded Gauss arc-length agreement check failed')

    def parameter(self, sigma):
        sigma = np.asarray(sigma, float).reshape(-1)
        if np.any(sigma < -LENGTH_TOLERANCE_MM) or np.any(sigma > self.length+LENGTH_TOLERANCE_MM):
            raise ValueError('Arc-length query outside Bezier')
        sigma = np.clip(sigma, 0., self.length)
        cell = np.clip(np.searchsorted(self.s_nodes, sigma, side='right')-1, 0, len(self.u_nodes)-2)
        base = self.u_nodes[cell]
        lo, hi = base.copy(), self.u_nodes[cell+1].copy()
        wanted = sigma-self.s_nodes[cell]
        u = lo+(hi-lo)*wanted/(self.s_nodes[cell+1]-self.s_nodes[cell])
        for _ in range(48):
            residual = self._integral(base, u, 32)-wanted
            if np.all(np.abs(residual) <= INVERSION_TOLERANCE_MM):
                u[sigma == 0.] = 0.; u[sigma == self.length] = 1.
                return u
            lo = np.where(residual < 0, u, lo)
            hi = np.where(residual > 0, u, hi)
            speed = np.linalg.norm(bernstein(self.derivatives[0], u), axis=1)
            trial = u-residual/speed
            u = np.where((trial > lo) & (trial < hi), trial, .5*(lo+hi))
        raise ValueError('Bounded true arc-length inversion did not converge')

    def position(self, sigma): return self.position_u(self.parameter(sigma))
    def source_s(self, sigma): return np.interp(self.parameter(sigma), [0., .5, 1.], self.source_knots)

    def peak_bounds(self):
        if self._peaks is not None:
            return self._peaks
        def leaf(derivatives, start, end):
            upper = _peak_enclosure(derivatives, self.direction)
            _, kappa, ks = self.geometry_u([start, .5*(start+end), end])
            lower = (float(np.max(np.abs(kappa))), float(np.max(np.hypot(kappa*kappa, ks))))
            return dict(derivatives=derivatives, start=start, end=end, upper=upper, lower=lower)
        leaves = [leaf(self.derivatives, 0., 1.)]
        lower = np.asarray(leaves[0]['lower'])
        for _ in range(MAX_PEAK_LEAVES):
            uppers = np.asarray([item['upper'] for item in leaves])
            upper = uppers.max(axis=0)
            if np.all(upper <= lower*(1+PEAK_RELATIVE_TOLERANCE)+PEAK_ABSOLUTE_TOLERANCE):
                self._peaks = dict(kappa_upper=float(upper[0]), constant_speed_jerk_factor_upper=float(upper[1]),
                    kappa_sample_lower=float(lower[0]), constant_speed_jerk_factor_sample_lower=float(lower[1]),
                    leaves=len(leaves), relative_tolerance=PEAK_RELATIVE_TOLERANCE,
                    method='adaptive derivative Bernstein numerical enclosure; complete physical audit still required')
                self.metadata['geometric_peaks'] = self._peaks
                return self._peaks
            if len(leaves) >= MAX_PEAK_LEAVES:
                break
            index = int(np.argmax(np.max(uppers/(lower+PEAK_ABSOLUTE_TOLERANCE), axis=1)))
            old = leaves.pop(index)
            pairs = [split_controls(points) for points in old['derivatives']]
            middle = .5*(old['start']+old['end'])
            children = [leaf([pair[0] for pair in pairs], old['start'], middle),
                        leaf([pair[1] for pair in pairs], middle, old['end'])]
            leaves.extend(children)
            lower = np.maximum(lower, np.max([item['lower'] for item in children], axis=0))
        raise ValueError('Bounded curve-peak enclosure did not reach its declared tolerance')


class Compound:
    kind = 'compound_curve'

    def __init__(self, pieces, *, metadata):
        self.pieces = list(pieces)
        self.ends = np.cumsum([piece.length for piece in self.pieces])
        self.length = float(self.ends[-1])
        self.source_start, self.source_end = pieces[0].source_start, pieces[-1].source_end
        self.metadata = dict(metadata, kind=self.kind, pieces=[piece.metadata for piece in pieces], length_mm=self.length)

    def _query(self, sigma, method):
        sigma = np.asarray(sigma, float).reshape(-1)
        output = np.empty((len(sigma), 2)) if method == 'position' else np.empty(len(sigma))
        indices = np.clip(np.searchsorted(self.ends, sigma, side='right'), 0, len(self.pieces)-1)
        starts = np.r_[0., self.ends[:-1]]
        for index, piece in enumerate(self.pieces):
            selected = indices == index
            if np.any(selected):
                output[selected] = getattr(piece, method)(sigma[selected]-starts[index])
        return output

    def position(self, sigma): return self._query(sigma, 'position')
    def source_s(self, sigma): return self._query(sigma, 'source_s')

    def peak_bounds(self):
        peaks = [piece.peak_bounds() for piece in self.pieces]
        return dict(kappa_upper=max(row['kappa_upper'] for row in peaks),
            constant_speed_jerk_factor_upper=max(row['constant_speed_jerk_factor_upper'] for row in peaks),
            method='maximum of both C2 connectors and analytic arc; common cruise speed', pieces=peaks)


def _source_geometry(path_id):
    if path_id not in PATH_IDS or sha(SOURCE / 'registration.json') != SOURCE_REGISTRATION_SHA:
        raise ValueError('Require one frozen original geometry identity')
    registration = json.loads((SOURCE / 'registration.json').read_text(encoding='utf-8'))
    freeze = json.loads((SOURCE / 'freeze_receipt.json').read_text(encoding='utf-8'))
    if freeze['status'] != 'frozen_for_shared_future_comparison' or freeze['registration_sha256'] != SOURCE_REGISTRATION_SHA:
        raise ValueError('Geometry freeze receipt differs')
    item = next(row for row in registration['paths'] if row['id'] == path_id)
    reference_path = (ROOT / item['reference_file']).resolve()
    if not reference_path.is_relative_to(ROOT.resolve()) or sha(reference_path) != item['reference_sha256']:
        raise ValueError('Original CSV identity differs')
    reference = np.loadtxt(reference_path, delimiter=',', skiprows=1)
    source_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(reference, axis=0), axis=1))]
    cursor = 0
    raw = []
    def lines(vertices):
        nonlocal cursor
        vertices = np.asarray(vertices, float)
        anchors = []
        for point in vertices:
            matches = np.flatnonzero(np.linalg.norm(reference[cursor:]-point, axis=1) <= MATCH_TOLERANCE_MM)
            if not len(matches):
                raise ValueError('Defining source vertex absent from the original ordered CSV')
            cursor += int(matches[0]); anchors.append(cursor)
        for a, b, ia, ib in zip(vertices[:-1], vertices[1:], anchors[:-1], anchors[1:]):
            segment = Line(a, b, source_s[ia], source_s[ib])
            if segment.length <= 0:
                raise ValueError('Unexpected duplicate defining vertices')
            offsets = reference[ia:ib+1]-a
            along = offsets @ segment.tangent
            if (np.max(np.abs(cross(offsets, segment.tangent))) > MATCH_TOLERANCE_MM
                    or np.any(np.diff(along) < -MATCH_TOLERANCE_MM)
                    or np.min(along) < -MATCH_TOLERANCE_MM or np.max(along) > segment.length+MATCH_TOLERANCE_MM):
                raise ValueError('Defining line is not the same ordered CSV subpath')
            raw.append(segment)
    parameters = item['parameters']
    chord_error = 0.
    if not item['closed']:
        lines(parameters['defining_vertices_mm'])
    else:
        lines(parameters['bottom_defining_vertices_mm'])
        circle = parameters['circle_arc']
        if circle != dict(center_mm=[5200, 1800], radius_mm=1800., start_angle_deg=-90,
                          end_angle_deg=90, direction='CCW', sweep_deg=180, polyline_segments=1054):
            raise ValueError('Only the declared original circular arc is supported')
        first, last = cursor, cursor+circle['polyline_segments']
        angles = np.linspace(-math.pi/2, math.pi/2, circle['polyline_segments']+1)
        expected = np.asarray(circle['center_mm'])+circle['radius_mm']*np.column_stack((np.cos(angles), np.sin(angles)))
        if last >= len(reference) or np.max(np.linalg.norm(reference[first:last+1]-expected, axis=1)) > MATCH_TOLERANCE_MM:
            raise ValueError('Declared analytic arc does not match the original 1054 CSV chords')
        raw.append(Arc(circle['center_mm'], circle['radius_mm'], -math.pi/2, math.pi/2, source_s[first], source_s[last]))
        chord_error = 2*circle['radius_mm']*math.sin(math.pi/(4*circle['polyline_segments']))**2
        if chord_error > parameters['requested_arc_chord_error_bound_mm']:
            raise ValueError('Original arc chord error exceeds its registered bound')
        cursor = last
        lines(parameters['top_defining_vertices_mm'])
        lines(parameters['left_defining_vertices_mm'])
    if cursor != len(reference)-1:
        raise ValueError('Source reconstruction did not consume exactly one complete ordered CSV')
    return item, reference, raw, chord_error


def _corner(a, b):
    phi = math.atan2(abs(float(cross(a.tangent, b.tangent))), float(a.tangent @ b.tangent))
    if phi <= 1e-12:
        return None
    degrees = math.degrees(phi)
    if not 30.-1e-7 <= degrees <= 170.+1e-7:
        raise ValueError('Unexpected true non-arc turn outside the audited Sencer fit domain: ' + str(degrees))
    theta = math.pi-phi
    ratio = theta**.9927/2.0769
    d = 16*CORNER_ERROR_MM/((7*ratio+16)*math.sin(phi/2))
    return dict(phi_rad=phi, theta_rad=theta, n_c_over_d=ratio, nominal_d_mm=d,
                nominal_LT_mm=(2*ratio+1)*d)


def _quintic_hermite(start, end, tangent0, tangent1, curvature0, curvature1, scale, source_knots, role):
    normal0, normal1 = np.array([-tangent0[1], tangent0[0]]), np.array([-tangent1[1], tangent1[0]])
    d0, d1 = scale*tangent0, scale*tangent1
    a0, a1 = scale*scale*curvature0*normal0, scale*scale*curvature1*normal1
    points = [start, start+d0/5, start+2*d0/5+a0/20,
              end-2*d1/5+a1/20, end-d1/5, end]
    return Bezier(points, source_knots, metadata=dict(role=role, parameter_scale_mm=scale,
        endpoint_curvatures_mm_inv=[curvature0, curvature1], construction='unique quintic position/tangent/curvature Hermite extension'))


def _blend_error_bound(bezier, source_position, h, budget):
    derivative_bound = float(np.max(np.linalg.norm(bezier.derivatives[0], axis=1)))+2*h
    for cells in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        u = np.linspace(0., 1., cells+1)
        sample = float(np.max(np.linalg.norm(bezier.position_u(u)-source_position(u), axis=1)))
        upper = math.nextafter(sample+derivative_bound/(2*cells), math.inf)
        if sample > budget:
            return dict(accepted=False, sampled_error_mm=sample, error_upper_mm=upper, cells=cells)
        if upper <= budget:
            return dict(accepted=True, sampled_error_mm=sample, error_upper_mm=upper, cells=cells)
    return dict(accepted=False, sampled_error_mm=sample, error_upper_mm=upper, cells=cells,
                reason='finite pointwise source-correspondence bound not resolved')


def _arc_extension(before, arc, after, free_before, free_after, chord_error):
    if (np.linalg.norm(before.tangent-arc.tangent_at([0.])[0]) > 1e-10
            or np.linalg.norm(after.tangent-arc.tangent_at([arc.length])[0]) > 1e-10):
        raise ValueError('Original line-arc interfaces are not the expected tangent joins')
    maximum = min(free_before, free_after, arc.length/2)
    if maximum <= 0:
        raise ValueError('No nonoverlapping source interval for the line-arc C2 extension')
    budget = CORNER_ERROR_MM-chord_error
    attempts = []
    for reduction in range(MAX_ARC_BLEND_HALVINGS+1):
        h = maximum/(2**reduction)
        entry = _quintic_hermite(before.position([before.length-h])[0], arc.position([h])[0],
            before.tangent, arc.tangent_at([h])[0], 0., 1./arc.radius, 2*h,
            [float(before.source_s(before.length-h)), arc.source_start, float(arc.source_s(h))], 'line_to_arc_C2')
        exit_curve = _quintic_hermite(arc.position([arc.length-h])[0], after.position([h])[0],
            arc.tangent_at([arc.length-h])[0], after.tangent, 1./arc.radius, 0., 2*h,
            [float(arc.source_s(arc.length-h)), arc.source_end, float(after.source_s(h))], 'arc_to_line_C2')
        def entry_source(u):
            return np.where((u <= .5)[:, None], before.position(before.length-h+2*h*u), arc.position(2*h*u-h))
        def exit_source(u):
            return np.where((u <= .5)[:, None], arc.position(arc.length-h+2*h*u), after.position(2*h*u-h))
        checks = [_blend_error_bound(entry, entry_source, h, budget), _blend_error_bound(exit_curve, exit_source, h, budget)]
        attempts.append(dict(h_mm=h, checks=checks))
        if all(check['accepted'] for check in checks):
            pieces = [entry]
            if arc.length > 2*h:
                pieces.append(Arc(arc.center, arc.radius, arc.angle0+h/arc.radius, arc.angle1-h/arc.radius,
                                  float(arc.source_s(h)), float(arc.source_s(arc.length-h))))
            pieces.append(exit_curve)
            return h, Compound(pieces, metadata=dict(role='declared_arc_with_two_C2_interfaces',
                original_arc=arc.metadata, source_chord_error_bound_mm=chord_error,
                connector_error_budget_mm=budget, selection='first passing geometric source-correspondence bound from maximum nonoverlap h, halved only on geometric rejection',
                geometric_attempts=attempts, selected_h_mm=h))
    raise ValueError('Bounded line-arc connector geometry checks failed')


class GeometryPath(Compound):
    def __init__(self, pieces, item, reference, *, settings):
        # Consecutive collinear remnants form one exact straight for scheduling.
        merged = []
        for piece in pieces:
            if piece.length <= 0:
                continue
            if merged and isinstance(piece, Line) and isinstance(merged[-1], Line):
                previous = merged[-1]
                if np.linalg.norm(previous.end-piece.start) > MATCH_TOLERANCE_MM or np.linalg.norm(previous.tangent-piece.tangent) > 1e-10:
                    raise ValueError('Noncollinear remnants cannot be merged into a scheduling straight')
                merged[-1] = Line(previous.start, piece.end, previous.source_start, piece.source_end)
            else:
                merged.append(piece)
        if not merged or not isinstance(merged[0], Line) or not isinstance(merged[-1], Line):
            raise ValueError('This cruise scheduler requires genuine entry and exit straight remnants')
        for left, right in zip(merged[:-1], merged[1:]):
            if (np.linalg.norm(left.position([left.length])[0]-right.position([0.])[0]) > MATCH_TOLERANCE_MM
                    or abs(left.source_end-right.source_start) > MATCH_TOLERANCE_MM):
                raise ValueError('Geometric/source interval discontinuity')
        super().__init__(merged, metadata=dict(path_id=item['id'], closed=item['closed'],
            source_reference=item['reference_file'], source_reference_sha256=item['reference_sha256'],
            source_registration_sha256=SOURCE_REGISTRATION_SHA, settings=settings,
            source_order_semantics='one monotone start-to-end source traversal; closed endpoint repeated only at the finite terminal time'))
        self.reference = reference
        straight, curves, caps = [], [], []
        pending_straight = 0.
        vmax, amax, jmax = settings['vmax'], settings['amax'], settings['jmax']
        for piece in merged:
            if isinstance(piece, Line):
                pending_straight += piece.length
            else:
                straight.append(pending_straight); pending_straight = 0.
                peaks = piece.peak_bounds()
                kappa, jerk_factor = peaks['kappa_upper'], peaks['constant_speed_jerk_factor_upper']
                cap = min(vmax, math.sqrt(amax/kappa) if kappa > 0 else vmax,
                          (jmax/jerk_factor)**(1/3) if jerk_factor > 0 else vmax)
                if not 0 < cap <= vmax:
                    raise ValueError('Nonpositive or nonfinite geometric cruise cap')
                curves.append(piece.length); caps.append(cap)
                piece.metadata.update(geometric_peaks=peaks, cruise_speed_cap_mm_s=cap)
        straight.append(pending_straight)
        self.straight_lengths_mm = np.asarray(straight)
        self.curve_lengths_mm = np.asarray(curves)
        self.curve_speed_caps_mm_s = np.asarray(caps)
        if len(straight) != len(curves)+1 or abs(sum(straight)+sum(curves)-self.length) > 1e-7:
            raise ValueError('n-curve/n+1-straight scheduling interface mismatch')
        if (np.linalg.norm(self.position([0.])[0]-reference[0]) > MATCH_TOLERANCE_MM
                or np.linalg.norm(self.position([self.length])[0]-reference[-1]) > MATCH_TOLERANCE_MM):
            raise ValueError('Original finite traversal endpoints changed')
        self.metadata.update(straight_lengths_mm=self.straight_lengths_mm.tolist(),
            curve_lengths_mm=self.curve_lengths_mm.tolist(), curve_speed_caps_mm_s=self.curve_speed_caps_mm_s.tolist(),
            performance_result=None, complete_physical_feasibility_asserted=False)

    def position(self, sigma):
        sigma = np.asarray(sigma, float).reshape(-1)
        if not np.isfinite(sigma).all() or np.any(sigma < -LENGTH_TOLERANCE_MM) or np.any(sigma > self.length+LENGTH_TOLERANCE_MM):
            raise ValueError('Query outside the one finite geometric traversal')
        return super().position(np.clip(sigma, 0., self.length))


def build_geometry(path_id, *, vmax=100., amax=1998., jmax=62437.5):
    """Construct once only after separate review/registration; no write or run."""
    if not all(math.isfinite(value) and value > 0 for value in (vmax, amax, jmax)):
        raise ValueError('Positive finite public vector limits required')
    item, reference, raw, chord_error = _source_geometry(path_id)
    corners = {index: _corner(a, b) for index, (a, b) in enumerate(zip(raw[:-1], raw[1:]))
               if isinstance(a, Line) and isinstance(b, Line)}
    corners = {index: value for index, value in corners.items() if value is not None}
    factors = np.ones(len(raw))
    for index, source in enumerate(raw):
        occupied = sum(corners[j]['nominal_LT_mm'] for j in (index-1, index) if j in corners)
        if isinstance(source, Line) and occupied > source.length:
            factors[index] = source.length/occupied
    trims_start, trims_end = np.zeros(len(raw)), np.zeros(len(raw))
    curves = {}
    for index, parameters in corners.items():
        a, b = raw[index], raw[index+1]
        alpha = min(factors[index], factors[index+1])
        ratio = parameters['n_c_over_d']
        d = parameters['nominal_d_mm']*alpha; c = ratio*d; length = 2*c+d
        vertex = a.end
        points = [vertex-length*a.tangent, vertex-(c+d)*a.tangent, vertex-d*a.tangent,
                  vertex+d*b.tangent, vertex+(c+d)*b.tangent, vertex+length*b.tangent]
        curves[index] = Bezier(points,
            [float(a.source_s(a.length-length)), a.source_end, float(b.source_s(length))],
            metadata=dict(parameters, role='Sencer_fitted_six_point_corner', overlap_scale=alpha,
                LT_mm=length, c_mm=c, d_mm=d, nominal_corner_error_mm=CORNER_ERROR_MM,
                achieved_midpoint_corner_error_mm=CORNER_ERROR_MM*alpha))
        trims_end[index] = length; trims_start[index+1] = length
    arc_curves = {}
    for index, source in enumerate(raw):
        if isinstance(source, Arc):
            if index == 0 or index == len(raw)-1 or not isinstance(raw[index-1], Line) or not isinstance(raw[index+1], Line):
                raise ValueError('Declared arc lacks its original adjacent straight segments')
            h, compound = _arc_extension(raw[index-1], source, raw[index+1],
                raw[index-1].length-trims_start[index-1], raw[index+1].length-trims_end[index+1], chord_error)
            trims_end[index-1] = h; trims_start[index+1] = h; arc_curves[index] = compound
    pieces = []
    for index, source in enumerate(raw):
        if isinstance(source, Line):
            start, end = trims_start[index], source.length-trims_end[index]
            if end < start-1e-10:
                raise ValueError('Local geometry trims overlap')
            if end > start:
                pieces.append(Line(source.position([start])[0], source.position([end])[0],
                                   float(source.source_s(start)), float(source.source_s(end))))
        else:
            pieces.append(arc_curves[index])
        if index in curves:
            pieces.append(curves[index])
    return GeometryPath(pieces, item, reference, settings=dict(vmax=vmax, amax=amax, jmax=jmax,
        planning_limits=dict(vmax_mm_s=vmax, amax_mm_s2=amax, jmax_mm_s3=jmax),
        public_audit_limits=dict(vmax_mm_s=100., amax_mm_s2=2000., jmax_mm_s3=62500.),
        planning_reserve='Same acceleration/jerk interior values as X KCM; no straight speed discount',
        inversion_tolerance_mm=INVERSION_TOLERANCE_MM,
        corner_error_mm=CORNER_ERROR_MM, fit='theta^0.9927/2.0769, theta radians, n=c/d, true phi in30..170degrees',
        extra_n_optimizations=0, circle_sampling_seams_removed_by_explicit_source_arc_extension=True,
        time_optimization=False, hidden_global_speed_scale=False))
