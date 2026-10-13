"""Finite candidate projection onto an intersection of closed planar disks.

This module performs vector geometry only. It is not wired into any controller.
"""
from __future__ import annotations
import math
import numpy as np


def project_disks(point, centers, radii, *, tolerance=1e-10):
    """Return the nearest enumerated feasible point, or an explicit empty flag.

    A nontrivial Euclidean projection has either one active circular boundary
    (the radial projection), or at least two (a pairwise circle intersection).
    Centers additionally cover zero-radius and nested-disk feasibility cases.
    Every candidate is independently checked against every original disk.
    """
    point = np.asarray(point, dtype=float)
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if (point.shape != (2,) or centers.ndim != 2 or centers.shape[1] != 2
            or radii.shape != (len(centers),) or len(centers) == 0
            or not np.isfinite(np.r_[point, centers.ravel(), radii]).all()
            or (radii < 0.).any() or not math.isfinite(tolerance) or tolerance <= 0.):
        raise ValueError('Finite 2-D disks, nonnegative radii and positive tolerance required')
    candidates = [(point.copy(), 'unchanged', [])]
    for i, (center, radius) in enumerate(zip(centers, radii)):
        candidates.append((center.copy(), 'center', [i]))
        vector = point - center
        distance = float(np.linalg.norm(vector))
        if distance > 0.:
            candidates.append((center + vector * (radius / distance), 'radial', [i]))
    pair_geometry = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            delta = centers[j] - centers[i]
            distance = float(np.linalg.norm(delta))
            left, right = float(radii[i]), float(radii[j])
            detail = dict(pair=[i, j], center_distance=distance,
                          external_gap=distance-left-right, containment_gap=abs(left-right)-distance)
            if distance == 0.:
                detail['case'] = 'concentric'
            elif distance > left + right + tolerance:
                detail['case'] = 'disjoint'
            elif distance < abs(left - right) - tolerance:
                detail['case'] = 'strict_containment'
            else:
                along = (left * left - right * right + distance * distance) / (2. * distance)
                height_squared = (left - along) * (left + along)
                h_tolerance = 4. * tolerance * max(1., left, right)
                detail.update(height_squared=height_squared, h_tolerance=h_tolerance,
                              height_was_clamped=bool(-h_tolerance <= height_squared < 0.))
                if height_squared < -h_tolerance:
                    detail['case'] = 'no_real_boundary_intersection'
                else:
                    direction = delta / distance
                    base = centers[i] + along * direction
                    perpendicular = np.array([-direction[1], direction[0]])
                    height = math.sqrt(max(0., height_squared))
                    candidates.append((base + height * perpendicular, 'pair_intersection', [i, j]))
                    if height > 0.:
                        candidates.append((base - height * perpendicular, 'pair_intersection', [i, j]))
                    detail['case'] = 'boundary_candidates'
                    detail['height_squared'] = height_squared
            pair_geometry.append(detail)
    accepted = []
    for candidate, kind, indices in candidates:
        residuals = np.linalg.norm(candidate[None, :] - centers, axis=1) - radii
        if np.isfinite(candidate).all() and float(residuals.max()) <= tolerance:
            accepted.append(dict(point=candidate.tolist(), kind=kind, indices=indices,
                distance_to_request=float(np.linalg.norm(candidate-point)),
                disk_residuals=residuals.tolist(), max_residual=float(residuals.max())))
    accepted.sort(key=lambda item: (item['distance_to_request'], item['kind'], item['indices']))
    return dict(feasible_within_tolerance=bool(accepted), tolerance=tolerance,
                selected=accepted[0] if accepted else None,
                candidate_count=len(candidates), feasible_candidate_count=len(accepted),
                pair_geometry=pair_geometry,
                empty_scope='Finite-candidate test in floating point; numerical near-tangency must be inspected, never repaired by widening the physical limits.')
