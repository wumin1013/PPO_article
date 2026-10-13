"""Incremental counterpart of revision_metrics.ordered_projection.

The witness construction, matching radius and segment-time ordering are shared
with the frozen batch evaluator. This is a sampled-command coverage check, not
a continuous-motion or axis-constraint guarantee.
"""
from __future__ import annotations

import numpy as np

from .revision_metrics import _reference_witnesses


class OrderedCoverageTracker:
    """Track newly covered reference arc length without rewarding retracing.

    The initial command can already cover nearby reference witnesses. Their arc
    length establishes a baseline and is never counted as motion reward.
    """
    def __init__(self, reference, *, tolerance_mm, initial_position,
                 witness_spacing_mm=None):
        self.reference = np.asarray(reference, dtype=float).copy()
        self.tolerance_mm = float(tolerance_mm)
        if (self.reference.ndim != 2 or self.reference.shape[1] != 2
                or len(self.reference) < 2 or not np.isfinite(self.reference).all()
                or not np.isfinite(self.tolerance_mm) or self.tolerance_mm <= 0.0):
            raise ValueError("Need a finite 2-D reference and positive one-sided tolerance")
        self.spacing_mm = (min(.25, self.tolerance_mm / 2.) if witness_spacing_mm is None
                           else float(witness_spacing_mm))
        if not 0.0 < self.spacing_mm < 2.0 * self.tolerance_mm:
            raise ValueError("Witness spacing must be positive and smaller than twice tolerance")
        self.radius_mm = self.tolerance_mm - self.spacing_mm / 2.
        self.witnesses, self.arcs = _reference_witnesses(self.reference, self.spacing_mm)
        self.reference_length_mm = float(self.arcs[-1])
        self._starts = self.reference[:-1]
        self._vectors = np.diff(self.reference, axis=0)
        self._lengths2 = np.sum(self._vectors * self._vectors, axis=1)
        self.frontier = 0
        self.covered_s_mm = 0.0
        self.previous_position = self._point(initial_position)
        self.start_error_mm = float(np.linalg.norm(self.previous_position - self.reference[0]))
        self.contour_error_max_mm = 0.0
        self.geometry_feasible = True
        self._advance(self.previous_position)
        self.initial_covered_s_mm = self.covered_s_mm

    @staticmethod
    def _point(position):
        point = np.asarray(position, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Command position must be a finite 2-D point")
        return point.copy()

    def _advance(self, point):
        start = self.previous_position
        vector = point - start
        length2 = float(vector @ vector)
        earliest = 0.0
        while self.frontier < len(self.witnesses):
            relative = self.witnesses[self.frontier] - start
            if length2 <= 1e-24:
                if np.linalg.norm(relative) > self.radius_mm + 1e-12:
                    break
            else:
                center = float(relative @ vector / length2)
                perpendicular2 = float(relative @ relative - center * center * length2)
                discriminant = self.radius_mm * self.radius_mm - max(0.0, perpendicular2)
                if discriminant < -1e-12:
                    break
                half_width = np.sqrt(max(0.0, discriminant) / length2)
                lower = max(0.0, center - half_width)
                upper = min(1.0, center + half_width)
                earliest = max(earliest, lower)
                if earliest > upper + 1e-12:
                    break
            self.frontier += 1
        self.covered_s_mm = float(self.arcs[self.frontier - 1]) if self.frontier else 0.0
        fractions = np.clip(np.sum((point - self._starts) * self._vectors, axis=1)
                            / self._lengths2, 0.0, 1.0)
        projections = self._starts + fractions[:, None] * self._vectors
        self.contour_error_mm = float(np.min(np.linalg.norm(point - projections, axis=1)))
        self.contour_error_max_mm = max(self.contour_error_max_mm, self.contour_error_mm)
        self.geometry_feasible = bool(self.geometry_feasible
                                      and self.contour_error_mm <= self.tolerance_mm + 1e-8)
        self.previous_position = point

    def snapshot(self, delta_s_mm=0.0):
        endpoint_error = float(np.linalg.norm(self.previous_position - self.reference[-1]))
        complete = self.frontier == len(self.witnesses)
        terminal_geometry = bool(self.geometry_feasible and self.start_error_mm <= self.tolerance_mm
                                 and endpoint_error <= self.tolerance_mm)
        return dict(delta_s_mm=float(delta_s_mm), covered_s_mm=self.covered_s_mm,
                    initial_covered_s_mm=self.initial_covered_s_mm,
                    ordered_progress=self.covered_s_mm / self.reference_length_mm,
                    covered_witness_count=self.frontier, witness_count=len(self.witnesses),
                    witness_spacing_mm=self.spacing_mm, witness_match_radius_mm=self.radius_mm,
                    ordered_coverage_complete=bool(complete), geometry_feasible=self.geometry_feasible,
                    start_error_mm=self.start_error_mm, endpoint_error_mm=endpoint_error,
                    contour_error_mm=self.contour_error_mm, contour_error_max_mm=self.contour_error_max_mm,
                    completion_valid=bool(complete and terminal_geometry))

    def update(self, position):
        previous_arc = self.covered_s_mm
        self._advance(self._point(position))
        return self.snapshot(self.covered_s_mm - previous_arc)


__all__ = ["OrderedCoverageTracker"]
