"""Pure local-corner geometry for a separately registered observation ablation.

This module does not import an environment, select a preview/control action,
modify a reward, or install adapters. Cache ``NearestTurnGeometry`` per path.
Arc coordinates use the finite reference order; a query exactly on a vertex
looks strictly forward, matching the existing turn scanner.
"""
from __future__ import annotations

import math
from typing import TypedDict

import numpy as np


class TurnState(TypedDict):
    dist_to_turn: float
    turn_angle: float
    turn_sign: int
    turn_s: float | None
    start_vertex: int | None
    end_vertex: int | None
    vertex_count: int


class NearestTurnGeometry:
    """Nearest consecutive significant turns with one consistent sign.

    A sign reversal or a nonsignificant vertex ends a group. Its distance is
    measured to its first remaining vertex, never a later curvature maximum.
    Same-direction vertices of a discretized curve retain their accumulated
    angle. ``turn_angle`` is that signed sum in radians, without wrapping a
    turn above pi into the opposite direction; normalize/clip only in the
    observation wrapper. This class has no corner-phase or control state.
    """

    def __init__(self, reference, *, one_sided_tolerance_mm: float,
                 kappa_threshold: float = 0.0):
        points = np.array(reference, dtype=float, copy=True)
        tolerance = float(one_sided_tolerance_mm)
        configured_threshold = float(kappa_threshold)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all()):
            raise ValueError('reference must be a finite N x 2 array with N >= 2')
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('one_sided_tolerance_mm must be positive and finite')
        if not math.isfinite(configured_threshold) or configured_threshold < 0:
            raise ValueError('kappa_threshold must be nonnegative and finite')
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 0):
            raise ValueError('adjacent reference points must be distinct')
        heads = np.arctan2(vectors[:, 1], vectors[:, 0])
        turns = (np.diff(heads)+math.pi) % (2*math.pi)-math.pi
        self.reference_length_mm = float(lengths.sum())
        # Exactly the existing scanner's half-epsilon-dependent proxy/threshold.
        local_distance = np.maximum(.5*(lengths[:-1]+lengths[1:]), 1e-6)
        self.kappa_threshold = max(configured_threshold, .25/(tolerance+1e-6))
        significant = np.abs(turns)/(local_distance+1e-6) > self.kappa_threshold
        self._arcs = np.cumsum(lengths)[:-1]
        self._turns = turns
        self._signs = np.sign(turns).astype(int)
        self._eligible = np.flatnonzero(significant)
        self._eligible_arcs = self._arcs[self._eligible]
        self._group_ends = np.arange(len(turns), dtype=int)
        for i in range(len(turns)-2, -1, -1):
            if significant[i] and significant[i+1] and self._signs[i] == self._signs[i+1]:
                self._group_ends[i] = self._group_ends[i+1]

    def at(self, s_mm: float) -> TurnState:
        """Query finite reference arc without changing this object or the path."""
        s = float(s_mm)
        if not math.isfinite(s):
            raise ValueError('s_mm must be finite')
        s = float(np.clip(s, 0., self.reference_length_mm))
        eligible_index = int(np.searchsorted(self._eligible_arcs, s, side='right'))
        if eligible_index == len(self._eligible):
            return dict(dist_to_turn=math.inf, turn_angle=0., turn_sign=0,
                        turn_s=None, start_vertex=None, end_vertex=None, vertex_count=0)
        first = int(self._eligible[eligible_index])
        last = int(self._group_ends[first])
        return dict(dist_to_turn=float(self._arcs[first]-s),
                    turn_angle=float(np.sum(self._turns[first:last+1])),
                    turn_sign=int(self._signs[first]), turn_s=float(self._arcs[first]),
                    start_vertex=first+1, end_vertex=last+1, vertex_count=last-first+1)


def nearest_turn_state(reference, s_mm: float, one_sided_tolerance_mm: float, *,
                       kappa_threshold: float = 0.0) -> TurnState:
    """Convenience for an isolated query; cache the class for repeated queries."""
    return NearestTurnGeometry(reference, one_sided_tolerance_mm=one_sided_tolerance_mm,
                               kappa_threshold=kappa_threshold).at(s_mm)


__all__ = ['NearestTurnGeometry', 'TurnState', 'nearest_turn_state']
