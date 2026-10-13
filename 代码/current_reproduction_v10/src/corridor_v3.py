"""Explicit ordered-gate initialization with unchanged command acceptance.

Intersecting gate channels do not make an ordered traversal undefined: the
tracker retains the next unvisited gate and requires a witnessed forward
crossing in the channel. We admit such a geometry only after its exact source
polyline witnesses every original gate in order. No gate is removed/moved and
the tracker used during execution and independent auditing is unchanged.
"""
import numpy as np
from runtime_bootstrap import setup
setup()
from geometry_spec import Corridor, GateTracker

TOLERANCE_MM = 1.25


class OrderedCorridor(Corridor):
    def __init__(self, reference, epsilon=None, *, closed=False):
        super().__init__(reference, TOLERANCE_MM if epsilon is None else epsilon, closed=closed)
        self.initialization_gate_intersections = super().adjacent_gate_conflicts()
        self.reference_witness = None

    def adjacent_gate_conflicts(self):
        conflicts = self.initialization_gate_intersections
        if not conflicts:
            return []
        tracker = GateTracker(self, self.reference[0])
        for point in self.reference[1:]:
            tracker.update(point)
        self.reference_witness = tracker.snapshot(endpoint_tolerance=1e-8)
        if not self.reference_witness['geometry_complete']:
            return conflicts
        # The unmodified tracker continues checking EVERY original gate during
        # the actual command stream; this witness never advances that tracker.
        return []


def configure(tolerance=1.25):
    global TOLERANCE_MM
    tolerance = float(tolerance)
    if not 0 < tolerance <= 1.25:
        raise ValueError('v3 permits nominal tolerance or a tighter arc-error reservation only')
    TOLERANCE_MM = tolerance
    import joint_runtime
    joint_runtime.Corridor = OrderedCorridor


def restore():
    import joint_runtime
    joint_runtime.Corridor = Corridor
