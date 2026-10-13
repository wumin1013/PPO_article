"""Anticipate a finite brake only after validating its full ordered trace.

Development version; keeps the original bounded cubic brake and acceptance.
No weight, steering, speed cap, gate, or endpoint-tolerance change is made.
"""
import copy
import numpy as np

VERSION = 'v4c_verified_terminal_preview'


def consider_verified_terminal(self):
    import joint_runtime as joint
    progress, _ = self.progress()
    if self.corridor.arcs[-1] - progress > 12.:
        return
    self.v4c_terminal_attempts = getattr(self, 'v4c_terminal_attempts', 0) + 1
    brake = joint.terminal.plan_planar_terminal_brake(
        self.position, self.velocity, self.acceleration,
        dt=joint.DT, limits=joint.terminal.LIMITS)
    columns = brake['columns']
    points = np.column_stack([columns['x_mm'], columns['y_mm']])
    if np.linalg.norm(points[-1] - self.corridor.reference[-1]) > 1.0:
        return
    tracker = copy.copy(self.tracker)
    for key, value in vars(self.tracker).items():
        if key != 'corridor':
            setattr(tracker, key, copy.deepcopy(value))
    # These are the same command endpoints consumed by JointRuntime.step;
    # GateTracker checks every entire command interval, not only each endpoint.
    for point in points[1:]:
        tracker.update(point)
        if not tracker.inside:
            return
    if not tracker.snapshot(endpoint_tolerance=1.0)['geometry_complete']:
        return
    self.v4c_terminal_start_frontier = self.tracker.frontier
    self.v4c_terminal_remaining_gates = len(self.corridor.gates) - self.tracker.frontier
    self.brake = columns
    self.brake_index = 0


def install(*, dt=.001, tolerance=1.25):
    from runtime_no_target_override_v4b import install as install_v4b
    install_v4b(dt=dt, tolerance=tolerance)
    import joint_runtime as joint
    joint.JointRuntime._consider_terminal = consider_verified_terminal
    return dict(version=VERSION, base='v4b_no_target_override',
                terminal_family='unchanged finite bounded vector cubic',
                terminal_horizon_mm=12., endpoint_trigger_tolerance_mm=1.,
                all_remaining_gates_checked=True,
                every_brake_command_interval_checked=True)
