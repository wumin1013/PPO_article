"""Experimental speed-scaled LOS pursuit; frozen controllers are never edited.

D remains the policy's physical arc lookahead. Fixed and adaptive methods use
the same geometry-to-yaw conversion, followed by the original angular KCM.
"""
from contextlib import contextmanager
import math

import numpy as np

from src.environment.cnc_env import Env


def pursuit_rate(speed_mm_s, heading_error_rad, chord_mm):
    """Circle-through-target curvature times the preceding executed feed.

    The reference angle retains the registered once-per-tick EMA. There is no
    positive speed floor, distance selection rule or hard-bound bypass here.
    """
    speed, error, chord = map(float, (speed_mm_s, heading_error_rad, chord_mm))
    if not all(map(math.isfinite, (speed, error, chord))) or speed < 0 or chord < 0:
        raise ValueError('Finite nonnegative speed/chord and finite heading error required')
    if chord <= 1e-9:
        return 0.0
    return 2.0 * speed * math.sin(error) / chord


def speed_scaled_heading_intent(self, policy_omega, velocity_intent):
    if abs(float(policy_omega)) > 1e-12:
        raise ValueError('This registered pursuit candidate has no angular policy residual')
    theta_ref = float(self._advance_path_reference(self.current_position, v_exec=velocity_intent))
    heading = float(self.heading)
    error = float(self._wrap_angle(theta_ref-heading))
    _, _, arc, _, _ = self._project_onto_progress_path(self.current_position, update=False)
    distance = float(self._lookahead_dist_active)
    target = np.asarray(self._interpolate_progress_point_at_s(float(arc)+distance), dtype=float)
    chord = float(np.linalg.norm(target-self.current_position))
    speed = float(self.velocity)
    intent = pursuit_rate(speed, error, chord)
    self._heading_intent_status = dict(mode=3., theta_ref=theta_ref, heading_pred=heading,
        heading_corr=error, heading_desired=self._wrap_angle(heading+intent*self.interpolation_period),
        heading_rate_policy=0., heading_rate_intent=intent, blend_active=0.,
        pursuit_chord_mm=chord, pursuit_previous_feed_mm_s=speed,
        pursuit_curvature_per_mm=0. if chord <= 1e-9 else 2.*math.sin(error)/chord)
    return intent


@contextmanager
def speed_scaled_execution():
    original = Env._resolve_heading_intent
    if original is speed_scaled_heading_intent:
        raise RuntimeError('Pursuit wrapper already installed')
    Env._resolve_heading_intent = speed_scaled_heading_intent
    try:
        yield
    finally:
        Env._resolve_heading_intent = original
