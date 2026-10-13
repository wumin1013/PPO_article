"""Ordered coverage and physical-time objective checks for development round two."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from preview_research_v2 import TimeIntegratedObjective
from src.utils.ordered_coverage import OrderedCoverageTracker
from src.utils.revision_metrics import ordered_projection


STRAIGHT = np.array([[0., 0.], [2., 0.]])
CORNER = np.array([[0., 0.], [2., 0.], [2., 2.]])
CLOSED = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.]])


@pytest.mark.parametrize('reference,positions', [
    (STRAIGHT, [[0., 0.], [.2, 0.], [1., 0.], [2., 0.]]),
    (STRAIGHT, [[0., 0.], [1., 0.], [.2, 0.], [1., 0.], [2., 0.]]),
    (CORNER, [[0., 0.], [1., .02], [2., 0.], [2., 1.], [2., 2.]]),
    (CORNER, [[0., 0.], [2., 2.]]),
    (CLOSED, [[0., 0.], [.01, 0.], [0., 0.], [0., 0.]]),
    (CLOSED, CLOSED),
    (STRAIGHT, [[0., 0.], [.4, 1.], [0., 0.], [2., 0.]]),
])
def test_streaming_matches_batch_at_every_command_sample(reference, positions):
    positions = np.asarray(positions)
    batch = ordered_projection(reference, positions, tolerance_mm=.1)
    tracker = OrderedCoverageTracker(reference, tolerance_mm=.1,
                                     initial_position=positions[0])
    snapshots = [tracker.snapshot()] + [tracker.update(p) for p in positions[1:]]
    np.testing.assert_array_equal([s['covered_witness_count'] for s in snapshots],
                                  batch['witness_frontier'])
    np.testing.assert_array_equal([s['ordered_progress'] for s in snapshots],
                                  batch['ordered_progress'])
    np.testing.assert_allclose([s['contour_error_mm'] for s in snapshots],
                               batch['contour_error_mm'], atol=1e-12)
    assert snapshots[-1]['ordered_coverage_complete'] == batch['ordered_coverage_complete']
    assert sum(s['delta_s_mm'] for s in snapshots) == pytest.approx(
        snapshots[-1]['covered_s_mm'] - snapshots[0]['initial_covered_s_mm'])


def test_initial_coverage_and_retracing_never_award_progress_twice():
    tracker = OrderedCoverageTracker(STRAIGHT, tolerance_mm=.1, initial_position=[0., 0.])
    assert tracker.initial_covered_s_mm > 0.
    assert tracker.update([0., 0.])['delta_s_mm'] == 0.
    first = tracker.update([1., 0.])
    assert first['delta_s_mm'] > 0.
    assert tracker.update([.1, 0.])['delta_s_mm'] == 0.
    assert tracker.update([1., 0.])['delta_s_mm'] == 0.
    final = tracker.update([2., 0.])
    assert final['completion_valid']
    assert first['delta_s_mm'] + final['delta_s_mm'] == pytest.approx(
        2. - tracker.initial_covered_s_mm)


def test_closed_start_does_not_complete_and_full_lap_does():
    tracker = OrderedCoverageTracker(CLOSED, tolerance_mm=.1, initial_position=CLOSED[0])
    assert not tracker.update(CLOSED[0])['completion_valid']
    assert tracker.snapshot()['ordered_progress'] < .02
    for point in CLOSED[1:]:
        result = tracker.update(point)
    assert result['completion_valid']


def test_shortcut_with_endpoints_on_reference_cannot_skip_ordered_branches():
    reference = [[0., 0.], [0., 2.], [.2, 2.], [.2, 0.], [2., 0.]]
    tracker = OrderedCoverageTracker(reference, tolerance_mm=.1, initial_position=[0., 0.])
    result = tracker.update([2., 0.])
    assert result['geometry_feasible']
    assert result['endpoint_error_mm'] == 0.
    assert not result['ordered_coverage_complete']
    assert not result['completion_valid']


def test_long_command_segment_on_straight_reference_is_fully_covered():
    tracker = OrderedCoverageTracker(STRAIGHT, tolerance_mm=.1, initial_position=STRAIGHT[0])
    assert tracker.update(STRAIGHT[-1])['completion_valid']


def test_previous_contour_violation_survives_a_later_complete_traversal():
    tracker = OrderedCoverageTracker(STRAIGHT, tolerance_mm=.1, initial_position=STRAIGHT[0])
    tracker.update([.3, 1.])
    tracker.update(STRAIGHT[0])
    result = tracker.update(STRAIGHT[-1])
    assert result['ordered_coverage_complete']
    assert result['contour_error_mm'] == 0.
    assert result['contour_error_max_mm'] == 1.
    assert not result['geometry_feasible']
    assert not result['completion_valid']


class RateEnvironment:
    """Deterministic commands with deliberately unrelated legacy progress reward."""
    def __init__(self, dt, positions, reason='success', tolerance=.1, raw_progress=12345.):
        self.interpolation_period = dt
        self.positions = np.asarray(positions, dtype=float)
        self.done_reason = reason
        self.half_epsilon = tolerance
        self.raw_progress = raw_progress
        self.reset()

    def reset(self):
        self.index = 0
        self.current_position = self.positions[0].copy()
        return np.zeros(18)

    def step(self, action):
        self.index += 1
        self.current_position = self.positions[self.index].copy()
        done = self.index == len(self.positions) - 1
        components = dict(r_progress=self.raw_progress, r_track=-2., r_time=-1.,
                          r_stall=-8. if done and self.done_reason != 'success' else 0.)
        info = dict(reward_components=components,
                    done_reason=self.done_reason if done else 'running')
        return np.zeros(18), sum(components.values()), done, info


def objective(positions, reference=STRAIGHT, dt=.001, reason='success', **kwargs):
    env = RateEnvironment(dt, positions, reason=reason, **kwargs)
    cfg = dict(revision=dict(completion_bonus=10., failure_penalty=-20.,
                             progress_reward_per_mm=1.))
    return TimeIntegratedObjective(env, cfg, reference)


def rollout(positions, **kwargs):
    env = objective(positions, **kwargs)
    records = [env.step([0., .05]) for _ in positions[1:]]
    return sum(item[1] for item in records), records[-1][3]


def test_equal_physical_motion_has_same_return_at_two_sampling_rates():
    # 1 mm/s for 2 seconds: coverage delta, integrated rates and terminal match.
    p1 = np.column_stack([np.linspace(0., 2., 2001), np.zeros(2001)])
    p2 = np.column_stack([np.linspace(0., 2., 1001), np.zeros(1001)])
    total1, info1 = rollout(p1, dt=.001)
    total2, info2 = rollout(p2, dt=.002)
    expected = 2. - info1['ordered_coverage']['initial_covered_s_mm'] - 6. + 10.
    assert total1 == pytest.approx(expected, abs=1e-10)
    assert total1 == pytest.approx(total2, abs=1e-10)
    assert info1['done_reason'] == info2['done_reason'] == 'success'


def test_raw_progress_is_removed_and_stationary_start_gets_no_distance_reward():
    env = objective([[0., 0.], [0., 0.], [2., 0.]], raw_progress=100000.)
    _, reward, done, info = env.step([0., .05])
    assert not done
    assert reward == pytest.approx(-.003)
    assert info['reward_components']['r_progress'] == 0.
    assert info['reward_components']['running_cost_rate'] == -3.
    assert info['reward_components']['original_per_tick_total'] == 99997.


def test_feasible_completion_is_better_than_early_stall_in_matched_trace():
    successful, _ = rollout([[0., 0.], [2., 0.]], dt=2.)
    stalled, info = rollout([[0., 0.], [0., 0.]], dt=.3, reason='stall')
    assert successful > stalled
    assert info['done_reason'] == 'stall'
    assert info['reward_components']['r_stall'] == 0.
    assert info['reward_components']['r_terminal'] == -20.


def test_terminal_event_is_not_scaled_down_with_dt():
    successful, _ = rollout([[0., 0.], [2., 0.]], dt=.001)
    failed, _ = rollout([[0., 0.], [2., 0.]], dt=.001, reason='timeout')
    assert successful - failed == pytest.approx(30.)


@pytest.mark.parametrize('reference,positions', [
    (CLOSED, [[0., 0.], [0., 0.]]),
    (CORNER, [[0., 0.], [2., 2.]]),
    (STRAIGHT, [[0., 0.], [.3, 1.], [0., 0.], [2., 0.]]),
])
def test_false_environment_success_receives_failure_penalty(reference, positions):
    _, info = rollout(positions, reference=reference)
    assert info['environment_done_reason'] == 'success'
    assert info['done_reason'] == 'ordered_incomplete'
    assert info['reward_components']['r_terminal'] == -20.
    assert not info['ordered_coverage']['completion_valid']


def test_terminal_reward_requires_reset_before_next_episode():
    env = objective([[0., 0.], [2., 0.]])
    first = env.step([0., .05])
    with pytest.raises(RuntimeError, match='Reset'):
        env.step([0., .05])
    env.reset()
    assert env.coverage.snapshot()['delta_s_mm'] == 0.
    assert not env.coverage.snapshot()['completion_valid']
    second = env.step([0., .05])
    assert first[1] == second[1]


@pytest.mark.parametrize('reference', [
    [[0., 0.], [0., 0.]], [[0., 0.], [np.nan, 0.]], [[0., 0.]],
])
def test_invalid_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        OrderedCoverageTracker(reference, tolerance_mm=.1, initial_position=[0., 0.])
