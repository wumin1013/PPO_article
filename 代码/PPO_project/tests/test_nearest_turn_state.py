"""Physical corner cases for the proposed observation-only geometry helper."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'PPO_project/tools'))
from nearest_turn_state import NearestTurnGeometry, nearest_turn_state


def polyline(headings_deg, lengths=None):
    heads = np.radians(headings_deg)
    lengths = np.ones(len(heads)) if lengths is None else np.asarray(lengths)
    vectors = lengths[:, None]*np.c_[np.cos(heads), np.sin(heads)]
    return np.r_[[[0., 0.]], np.cumsum(vectors, axis=0)]


@pytest.mark.parametrize('split,distance,angle,end', [
    ('train', .65, -100., 1), ('validation', .8, -125., 2)])
def test_real_dense_path_reports_first_corner_not_far_curvature_maximum(split, distance, angle, end):
    reference = np.loadtxt(ROOT/f'PPO_project/data/preview_event_v1/paths/dense_corners_{split}.csv',
                           delimiter=',', skiprows=1)
    state = nearest_turn_state(reference, 0., 1.25)
    assert state['dist_to_turn'] == pytest.approx(distance, abs=1e-10)
    assert math.degrees(state['turn_angle']) == pytest.approx(angle, abs=1e-8)
    assert state['turn_sign'] == -1
    assert (state['start_vertex'], state['end_vertex']) == (1, end)


def test_opposite_turns_are_not_cancelled_or_merged():
    geometry = NearestTurnGeometry(polyline([0, 90, 0, -90]), one_sided_tolerance_mm=1.25)
    first = geometry.at(0.)
    assert first['turn_angle'] == pytest.approx(math.pi/2)
    assert first['dist_to_turn'] == pytest.approx(1.)
    assert first['vertex_count'] == 1
    second = geometry.at(1.)
    assert second['dist_to_turn'] == pytest.approx(1.)
    assert second['turn_sign'] == -1
    assert second['turn_angle'] == pytest.approx(-math.pi)
    assert (second['start_vertex'], second['end_vertex']) == (2, 3)


def test_same_direction_discretized_curve_retains_aggregate_turn():
    reference = polyline(np.arange(0., 95., 5.), [.1]*19)
    geometry = NearestTurnGeometry(reference, one_sided_tolerance_mm=1.25)
    state = geometry.at(0.)
    assert state['dist_to_turn'] == pytest.approx(.1)
    assert state['turn_angle'] == pytest.approx(math.pi/2)
    assert state['turn_sign'] == 1
    assert state['vertex_count'] == 18
    at_vertex = geometry.at(.1)
    assert at_vertex['start_vertex'] == 2
    assert math.degrees(at_vertex['turn_angle']) == pytest.approx(85.)


def test_large_same_direction_curve_never_wraps_to_opposite_sign():
    reference = polyline(np.arange(0., 305., 5.), [.1]*61)
    state = nearest_turn_state(reference, 0., 1.25)
    assert state['turn_sign'] == 1
    assert math.degrees(state['turn_angle']) == pytest.approx(300.)


def test_nonsignificant_vertex_separates_same_direction_groups():
    # A straight interlude separates two positive corners.
    state = nearest_turn_state(polyline([0, 90, 90, 180]), 0., 1.25)
    assert state['vertex_count'] == 1
    assert (state['start_vertex'], state['end_vertex']) == (1, 1)
    assert state['turn_angle'] == pytest.approx(math.pi/2)


def test_far_larger_curvature_cannot_move_anchor_inside_a_group():
    state = nearest_turn_state(polyline([0, 20, 100], [1, 1, .1]), 0., 1.25)
    assert state['turn_s'] == pytest.approx(1.)
    assert (state['start_vertex'], state['end_vertex']) == (1, 2)
    assert math.degrees(state['turn_angle']) == pytest.approx(100.)


@pytest.mark.parametrize('s', [2., 2.1, 1000.])
def test_endpoint_has_no_next_corner(s):
    state = nearest_turn_state([[0, 0], [1, 0], [1, 1]], s, 1.25)
    assert math.isinf(state['dist_to_turn'])
    assert state['turn_angle'] == 0 and state['turn_sign'] == 0
    assert state['start_vertex'] is None and state['end_vertex'] is None
    assert state['turn_s'] is None and state['vertex_count'] == 0


def test_finite_closed_reference_does_not_wrap_at_end():
    ref = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    assert nearest_turn_state(ref, 4., 1.25)['vertex_count'] == 0


def test_straight_two_point_reference_has_no_corner():
    assert nearest_turn_state([[0, 0], [5, 0]], 0., 1.25)['vertex_count'] == 0


def test_existing_curvature_threshold_and_configured_override():
    ref = polyline([0, 10])
    assert nearest_turn_state(ref, 0., 1.25)['vertex_count'] == 0
    assert nearest_turn_state(ref, 0., 2.)['vertex_count'] == 1
    assert nearest_turn_state(ref, 0., 2., kappa_threshold=1.)['vertex_count'] == 0


def test_queries_do_not_mutate_reference_or_remember_query_order():
    ref = polyline([0, 90, 0])
    original = ref.copy()
    geometry = NearestTurnGeometry(ref, one_sided_tolerance_mm=1.25)
    first = geometry.at(0.)
    geometry.at(2.)
    assert geometry.at(0.) == first
    assert geometry.at(-10.) == first
    np.testing.assert_array_equal(ref, original)
    ref[:] = -9
    assert geometry.at(0.) == first


@pytest.mark.parametrize('s', [math.nan, math.inf, -math.inf])
def test_nonfinite_arc_rejected(s):
    with pytest.raises(ValueError, match='s_mm'):
        nearest_turn_state([[0, 0], [1, 0]], s, 1.25)


@pytest.mark.parametrize('tolerance', [0., -1., math.inf, math.nan])
def test_invalid_tolerance_rejected(tolerance):
    with pytest.raises(ValueError, match='tolerance'):
        nearest_turn_state([[0, 0], [1, 0]], 0., tolerance)


@pytest.mark.parametrize('reference', [[], [[0, 0]], [[0, 0, 0], [1, 0, 0]],
                                      [[0, 0], [0, 0]], [[0, 0], [math.nan, 0]]])
def test_invalid_geometry_rejected(reference):
    with pytest.raises(ValueError):
        nearest_turn_state(reference, 0., 1.25)
