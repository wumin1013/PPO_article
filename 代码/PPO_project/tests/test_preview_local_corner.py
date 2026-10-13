from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PPO_project'),str(ROOT/'PPO_project/tools')]
import preview_local_corner_study as local
import preview_positive_feed_study as positive


def test_corner_state_uses_first_local_event_without_mutating_controller():
    cfg=local.configuration('preview_time','adaptive',101)
    before=positive.PositiveFeedEnvironment(cfg,'dense_corners_validation')
    after=local.LocalCornerEnvironment(cfg,'dense_corners_validation')
    unchanged=[i for i in range(26) if i not in range(7,12)]
    np.testing.assert_array_equal(before.state[unchanged],after.state[unchanged])
    assert after.trace[-1]['local_turn_distance_mm']==pytest.approx(.8)
    assert after.state[7]==pytest.approx(.8/60.)
    assert after.state[10]==1.
    assert before.inner.raw._lookahead_dist_active==after.inner.raw._lookahead_dist_active
    for _ in range(20):
        command=np.array([0.,-3.,.5])
        a=before.step(command);b=after.step(command)
        np.testing.assert_allclose(before.inner.raw.current_position,after.inner.raw.current_position,atol=0,rtol=0)
        assert a[1]==b[1] and a[2]==b[2]
        np.testing.assert_array_equal(before.state[unchanged],after.state[unchanged])


@pytest.mark.parametrize('method',local.METHODS)
def test_fixed_controls_share_state_correction_and_keep_registered_distance(method):
    cfg=local.configuration('preview_time',method,901)
    env=local.LocalCornerEnvironment(cfg,'unit',reference=np.array([[0.,0.],[3.,0.],[3.,4.]]))
    env.step(np.array([0.,-3.,.1]))
    expected=1.18 if method=='adaptive' else local.core.DISTANCES[method]
    assert env.trace[-1]['lookahead_mm']==pytest.approx(expected)
    assert env.state.shape==(26,)
    assert np.isfinite(env.state).all()
