import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from terminal_braking import discrete_stop_profile, plan_terminal_brake


LIMITS = dict(MAX_VEL=100., MAX_ACC=2000., MAX_JERK=62500.,
              MAX_ANG_VEL=2*np.pi, MAX_ANG_ACC=10*np.pi, MAX_ANG_JERK=1000.)


@pytest.mark.parametrize('v,a,w,alpha', [(5,0,0,0), (20,100,.2,1),
                                      (5,-100,-.2,-1), (50,500,1,-2)])
def test_complete_discrete_brake(v,a,w,alpha):
    tail = plan_terminal_brake([10.,20.], .3, v,a,w,alpha,dt=.001,limits=LIMITS)
    c = tail['columns']
    assert tail['internal_terminal_rest']
    assert np.min(c['v_mm_s']) >= 0
    assert np.max(abs(np.diff(c['v_mm_s'])/.001)) <= LIMITS['MAX_ACC']
    assert np.max(abs(np.diff(c['a_mm_s2'])/.001)) <= LIMITS['MAX_JERK']
    assert np.max(abs(np.diff(c['omega_rad_s'])/.001)) <= LIMITS['MAX_ANG_ACC']
    assert np.max(abs(np.diff(c['alpha_rad_s2'])/.001)) <= LIMITS['MAX_ANG_JERK']
    expected = .001*c['v_mm_s'][1:,None]*np.c_[np.cos(c['heading_rad'][1:]),np.sin(c['heading_rad'][1:])]
    np.testing.assert_allclose(np.diff(np.c_[c['x_mm'],c['y_mm']],axis=0),expected,atol=1e-13)
    assert np.all(c['v_mm_s'][-21:] == 0)
    assert np.all(c['omega_rad_s'][-21:] == 0)


def test_profile_preserves_discrete_splice_and_complete_rest():
    v,a,j = discrete_stop_profile(5.,23.,.001,35)
    assert v[0] == 5 and a[0] == 23
    assert v[-1] == v[-2] == a[-1] == 0
    assert j[1] == pytest.approx((a[1]-23)/.001)


def test_invalid_state_cannot_be_repaired_by_zero_tail():
    with pytest.raises(ValueError):
        plan_terminal_brake([0,0],0,0,-400,0,0,dt=.001,limits=LIMITS)


def test_five_mm_s_stop_is_milliseconds_not_seconds():
    out=plan_terminal_brake([0,0],0,5,0,0,0,dt=.001,limits=LIMITS)
    assert out['braking_time_s'] < .05
    assert out['total_tail_time_s'] < .08


@pytest.mark.parametrize('sign', [-1.,1.])
def test_brake_accepts_a_valid_state_at_the_existing_angular_limit(sign):
    # The earlier v1 trial incorrectly applied its jerk margin to the velocity
    # limit too, making a valid saturated starting state appear infeasible.
    out=plan_terminal_brake([0,0],0,5,0,sign*LIMITS['MAX_ANG_VEL'],0,dt=.001,limits=LIMITS)
    assert out['internal_terminal_rest']
    assert np.max(abs(out['columns']['omega_rad_s'])) <= LIMITS['MAX_ANG_VEL']
