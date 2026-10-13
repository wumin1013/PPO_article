from pathlib import Path
import math
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from terminal_braking_v3 import BrakingPolicy, discrete_jerk_stop_profile, plan_terminal_brake


LIMITS = dict(MAX_VEL=100., MAX_ACC=2000., MAX_JERK=62500.,
              MAX_ANG_VEL=2*np.pi, MAX_ANG_ACC=10*np.pi, MAX_ANG_JERK=1000.)
DT=.001
REAL_STATE=(12.790441626443162,-562.3767662953476,-.26926145027512516,-31.41592653589706)


def independently_check(tail, state):
    c=tail['columns'];v,a,w,alpha=state
    assert c['v_mm_s'][0]==v and c['a_mm_s2'][0]==a
    assert c['omega_rad_s'][0]==w and c['alpha_rad_s2'][0]==alpha
    for velocity,acceleration,jerk,A,J in (
        ('v_mm_s','a_mm_s2','j_mm_s3',LIMITS['MAX_ACC'],LIMITS['MAX_JERK']),
        ('omega_rad_s','alpha_rad_s2','angular_jerk_rad_s3',LIMITS['MAX_ANG_ACC'],LIMITS['MAX_ANG_JERK'])):
        measured_a=np.diff(c[velocity])/DT
        measured_j=np.diff(np.r_[c[acceleration][0],measured_a])/DT
        np.testing.assert_allclose(measured_a,c[acceleration][1:],atol=1e-11,rtol=1e-11)
        np.testing.assert_allclose(measured_j,c[jerk][1:],atol=1e-8,rtol=1e-11)
        assert np.max(np.abs(measured_a)) <= A*(1+1e-9)
        assert np.max(np.abs(measured_j)) <= J*(1+1e-9)
    assert np.min(c['v_mm_s']) >= 0
    assert np.max(c['v_mm_s']) <= LIMITS['MAX_VEL']*(1+1e-9)
    assert np.max(np.abs(c['omega_rad_s'])) <= LIMITS['MAX_ANG_VEL']*(1+1e-9)
    xy=np.c_[c['x_mm'],c['y_mm']]
    increments=DT*c['v_mm_s'][1:,None]*np.c_[np.cos(c['heading_rad'][1:]),np.sin(c['heading_rad'][1:])]
    np.testing.assert_allclose(np.diff(xy,axis=0),increments,atol=5e-14,rtol=1e-10)
    np.testing.assert_allclose(np.diff(c['heading_rad'])/DT,c['omega_rad_s'][1:],atol=2e-12)
    # Recompute from positions independently of exported v/a/omega/alpha.
    delta=np.diff(xy,axis=0)
    measured_v=np.r_[v,np.linalg.norm(delta,axis=1)/DT]
    measured_a=np.r_[a,np.diff(measured_v)/DT]
    measured_j=np.diff(measured_a)/DT
    measured_heading=[c['heading_rad'][0]]
    for displacement in delta:
        previous=measured_heading[-1]
        if np.linalg.norm(displacement)==0:
            measured_heading.append(previous)
        else:
            angle=math.atan2(displacement[1],displacement[0])
            measured_heading.append(previous+(angle-previous+math.pi)%(2*math.pi)-math.pi)
    measured_w=np.r_[w,np.diff(measured_heading)/DT]
    measured_alpha=np.r_[alpha,np.diff(measured_w)/DT]
    measured_angular_jerk=np.diff(measured_alpha)/DT
    for values,limit in ((measured_v,LIMITS['MAX_VEL']),(measured_a,LIMITS['MAX_ACC']),
                         (measured_j,LIMITS['MAX_JERK']),(measured_w,LIMITS['MAX_ANG_VEL']),
                         (measured_alpha,LIMITS['MAX_ANG_ACC']),(measured_angular_jerk,LIMITS['MAX_ANG_JERK'])):
        assert np.max(np.abs(values)) <= limit*(1+1e-5)
    assert tail['internal_terminal_rest']
    for key in ('v_mm_s','a_mm_s2','omega_rad_s','alpha_rad_s2'):
        assert np.all(c[key][-21:]==0)
    if w!=0 or alpha!=0:
        assert tail['feed_braking_steps']-tail['angular_braking_steps']>=10
        angular_last=int(np.flatnonzero((c['omega_rad_s']!=0)|(c['alpha_rad_s2']!=0))[-1])
        feed_last=int(np.flatnonzero((c['v_mm_s']!=0)|(c['a_mm_s2']!=0))[-1])
        assert feed_last-angular_last>=10


@pytest.mark.parametrize('state',[(5.,0.,0.,0.),(20.,100.,.2,1.),(5.,-100.,-.2,-1.),
    (50.,500.,1.,-2.),REAL_STATE,(100.,0.,0.,0.),(5.,0.,2*np.pi,0.),
    (5.,0.,-2*np.pi,0.),(5.,0.,0.,10*np.pi),(5.,0.,0.,-10*np.pi),
    (99.,250.,.2,-5.),(.2,-100.,0.,0.)])
def test_discrete_splice_bounds_and_complete_rest(state):
    tail=plan_terminal_brake([10.,20.],.3,*state,dt=DT,limits=LIMITS)
    independently_check(tail,state)


def test_real_legal_state_has_a_short_brake_instead_of_a_cubic_failure():
    from terminal_braking import plan_terminal_brake as old_brake
    with pytest.raises(ValueError,match='feed braking'):
        old_brake([23.638970173649394,5.038403049840184],-.006817399371685795,
                  *REAL_STATE,dt=DT,limits=LIMITS)
    tail=plan_terminal_brake([23.638970173649394,5.038403049840184],-.006817399371685795,
                            *REAL_STATE,dt=DT,limits=LIMITS)
    independently_check(tail,REAL_STATE)
    assert tail['braking_time_s'] < .15
    assert tail['total_tail_time_s'] < .18


def test_five_mm_s_straight_stop_does_not_wait_for_a_nonmoving_angular_channel():
    tail=plan_terminal_brake([0,0],0,5,0,0,0,dt=DT,limits=LIMITS)
    assert tail['braking_time_s'] <= .020
    assert tail['total_tail_time_s'] <= .043


def test_acceleration_plateau_avoids_long_slow_triangle_at_high_angular_speed():
    tail=plan_terminal_brake([0,0],0,5,0,2*np.pi,0,dt=DT,limits=LIMITS)
    independently_check(tail,(5,0,2*np.pi,0))
    assert tail['braking_time_s'] <= .250
    assert tail['angular_braking_steps'] <= 240


def test_zero_state_only_needs_the_auditable_hold_window():
    tail=plan_terminal_brake([0,0],0,0,0,0,0,dt=DT,limits=LIMITS)
    assert tail['braking_steps']==0
    assert tail['hold_steps']==23
    assert tail['total_tail_time_s']==pytest.approx(.023)


def test_lengthening_zero_acceleration_pulse_keeps_feed_positive_until_stop():
    result=discrete_jerk_stop_profile(12.79,-562.37,DT,limit_v=100,limit_a=2000,limit_j=62500,
                                    nonnegative=True,min_steps=100)
    v,a,_=result['profile']
    assert result['stop_steps']==100
    assert np.all(v[:-2]>0)
    assert v[-1]==v[-2]==a[-1]==0


@pytest.mark.parametrize('state',[(0.,-400.,0.,0.),(100.,400.,0.,0.),(5.,0.,2*np.pi,10.),
                                 (-1.,0.,0.,0.),(101.,0.,0.,0.)])
def test_invalid_or_nonviable_state_cannot_be_repaired(state):
    with pytest.raises(ValueError):
        plan_terminal_brake([0,0],0,*state,dt=DT,limits=LIMITS)


def test_an_already_stopped_feed_does_not_fake_angular_lead_by_zero_padding():
    with pytest.raises(ValueError,match='stationary'):
        plan_terminal_brake([0,0],0,0,0,1,0,dt=DT,limits=LIMITS)


def test_horizon_is_a_real_bound():
    with pytest.raises(ValueError):
        plan_terminal_brake([0,0],0,*REAL_STATE,dt=DT,limits=LIMITS,
                            policy=BrakingPolicy(max_braking_s=.002))


@pytest.mark.parametrize('dt',[0.,-1.,math.nan])
def test_invalid_sampling_is_rejected(dt):
    with pytest.raises(ValueError):
        plan_terminal_brake([0,0],0,5,0,0,0,dt=dt,limits=LIMITS)
