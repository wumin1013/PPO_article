"""Finite local transition Jacobians with explicit mode and failure accounting."""
import numpy as np
from probe_closed_loop_tangent_credit import STATE_SCALE, ACTION_SCALE, normalized_state, shift_state


def transition_jacobian(env, policy, state, latent, nominal_rows, next_state,
                        epsilon, terminal, costs, budget_check):
    """Return normalized next-state and original1ms row responses.

    Ordinary transitions use centered differences. Terminal credit is analytic
    within the frozen selected duration; event guards must retain that branch.
    """
    count=len(nominal_rows);sj=np.zeros((7,10));rj=np.zeros((count,6,10))
    if terminal:
        from joint_runtime import terminal as planner
        from terminal_duration_credit import fixed_duration_rows, fixed_duration_jacobian
        budget_check()
        costs['terminal_planner_queries']=costs.get('terminal_planner_queries',0)+1
        plan=planner.plan_planar_terminal_brake(state['position'],state['velocity'],state['acceleration'],
                                               dt=.001,limits=planner.LIMITS)
        state6=np.r_[state['position'],state['velocity'],state['acceleration']]
        assert nominal_rows[:,12].all() and len(plan['columns']['time_s'])-1==count
        assert np.max(np.abs(fixed_duration_rows(state6,plan['braking_steps'],count)-nominal_rows[:,:6]))<1e-7
        return sj,fixed_duration_jacobian(plan['braking_steps'],count),['exact_fixed_duration']*10,[]
    nominal_next=normalized_state(next_state) if next_state is not None else None
    modes=[];invalid=[]
    for axis in range(10):
        pair=[]
        for sign in (-1,1):
            budget_check()
            env.restore(shift_state(state,axis,sign*epsilon) if axis<7 else state)
            action=latent.copy()
            if axis<7:
                action=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
            else:action[axis-7]+=sign*epsilon*ACTION_SCALE[axis-7]
            initial_tick=env.ticks
            try:
                _,_,_,info=env.step(action)
                result=info['rows']
                if len(result)!=20 or env.done or result[:,12].any():
                    raise ValueError('Changed ordinary transition mode or length')
                pair.append((normalized_state(env.state()),result[:,:6].copy()))
            except ValueError as exc:
                pair.append(None);invalid.append(dict(axis=axis,sign=sign,error=str(exc)))
            finally:
                costs['new_physics_ticks']+=env.ticks-initial_tick
        if all(v is not None for v in pair):
            sj[:,axis]=(pair[1][0]-pair[0][0])/(2*epsilon)
            rj[:,:,axis]=(pair[1][1]-pair[0][1])/(2*epsilon);mode='central'
        elif any(v is not None for v in pair):
            good=0 if pair[0] is not None else 1;sign=(-1,1)[good]
            sj[:,axis]=(pair[good][0]-nominal_next)/(sign*epsilon)
            rj[:,:,axis]=(pair[good][1]-nominal_rows[:,:6])/(sign*epsilon);mode='one_sided'
        else:
            raise ValueError(f'Both local derivative queries invalid at axis{axis}: {invalid[-2:]}')
        modes.append(mode)
    return sj,rj,modes,invalid
