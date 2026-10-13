"""Offline hybrid prediction: affine ordinary motion, exact unchanged brake selection."""
import numpy as np
from joint_runtime import terminal
from terminal_duration_credit import fixed_duration_rows


def predict_terminal(reference,ordinary_rows,maximum_ordinary_ticks,factory,costs,check_budget):
    """Allow an earlier terminal event or a different selected brake duration.

    Ordinary prefixes are supplied predictions, never newly executed physics.
    Later initiation than supplied support is explicitly unsupported.
    """
    env=factory(reference)
    prefix=np.asarray(ordinary_rows)
    for tick in range(0,maximum_ordinary_ticks+1,20):
        if tick:
            for row in prefix[tick-20:tick]:env.tracker.update(row[:2])
            env.position=prefix[tick-1,:2].copy()
            env.velocity=prefix[tick-1,2:4].copy()
            env.acceleration=prefix[tick-1,4:6].copy()
        progress,_=env.progress()
        if env.tracker.frontier<len(env.corridor.gates) or env.corridor.arcs[-1]-progress>12.:
            continue
        check_budget();costs['terminal_planner_queries']+=1
        try:
            plan=terminal.plan_planar_terminal_brake(env.position,env.velocity,env.acceleration,
                dt=.001,limits=terminal.LIMITS)
        except ValueError as error:
            if str(error)!='No feasible nonreversing vector cubic brake within registered horizon':raise
            continue
        columns=plan['columns'];endpoint=np.array([columns['x_mm'][-1],columns['y_mm'][-1]])
        if np.linalg.norm(endpoint-reference[-1])>1.:continue
        state=np.r_[env.position,env.velocity,env.acceleration]
        tail=fixed_duration_rows(state,plan['braking_steps'],len(columns['time_s'])-1)
        rows=np.vstack([prefix[:tick,:6],tail])
        return dict(supported=True,rows=rows,initiation_tick=tick,
            braking_steps=int(plan['braking_steps']),total_ticks=len(rows),
            predicted_prefix_inside=bool(env.tracker.inside))
    return dict(supported=False,reason='No eligible brake within supplied ordinary prediction support',
        maximum_ordinary_ticks=maximum_ordinary_ticks)
