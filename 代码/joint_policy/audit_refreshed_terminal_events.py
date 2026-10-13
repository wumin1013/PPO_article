"""Read observed prebrake states and query the unchanged physical brake planner."""
from functools import lru_cache
import numpy as np
from gate_band_state import environment_factory
from joint_runtime import terminal
from study_io import BASE,write_json,sha
from search_complete_return import read


STAGE=BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920'


@lru_cache(maxsize=4096)
def terminal_event(trace,tick):
    with np.load(trace) as z:ref,rows=z['reference'],z['rows']
    begin=int(np.flatnonzero(rows[:,12])[0])
    if tick>begin:
        return dict(already_braking=True,initiation_tick=begin,total_ticks=len(rows))
    env=environment_factory(ref)
    for row in rows[:tick]:env.tracker.update(row[:2])
    state=rows[tick-1,:6] if tick else np.r_[ref[0],np.zeros(4)]
    env.position=state[:2].copy();env.velocity=state[2:4].copy();env.acceleration=state[4:6].copy()
    plan=terminal.plan_planar_terminal_brake(state[:2],state[2:4],state[4:6],dt=.001,limits=terminal.LIMITS)
    cols=plan['columns'];end=np.array([cols['x_mm'][-1],cols['y_mm'][-1]])
    progress,_=env.progress()
    return dict(already_braking=False,initiation_tick=begin,total_ticks=len(rows),query_tick=tick,
        covered_gates=env.tracker.frontier,required_gates=len(env.corridor.gates),
        remaining_mm=float(env.corridor.arcs[-1]-progress),
        endpoint_margin_mm=float(1.-np.linalg.norm(end-ref[-1])),
        brake_steps=int(plan['braking_steps']),planned_total_ticks=tick+len(cols['time_s'])-1,
        state6=state.tolist())


def main():
    history=read(STAGE/'state_joint/results.json')['candidates'];by={c['tag']:c for c in history}
    baseline={r['path']:r for r in read(STAGE/'baseline.json')['baseline']['records']}
    rows=[]
    for path in (304,305):
        with np.load(baseline[path]['trace']) as z:tick=int(np.flatnonzero(z['rows'][:,12])[0])
        for tag in ('reused_current_actor','fresh_proposal',*[f'refreshed_trust_{i}' for i in range(6)]):
            r=next(r for r in by[tag]['records'] if r['path']==path)
            q=terminal_event(str(r['trace']),tick)
            rows.append(dict(path=path,tag=tag,**q))
            print('TERMINAL_EVENT',path,tag,q,flush=True)
    write_json(STAGE/'terminal_event_error_audit.json',dict(records=rows,new_physics_ticks=0,
        new_actor_calls=0,results_sha256=sha(STAGE/'state_joint/results.json'),
        scope='Exact planner query at saved observed prebrake state; no probe or event rule is deployed.'))


if __name__=='__main__':main()
