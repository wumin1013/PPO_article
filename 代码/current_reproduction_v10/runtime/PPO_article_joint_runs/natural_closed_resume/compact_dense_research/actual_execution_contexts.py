"""Reconstruct selected states from real complete executions or failure prefixes.

Unlike the preserved complete-trace helper, this explicitly records prefix
scope. It never invents a terminal marker or claims completion for a prefix.
"""
import copy
import json
import math
import numpy as np
from compact_farvector_runtime import FarVectorRuntime
from evaluate_historical_dense import sha


def reconstruct(trace,gates,geometry,corner_ids,dt):
    import joint_runtime
    assert joint_runtime.DT==dt and joint_runtime.MACRO_TICKS*dt==.02
    result=json.loads(trace.with_suffix('.json').read_text(encoding='utf-8'))
    assert result['trace_sha256']==sha(trace)
    with np.load(trace) as d:
        ref=d['reference'];rows=d['rows'];observations=d['observations'];actions=d['actions'];times=d['decision_times_s']
    events=json.loads(gates.read_text(encoding='utf-8'));crossing=np.array([e['command_segment'] for e in events])
    ticks=np.rint(times/dt).astype(int);terminal=np.flatnonzero(rows[:,12])
    stop=int(terminal[0]) if len(terminal) else len(rows)
    ids=np.flatnonzero((ticks>0)&(ticks<stop));ordinary=ticks[ids]
    assert np.all(np.diff(ticks)==round(.02/dt))
    runtime=FarVectorRuntime(ref);c=runtime.corridor
    frontiers=np.searchsorted(crossing,ordinary,side='left');segments=np.minimum(frontiers,len(c.lengths)-1)
    position=rows[ordinary-1,:2]
    along=np.clip(((position-c.starts[segments])*c.tangents[segments]).sum(1),0.,c.lengths[segments])
    progress=c.arcs[segments]+along;contexts=[]
    for ident in corner_ids:
        corner=next(v for v in geometry['sharp_corners'] if v['id']==ident)
        slot=int(np.flatnonzero(progress>=corner['reference_s_mm']-35.)[0]);index=int(ids[slot]);tick=int(ticks[index]);row=rows[tick-1]
        runtime=FarVectorRuntime(ref);state=runtime.state();physical=state['physical'];frontier=int(frontiers[slot])
        plane=float((row[:2]-c.gates[frontier].point)@c.gates[frontier].normal)
        assert abs(plane)>1e-8 and np.linalg.norm(row[8:10])>1e-6
        physical.update(position=row[:2].copy(),velocity=row[2:4].copy(),acceleration=row[4:6].copy(),
            heading=math.atan2(row[9],row[8]),held=actions[index-1].copy(),ticks=tick,decisions=index,rest_ticks=0)
        physical['tracker'].update(position=row[:2].copy(),frontier=frontier,pending_negative=False,inside=True,
            events=copy.deepcopy(events[:frontier]),segment_count=tick)
        runtime.restore(state);error=float(abs(runtime.observation()-observations[index]).max());assert error<2e-6
        contexts.append(dict(id=ident,state=state,tick=tick,decision_index=index,observation_error=error))
    return ref,rows,contexts,dict(source_success=result['success'],scope='complete_execution' if result['success'] else 'failure_prefix_states_only',
        trace_sha256=sha(trace),gates_sha256=sha(gates),dt_s=dt,maximum_observation_error=max(v['observation_error'] for v in contexts),
        requires_local_unmodified_replay=True)
