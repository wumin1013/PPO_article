"""Reconstruct ordinary-decision reset states from a witnessed complete trace.

No new physics is run here. Callers must check a local unmodified rollout before
using these resets: heading reconstruction can introduce floating-point roundoff.
"""
import copy
import json
import math
import numpy as np
from compact_farvector_runtime import FarVectorRuntime
from evaluate_historical_dense import sha


def from_complete_trace(trace, gates, geometry, *, before_mm=35., dt=.001):
    import joint_runtime
    stride = round(.020/dt)
    assert abs(joint_runtime.DT-dt)<1e-12
    assert joint_runtime.MACRO_TICKS==stride and abs(stride*dt-.020)<1e-12
    with np.load(trace) as data:
        reference = data['reference'].copy()
        rows = data['rows'].copy()
        observations = data['observations'].copy()
        actions = data['actions'].copy()
    events = json.loads(gates.read_text(encoding='utf-8'))
    crossing_ticks = np.array([e['command_segment'] for e in events])
    runtime = FarVectorRuntime(reference)
    ordinary = np.arange(len(observations))*stride
    ordinary = ordinary[ordinary < np.flatnonzero(rows[:,12])[0]]
    frontiers = np.searchsorted(crossing_ticks, ordinary, side='left')
    positions = np.vstack([reference[0], rows[ordinary[1:]-1,:2]])
    c = runtime.env.corridor
    segments = np.minimum(frontiers,len(c.lengths)-1)
    along = np.clip(np.sum((positions-c.starts[segments])*c.tangents[segments],axis=1),0.,c.lengths[segments])
    progress = c.arcs[segments]+along
    contexts = []
    for corner in geometry['sharp_corners']:
        index = int(np.flatnonzero(progress >= corner['reference_s_mm']-before_mm)[0])
        tick = int(ordinary[index])
        assert tick > 0 and tick%stride == 0
        runtime = FarVectorRuntime(reference)
        state = runtime.state()
        physical = state['physical']
        row = rows[tick-1]
        assert np.linalg.norm(row[8:10]) > 1e-6
        physical.update(position=row[:2].copy(),velocity=row[2:4].copy(),
            acceleration=row[4:6].copy(),heading=math.atan2(row[9],row[8]),
            held=actions[index-1].copy(),ticks=tick,decisions=index,rest_ticks=0)
        frontier = int(frontiers[index])
        plane_distance = float((row[:2]-c.gates[frontier].point)@c.gates[frontier].normal)
        assert abs(plane_distance)>1e-8, 'Contact states need explicit pending-negative reconstruction'
        physical['tracker'].update(position=row[:2].copy(),frontier=frontier,
            pending_negative=False,inside=True,events=copy.deepcopy(events[:frontier]),segment_count=tick)
        runtime.restore(state)
        error = float(np.max(np.abs(runtime.observation()-observations[index])))
        assert error < 2e-6,(corner['id'],error)
        contexts.append(dict(id=corner['id'],state=state,observation=runtime.observation(),
            decision_index=index,tick=tick,progress_mm=float(progress[index]),
            observation_reconstruction_error=error,heading_reconstructed_from_desired_vector=True))
    return reference, rows, contexts, dict(trace=str(trace),trace_sha256=sha(trace),
        gates=str(gates),gates_sha256=sha(gates),count=len(contexts),new_physics_ticks=0,
        dt_s=dt,decision_dt_s=.020,ordinary_stride_ticks=stride,
        maximum_observation_error=max(c['observation_reconstruction_error'] for c in contexts),
        requires_unmodified_local_rollout_check=True)
