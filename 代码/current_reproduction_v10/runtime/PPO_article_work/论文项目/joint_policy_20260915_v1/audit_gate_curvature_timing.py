"""Locate guide transitions relative to actual curvature peaks on saved paths."""
import time
import numpy as np
from geometry_spec import Corridor
from audit_current_command_coordinates import command_signals
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parent = BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    data = read(parent/'results.json'); model = parent/data['chosen']['model']
    pool = data['full_candidate']
    traces = {r['original_path']:BASE/t for r,t in zip(pool['results'],pool['traces'])}
    stage = register('canonical_gate_band_curvature_gate_timing_20260918',
        [__file__, model, parent/'results.json', parent/'receipt.json',
         BASE/'continuous_gate_guidance.py', BASE/'audit_current_command_coordinates.py']+
        [traces[p] for p in range(16)],
        dict(new_physics_ticks=0, new_actor_decisions=0, wall_s=30, isolated_corners=16),
        'Shape remains sharply concentrated despite current entry-side success. '
        'The guide blends across0.1mm whereas its state blends across2mm. '
        'Before a width intervention, compare actual physical curvature-peak '
        'timing with the real single gate plane and recorded combined/LOS '
        'target changes on all16 saved corners. Use unsmoothed1ms positions, '
        'velocity and acceleration. Timing overlap is association only; a '
        'peak before the gate can rule out that later blend as the onset '
        'cause on that path. Do not attribute all curvature or historical '
        'reverse learning to the guide, and do not rerun physics in this audit.')
    start=time.perf_counter();records=[]
    for path in range(16):
        with np.load(traces[path]) as z:
            ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        gate=Corridor(ref).gates[0]
        starts=np.vstack([ref[0],rows[:-1,:2]])
        h=(starts-gate.point)@gate.normal
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        ordinary=rows[:,12]==0; moving=ordinary&(speed>5.)
        curvature=np.zeros(len(rows))
        curvature[moving]=(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
        peak=int(np.argmax(abs(curvature)))
        cross=np.flatnonzero(ordinary&(h>=0.))
        first_cross=int(cross[0]) if len(cross) else None
        sig=command_signals(ref,rows,actions)
        changes=np.r_[0.,np.diff(sig['los_deg'])]
        candidate=np.flatnonzero(ordinary)
        los_peak=int(candidate[np.argmax(abs(changes[candidate]))])
        pre=(np.arange(len(rows))<=peak)&ordinary
        record=dict(path=path,physical_peak_curvature_mm_inv=float(abs(curvature[peak])),
            peak_tick=peak,peak_gate_plane_distance_mm=float(h[peak]),
            first_gate_crossing_tick=first_cross,
            peak_minus_crossing_ms=peak-first_cross if first_cross is not None else None,
            narrow_band_ticks=int((ordinary&(abs(h)<.05)).sum()),
            narrow_band_seen_before_peak=bool((pre&(abs(h)<.05)).any()),
            maximum_los_jump_deg=float(abs(changes[los_peak])),los_jump_tick=los_peak,
            los_jump_gate_plane_distance_mm=float(h[los_peak]),
            los_jump_inside_held_decision=bool(los_peak%20),
            peak_speed_mm_s=float(speed[peak]))
        records.append(record)
    write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,
        causality_proven=False,goal_complete=False))
    receipt(stage,dict(status='completed_saved_curvature_gate_timing',
        new_physics_ticks=0,new_actor_decisions=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('GATE_CURVATURE_TIMING',records,flush=True)


if __name__=='__main__':main()
