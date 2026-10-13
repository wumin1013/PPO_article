"""Refine rest boundaries of a completed planned command trajectory.

This is an offline command-planning stage. Planning observations remain bound
to the original policy trajectory, never to the refined physical commands.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
from boundary_timewarp import candidates,geometry_prefix,JOIN_S

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'terminal_research'))
import terminal_candidate


def refine(reference,positions,rows,*,dt,tolerance,completed):
    from common import ROOT as BASE
    from runtime_bootstrap import setup
    setup()
    import joint_runtime as joint
    from geometry_spec import Corridor,GateTracker
    from metrics_exact_index_v5 import metrics
    joint.DT=dt
    sys.path.insert(0,str(BASE/'literature/ruckig_vendor'))
    import ruckig
    started=time.perf_counter()
    attempts=[];prefixes=[];saved=0
    source=positions
    terminal_diagnostic=None
    start_cut=round(JOIN_S/dt)
    terminal_cut=int(np.flatnonzero(rows[:,12]>.5)[0]) if len(rows) and np.any(rows[:,12]>.5) else None
    if completed and terminal_cut is not None and terminal_cut>start_cut+3:
        for prefix,item in candidates(source,dt):
            item['geometry_pass']=False
            if item['vector_pass']:
                item['geometry_pass']=geometry_prefix(reference,prefix,source[:start_cut+1],tolerance=tolerance)
            attempts.append(item);prefixes.append(prefix)
            if not saved and item['vector_pass'] and item['geometry_pass']:
                selected_prefix=prefix;saved=item['saved_ticks']
        if not saved:selected_prefix=source[:start_cut+1]
        # The full original suffix and terminal entrance state are unchanged.
        corridor=Corridor(reference,epsilon=tolerance,
            closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
        tracker=GateTracker(corridor,source[0])
        for point in source[1:terminal_cut+1]:tracker.update(point)
        assert tracker.inside
        position=source[terminal_cut]
        velocity=(source[terminal_cut]-source[terminal_cut-1])/dt
        previous_velocity=(source[terminal_cut-1]-source[terminal_cut-2])/dt
        acceleration=(velocity-previous_velocity)/dt
        old_points=source[terminal_cut:]
        old_vectors=np.vstack((velocity,np.diff(old_points,axis=0)/dt))
        old_columns=dict(time_s=np.arange(len(old_points))*dt,
            x_mm=old_points[:,0],y_mm=old_points[:,1],
            velocity_x_mm_s=old_vectors[:,0],velocity_y_mm_s=old_vectors[:,1])
        env=SimpleNamespace(position=position,velocity=velocity,acceleration=acceleration,
            corridor=corridor,tracker=tracker,brake=old_columns,brake_index=0)
        terminal_candidate.consider(env,original_consider=lambda _:None,joint=joint,ruckig=ruckig)
        terminal_diagnostic=env.v9_terminal_diagnostics
        tail=np.column_stack((env.brake['x_mm'],env.brake['y_mm']))
        positions=np.vstack((selected_prefix,source[start_cut+1:terminal_cut+1],tail[1:]))
        assert np.array_equal(positions[start_cut-saved:terminal_cut-saved+1],source[start_cut:terminal_cut+1])
    audited=metrics(reference,positions,dt=dt,tolerance=tolerance,
        declared_success=completed,reason='original_planning_failure' if not completed else None)
    attempted_audit=audited
    fallback=False
    if completed and not audited['success']:
        positions=source;fallback=True
        audited=metrics(reference,positions,dt=dt,tolerance=tolerance)
        assert audited['success'],'Original registered planning trace no longer passes'
    evidence=dict(version='v9_offline_boundary_refinement',command_stage='offline after original policy command planning',
        original_planning_success=completed,planning_observations_are_execution_observations=False,
        start_candidates=attempts,start_selected_saved_ticks=saved if not fallback else 0,
        terminal_candidate_diagnostics=terminal_diagnostic,original_terminal_start_tick=terminal_cut,
        unchanged_body_start_original_tick=start_cut,complete_candidate_audit=attempted_audit,
        whole_plan_fallback=fallback,refinement_wall_s=time.perf_counter()-started)
    return positions,audited,evidence,prefixes
