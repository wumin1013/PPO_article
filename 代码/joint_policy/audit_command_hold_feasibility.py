"""Saved-path diagnostic of whether a20ms steering hold forces target reversal."""
import time
import math
import numpy as np
from audit_current_command_coordinates import command_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    data=read(parent/'results.json');pool=data['full_candidate'];model=parent/data['chosen']['model']
    stage=register('canonical_gate_band_current_hold_feasibility_20260918',
        [__file__,model,parent/'results.json',parent/'receipt.json',
         BASE/'audit_current_command_coordinates.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,isolated_corners=16),
        'Test the narrow hypothesis that a held20ms steering residual is structurally '
        'unable to avoid inward/outgoing-heading-cone violations under the recorded '
        '1ms guide motion. On each saved ordinary decision, intersect all residual '
        'angle intervals that would put the raw combined target between the incoming '
        'and outgoing tangents, plus the unchanged[-90,90]deg residual range. Feed '
        'and preview and saved positions stay fixed only for this algebraic diagnostic. '
        'An empty interval identifies a frozen-path time-structure conflict; a nonempty '
        'interval refutes unavoidable conflict on that recorded decision only. This '
        'does not define a trajectory template, deployed clamp, policy training target '
        'or prove physical feasibility after changing steering. No new motion or '
        'automatic candidate adoption. Next inspect shared-policy representability '
        'and full closed-loop response before proposing a new action mechanism.')
    start=time.perf_counter();records=[]
    for result,trace in zip(pool['results'],pool['traces']):
        if result['original_path']>=16:continue
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        u,v=np.diff(ref,axis=0)
        theta=math.atan2(u[0]*v[1]-u[1]*v[0],u@v)
        sign=np.sign(theta);turn=abs(math.degrees(theta))
        s=command_signals(ref,rows,actions)
        for k,action in enumerate(actions):
            a,b=k*20,min((k+1)*20,len(rows))
            if b-a!=20 or np.any(rows[a:b,12]!=0):continue
            los=s['los_deg'][a:b]
            lower=max(-90.,float(-los.min()))
            upper=min(90.,float(turn-los.max()))
            nominal=sign*math.degrees(action[0])
            violation=max(0.,lower-nominal,nominal-upper)
            required=(float(np.clip(nominal,lower,upper))-nominal if lower<=upper else None)
            records.append(dict(path=result['original_path'],decision=k,
                target_sweep_deg=float(los.max()-los.min()),turn_deg=turn,
                lower_residual_deg=lower,upper_residual_deg=upper,
                interval_nonempty=bool(lower<=upper),nominal_residual_deg=nominal,
                nominal_outside_interval_deg=violation,
                nearest_interval_correction_deg=required))
    affected=[r for r in records if r['nominal_outside_interval_deg']>.05]
    summary=dict(parent_sha256=sha(model),ordinary_decisions=len(records),
        empty_intervals=sum(not r['interval_nonempty'] for r in records),
        affected_decisions_at_005deg=len(affected),
        affected_paths=len({r['path'] for r in affected}),
        max_needed_residual_correction_deg=max(abs(r['nearest_interval_correction_deg']) for r in affected if r['interval_nonempty']),
        evidence_scope='Frozen saved paths only; no claim of new closed-loop feasibility.',
        learned_policy=False,goal_complete=False)
    write_json(stage/'results.json',dict(summary,records=records))
    receipt(stage,dict(status='completed_saved_command_hold_interval_audit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('HOLD_INTERVAL_AUDIT',summary,flush=True)


if __name__=='__main__':main()
