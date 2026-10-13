"""Matched temporal action-interface probes on the retained86cd policy."""
import time
import math
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from linear_action_hold_runtime import LinearActionHoldRuntime
from probe_coordinated_suffix import reconstruct, encode
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from audit_function_constraint_interference import command_segment_margin
from validate_actions import difference_audit
from joint_runtime import FILTER_TAU, wrap
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def target_motion(reference, rows):
    u, v = np.diff(reference, axis=0)
    sign = np.sign(u[0]*v[1]-u[1]*v[0])
    origin = math.atan2(u[1], u[0])
    filtered = np.arctan2(rows[:, 9], rows[:, 8])
    previous = np.r_[origin, filtered[:-1]]
    combined = previous + rows[:, 10]*.001/(1-math.exp(-.001/FILTER_TAU))
    combined = sign*np.array([wrap(a-origin) for a in combined])*180/math.pi
    ordinary = rows[:, 12] == 0
    pairs = ordinary[1:] & ordinary[:-1]
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    heading = sign*np.array([wrap(a-origin) for a in np.arctan2(rows[:, 3], rows[:, 2])])*180/math.pi
    apex = int(np.linalg.norm(rows[:, :2]-reference[1], axis=1).argmin())
    before = (np.arange(len(rows)) < apex) & ordinary & (speed > 5.)
    first = np.flatnonzero(before & (abs(heading) > .05))
    return dict(maximum_target_jump_deg=float(abs(np.diff(combined))[pairs].max()),
        first_turn_inside=bool(heading[first[0]]>0) if len(first) else None,
        negative_preapex_physical_heading_ms=int((before & (heading < -.05)).sum()))


def main():
    torch.set_num_threads(1)
    parent = BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918'
    saved = read(parent/'results.json'); pool = saved['full_candidate']
    adopted = read(parent/'adoption.json'); model = BASE/adopted['model']
    assert sha(model) == adopted['sha256'] and pool['checks']['eligible']
    paths = [6,7,12,13,14,15,19,31,57]
    indexed = {r['original_path']:(r, BASE/t) for r,t in zip(pool['results'], pool['traces'])}
    prior = BASE/'canonical_gate_band_peak_preview_coupling_20260918'
    temporal = BASE/'canonical_gate_band_target_jump_temporal_probe_20260918'
    stage = register('canonical_gate_band_linear_action_hold_probe_20260918',
        [__file__, BASE/'linear_action_hold_runtime.py', model, parent/'results.json',
         parent/'adoption.json', parent/'receipt.json', prior/'results.json', temporal/'results.json',
         BASE/'joint_runtime.py', BASE/'gate_band_state.py', BASE/'validate_actions.py']+
         [indexed[p][1] for p in paths],
        dict(paths=paths, arms=['preview_only','all_three'], new_complete_paths=18,
            action_interval_ms=20, physical_tick_ms=1, identity_physics_ticks=20,
            actor_calls=6000, physics_ticks=140000, saved_geometry_ticks=1000, wall_s=180),
        'Previous399a fixed-state and full-suffix probes show large20ms preview '
        'jumps, partial steering cancellation and adverse effects of simple late '
        'damping/early pulses. Test a generic temporal action representation on '
        'the NEW retained86cd policy, whose exact nominal full paths are reused. '
        'Compare only-preview first-order hold with all-three first-order hold; '
        'each interpolates the preceding physical action endpoint to the newly '
        'selected endpoint in20 original1ms execution ticks. No extra decisions, '
        'state/trajectory interpolation, acceleration reset or altered V/A/J, '
        'coverage, braking or rest. Zero interpolation must reproduce one saved '
        'decision bitwise before18 complete probes. Record actually executed '
        '1ms actions separately from requested endpoints; old held-action '
        'reconstruction must NOT be applied to these traces. The network is '
        'frozen and masks action history, so these interventions alone cannot '
        'establish learning or rule out a retrained history-aware policy. '
        'Report startup/complete time, raw XY shape, curvature, cruise, failures '
        'and actual target jumps. No failed-prefix benefit or adoption; stop '
        'after18 paths. If phase lag worsens coverage/shape, do not deploy this '
        'filter or continue training it without revising the mechanism.')
    start = time.perf_counter(); deadline = start+175.
    costs = dict(actor_calls=0, physics_ticks=0, saved_prefix_geometry_ticks=0)
    md = torch.load(model, map_location='cpu', weights_only=False)
    policy = CanonicalGatePolicy(256); policy.load_state_dict(md['policy'])
    factory = checkpoint_environment_factory(md['architecture'])
    # Check the instrumentation, using just one existing ordinary decision.
    with np.load(indexed[6][1]) as z:
        ref, rows, actions = [z[k].copy() for k in ('reference','rows','actions')]
    frame=400
    base=reconstruct(ref,rows,actions,frame,costs,env_factory=factory)
    check=LinearActionHoldRuntime(ref,interpolate_heads=(),max_time=base.max_ticks*.001)
    check.restore(base.state())
    decode_error=float(abs(check.decode_action(encode(actions[frame//20]))-actions[frame//20]).max())
    latent=policy.act(check.observation(),True)[0];costs['actor_calls']+=1
    assert np.array_equal(check.decode_action(latent),actions[frame//20])
    _,_,_,info=check.step(latent);costs['physics_ticks']+=len(info['rows'])
    assert np.array_equal(info['rows'],rows[frame:frame+20])
    write_json(stage/'identity_check.json',dict(original_latent_bitwise_equal=True,
        inverse_encoded_max_action_error=decode_error,physical_ticks=20))
    records=[]
    try:
        for arm, heads in [('preview_only',(2,)),('all_three',(0,1,2))]:
            for path in paths:
                old, trace=indexed[path]
                with np.load(trace) as z:ref,nominal=[z[k].copy() for k in ('reference','rows')]
                length=np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()
                env=LinearActionHoldRuntime(ref,interpolate_heads=heads,
                    max_time=max(5.,float(length)/20.+1.),closed=bool(np.linalg.norm(ref[0]-ref[-1])<1e-9))
                chunks=[];endpoints=[];executed=[]
                while not env.done:
                    assert time.perf_counter()<deadline and costs['actor_calls']<6000 and costs['physics_ticks']<137800
                    latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
                    _,_,_,info=env.step(latent)
                    chunks.append(info['rows']);endpoints.append(info['requested_action'])
                    executed.append(info['executed_actions_1ms']);costs['physics_ticks']+=len(info['rows'])
                full=np.vstack(chunks);commands=np.vstack(executed)
                audit=difference_audit(ref[0],full)
                corridor=command_segment_margin(ref,full[:,:2],tolerance=1e-10)
                success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'] and
                             corridor['max_distance_upper_mm']<=1.25+1e-8)
                filename=f'path{path}_{arm}.npz'
                np.savez_compressed(stage/filename,reference=ref,rows=full,
                    requested_actions_20ms=np.asarray(endpoints),executed_actions_1ms=commands)
                r=dict(path=path,arm=arm,success=success,reason=env.reason,trace=filename,
                    audit=audit,corridor=corridor,geometry=env.tracker.snapshot(1.25),
                    complete_time_change_s=len(full)*.001-old['completion_time_s'] if success else None,
                    baseline_motion=complete_motion_metrics(ref,nominal),
                    motion=complete_motion_metrics(ref,full) if success else None)
                if len(ref)==3:
                    r['baseline_shape']=complete_corner_metrics(ref,nominal)
                    r['shape']=complete_corner_metrics(ref,full) if success else None
                    r['baseline_target_motion']=target_motion(ref,nominal)
                    r['target_motion']=target_motion(ref,full) if success else None
                records.append(r)
                write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,
                    costs=costs,diagnostic_only=True,learned_candidate_adopted=False,goal_complete=False))
                print('LINEAR_HOLD',arm,path,success,r['complete_time_change_s'],
                    None if not success or len(ref)!=3 else
                    {k:r['shape'][k]-r['baseline_shape'][k] for k in ('out_edge_outward_mm','full_physical_reverse_deg','full_normalized_energy')},flush=True)
        receipt(stage,dict(status='completed_linear_action_hold_probe',costs=costs,
            successes=sum(r['success'] for r in records),wall_s=time.perf_counter()-start,
            diagnostic_only=True,learned_candidate_adopted=False,goal_complete=False))
    except Exception as error:
        receipt(stage,dict(status='failed_linear_action_hold_probe',error=str(error),costs=costs,
            completed_paths=len(records),wall_s=time.perf_counter()-start,goal_complete=False))
        raise


if __name__=='__main__':main()
