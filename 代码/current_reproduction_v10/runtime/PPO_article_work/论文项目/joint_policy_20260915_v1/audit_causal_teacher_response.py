"""Audit saved full-trajectory teacher response without repeating simulation."""
import argparse
import numpy as np
from probe_coordinated_suffix import reconstruct
from probe_joint_target_coordinates import guide_angle
from canonical_gate_state import checkpoint_environment_factory
from joint_runtime import wrap
from localize_joint_shape_residual import characterize
from audit_saved_lateral_motion import lateral_motion
from learn_complete_joint_teachers import parameters_in_bounds
from study_io import BASE,write_json,sha
from search_complete_return import read


def decompose_decision_jumps(trace,localization):
    with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
    u,v=np.diff(ref,axis=0);sign=np.sign(u[0]*v[1]-u[1]*v[0])
    output=[]
    for jump in [j for j in localization['target_jumps'] if j['time_ms']%20==0][:2]:
        frame=jump['time_ms'];old,new=actions[frame//20-1],actions[frame//20]
        env=reconstruct(ref,rows,actions,frame-1,dict(saved_prefix_geometry_ticks=0),
            env_factory=checkpoint_environment_factory('gate_band_canonical_kernel'))
        beta_before=guide_angle(env,old)
        env.position=rows[frame-1,:2].copy();env.tracker.update(env.position)
        beta_old,beta_new=guide_angle(env,old),guide_angle(env,new)
        components=sign*np.degrees([wrap(beta_old-beta_before),wrap(beta_new-beta_old),new[0]-old[0]])
        total=sign*np.degrees(wrap(beta_new+new[0]-beta_before-old[0]))
        assert abs(abs(total)-jump['magnitude_deg'])<1e-8
        assert abs(components.sum()-total)<1e-8
        output.append(dict(time_ms=frame,state_advance_deg=float(components[0]),
            preview_change_deg=float(components[1]),steering_change_deg=float(components[2]),
            combined_target_change_deg=float(total),feed_change_mm_s=float(new[1]-old[1])))
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    args=parser.parse_args();stage=BASE/args.stage
    result=read(stage/'results.json');records=[]
    for teacher in result['teachers']:
        best=teacher['best'];before=teacher['baseline']
        trace=stage/best['trace']
        with np.load(trace) as z:
            basis=z['basis'].copy();start=int(z['start_tick'])
            residual=z['residual_labels'].copy()
        weights=basis@np.asarray(best['parameters']).reshape(-1,3)
        active=np.flatnonzero(np.any(weights,axis=1))
        record=dict(path=teacher['path'],coefficient_box_pass=parameters_in_bounds(best['parameters'],-.25,1.25),
            before=characterize(before['trace']),after=characterize(trace),
            lateral_before=lateral_motion(before['trace']),lateral_after=lateral_motion(trace),
            active_decisions=[dict(time_ms=start+20*int(j),weights=weights[j].tolist(),
                                   actual_latent_residual=residual[j].tolist()) for j in active],
            physical_and_completion_pass=bool(best['record']['success'] and
                best['record']['audit']['kinematic_pass'] and best['record']['audit']['rest_pass']))
        record['decision_jumps_before']=decompose_decision_jumps(before['trace'],record['before'])
        record['decision_jumps_after']=decompose_decision_jumps(trace,record['after'])
        records.append(record)
        print('SAVED_CAUSAL_RESPONSE',teacher['path'],
              'box',record['coefficient_box_pass'],
              'pre_reverse',record['before']['preapex_reverse_deg'],record['after']['preapex_reverse_deg'],
              'post_reverse',record['before']['postapex_reverse_deg'],record['after']['postapex_reverse_deg'],
              'jump',record['before']['target_jumps'][0]['magnitude_deg'],
              record['after']['target_jumps'][0]['magnitude_deg'],flush=True)
    write_json(stage/'saved_causal_response_audit.json',dict(records=records,new_rollouts=0,
        source_results_sha256=sha(stage/'results.json'),offline_teachers_only=True,
        evidence_boundary='Observed full-feedback responses of optimized joint teachers, not a unique causal decomposition.',
        final_shape_accepted=False))


if __name__=='__main__':main()
