"""Inspect physical-state information omitted by the current three-feature head."""
import argparse
import math
import numpy as np
from canonical_gate_state import checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_joint_target_coordinates import guide_angle
from localize_joint_shape_residual import characterize
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args();stage=BASE/args.stage
    baseline=read(stage/'baseline.json')['baseline']['records']
    paths=read(stage/'registration.json')['budget']['optimization_paths']
    queries=[];rates=[];phases=[]
    for r in baseline:
        with np.load(r['trace']) as z:ref,rows,actions,obs=[z[k].copy() for k in ('reference','rows','actions','observations')]
        observed_speed=100.*np.linalg.norm(obs[:,2:4],axis=1)
        phase=30.*obs[:,11]/(observed_speed+5.)
        phases.extend(abs(phase[(observed_speed>5.)&(abs(phase)>1e-4)]).tolist())
        decision_rows=rows[np.arange(19,len(actions)*20-1,20)]
        speed=np.linalg.norm(decision_rows[:,2:4],axis=1)
        omega=(decision_rows[:,2]*decision_rows[:,5]-decision_rows[:,3]*decision_rows[:,4])/(speed**2+25.)
        rates.extend(abs(omega[speed>5.]).tolist())
        if r['path'] not in paths:continue
        localized=characterize(r['trace'])
        frame=next(j['time_ms'] for j in localized['target_jumps'] if j['time_ms']%20==0)
        env=reconstruct(ref,rows,actions,frame,dict(saved_prefix_geometry_ticks=0),
                        env_factory=checkpoint_environment_factory('gate_band_canonical_kernel'))
        held=actions[frame//20];target=guide_angle(env,held)+held[0]
        jump=math.sin(target-guide_angle(env,env.held)-env.held[0])
        gap=math.sin(target-env.heading);speed=np.linalg.norm(env.velocity)
        lag=(math.cos(env.heading)*env.velocity[1]-math.sin(env.heading)*env.velocity[0])/max(speed,5.)
        inputs=np.array([jump,gap,lag]);x=np.tanh(inputs/.5)
        rate=(env.velocity[0]*env.acceleration[1]-env.velocity[1]*env.acceleration[0])/(speed**2+25.)
        tangent_acc=float(env.acceleration@env.velocity/max(speed,5.))
        queries.append(dict(path=r['path'],frame=frame,inputs=inputs.tolist(),even_features=(x*x).tolist(),
            speed_mm_s=float(speed),signed_turn_rate_rad_s=float(rate),
            tangent_acceleration_mm_s2=tangent_acc,actual_reverse_deg=r['shape']['full_physical_reverse_deg'],
            actual_exit_mm=r['shape']['out_edge_outward_mm']))
    comparisons=[]
    for a in queries:
        if a['path']==14:continue
        b=next(q for q in queries if q['path']==14)
        comparisons.append(dict(path=a['path'],reference_path=14,
            even_feature_distance=float(np.linalg.norm(np.asarray(a['even_features'])-b['even_features'])),
            turn_rate_difference_rad_s=abs(a['signed_turn_rate_rad_s']-b['signed_turn_rate_rad_s']),
            speed_difference_mm_s=abs(a['speed_mm_s']-b['speed_mm_s'])))
    audit=dict(queries=queries,comparisons=comparisons,new_rollouts=0,
        baseline_sha256=sha(stage/'baseline.json'),turn_rate_90pct_rad_s=float(np.quantile(rates,.9)),
        turn_rate_99pct_rad_s=float(np.quantile(rates,.99)),
        gate_forward_time_median_s=float(np.median(phases)),
        evidence_boundary='Information audit at saved decision states; feature similarity alone is not proof of state aliasing or causal sufficiency.')
    write_json(stage/'innovation_feedback_state_audit.json',audit)
    print('INNOVATION_FEEDBACK_STATE',audit,flush=True)


if __name__=='__main__':main()
