"""Candidate-bound, unsmoothed all-corner turn and exit-heading evidence."""
from pathlib import Path
import argparse,json,sys
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');args=ap.parse_args()
    folder=Path(args.candidate);stage=folder/'actual_turn_geometry';stage.mkdir(exist_ok=False)
    result=json.loads((folder/'full_result.json').read_text(encoding='utf-8'))
    assert result['success']
    z=np.load(folder/'full_trace.npz');rows=z['rows'];ref=z['reference'];dt=.001
    times=(np.arange(len(rows))+1)*dt
    geo_path=BASE/'paper_evidence_review/natural_closed_case/geometry.json'
    geometry=json.loads(geo_path.read_text(encoding='utf-8'))
    metrics=json.loads((folder/'full_metrics.json').read_text(encoding='utf-8'))
    records=[]
    for c,m in zip(geometry['sharp_corners'],metrics['corners']):
        assert c['id']==m['id']
        vertex=ref[c['reference_vertex_index']]
        incoming=vertex-ref[c['reference_vertex_index']-1];incoming/=np.linalg.norm(incoming)
        outgoing=ref[c['reference_vertex_index']+1]-vertex;outgoing/=np.linalg.norm(outgoing)
        handedness=np.sign(c['signed_heading_change_deg'])
        normal=handedness*np.array([-incoming[1],incoming[0]])
        tc=m['crossing_time_s'];indices=np.flatnonzero((times>=tc-.6)&(times<=tc+.8))
        rr=rows[indices];heading=np.rad2deg(np.arctan2(rr[:,2:4]@normal,rr[:,2:4]@incoming))
        candidate_indices=np.flatnonzero((np.convolve((abs(heading)>=1.).astype(int),np.ones(10,dtype=int),mode='valid')==10))
        first=next((int(k) for k in candidate_indices if times[indices[k]]<=tc),None)
        rec=dict(corner=c['id'],interior_angle_deg=c['interior_angle_deg'],crossing_time_s=tc,
            first_sustained_turn_inward=bool(np.mean(heading[first:first+10])>0) if first is not None else None,
            first_sustained_turn_distance_before_vertex_mm=float(-(rr[first,:2]-vertex)@incoming) if first is not None else None,
            first_sustained_turn_time_before_gate_s=float(tc-times[indices[first]]) if first is not None else None,
            maximum_reverse_heading_deg=float(max(0.,-heading.min())),exit_heading_error_deg={})
        after=times[indices]>=tc
        along=(rr[:,:2]-vertex)@outgoing
        normal_out=np.array([-outgoing[1],outgoing[0]])
        for distance in (10,20,50):
            selected=np.flatnonzero(after & (along>=distance))
            rec['exit_heading_error_deg'][str(distance)]=float(abs(np.rad2deg(np.arctan2(rr[selected[0],2:4]@normal_out,rr[selected[0],2:4]@outgoing)))) if len(selected) else None
        records.append(rec)
    report=dict(goal_complete=False,kind='raw_candidate_bound_turn_geometry',records=records,
        sustained_turn_threshold_deg=1.,sustained_duration_s=.010,
        all34_first_sustained_turn_inward=all(r['first_sustained_turn_inward'] is True for r in records),
        maximum_exit_heading_error_deg={str(d):max(r['exit_heading_error_deg'][str(d)] for r in records if r['exit_heading_error_deg'][str(d)] is not None) for d in (10,20,50)},
        geometry_note='Positive local heading is toward inside. Exit samples are explicitly after witnessed corner-gate crossing; incoming points of acute corners cannot masquerade as outgoing positions.',
        scope='Diagnostic actual motion, no smoothing or resampling; finite recovery errors do not prove exact analytic tangency.',
        source_sha256={str(p):sha(p) for p in [Path(__file__),geo_path,folder/'full_result.json',folder/'full_trace.npz',folder/'full_metrics.json']})
    write_json(stage/'audit.json',report)
    print('ALL34_TURN_GEOMETRY',report['all34_first_sustained_turn_inward'],report['maximum_exit_heading_error_deg'],flush=True)


if __name__=='__main__':main()
