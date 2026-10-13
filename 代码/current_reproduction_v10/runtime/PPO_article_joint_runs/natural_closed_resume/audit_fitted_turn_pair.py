"""Read-only audit of recorded fitted laps and projection reversals."""
from pathlib import Path
import sys,json
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from natural_case_metrics import ordered_projection
from study_io import write_json,sha
R=OUT/'all34_dominance_research'
case=BASE/'paper_evidence_review/natural_closed_case'
ref=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
g=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
prior=json.loads((R/'first_complete_turn_aware_pair.json').read_text(encoding='utf-8'))
data={}
for arm in ('joint','fixed'):
    folder=R/(arm+'_fit_seed922202_full')
    result=json.loads((folder/'full_result.json').read_text(encoding='utf-8'))
    m=json.loads((folder/'full_metrics.json').read_text(encoding='utf-8'))
    rows=np.load(folder/'full_trace.npz')['rows'];p=ordered_projection(ref,rows[:,:2]);ids=np.flatnonzero(np.diff(p)<-1e-7)
    reverse=[]
    for i in ids:
        c=min(g['sharp_corners'],key=lambda c:abs(c['reference_s_mm']-p[i]))
        reverse.append(dict(t_s=float((i+1)*.001),s_before=float(p[i]),s_after=float(p[i+1]),delta_mm=float(p[i+1]-p[i]),
            xy=rows[i:i+2,:2].tolist(),velocity=rows[i:i+2,2:4].tolist(),nearest_corner=c['id'],
            distance_to_corner_mm=float(p[i]-c['reference_s_mm'])))
    data[arm]=dict(success=result['success'],complete_time_s=result['complete_time_s'],
        recorded_time_s=result['elapsed_prefix_or_complete_s'],cruise99_s=m['cruise']['99']['time_s'],
        corner_deficit_sum_mm=m['corner_total_speed_deficit_mm'],projection_reversals=reverse,
        corners=m['corners'],source_trace_sha256=sha(folder/'full_trace.npz'))
records=[]
for j,f,p in zip(data['joint']['corners'],data['fixed']['corners'],prior['all34']):
    baseline=min(p['s2'],p['fixed'])
    record=dict(id=j['id'],angle=j['included_angle_deg'],joint=j['speed_deficit_mm'],
        fixed_successful_prior=p['fixed'],s2=p['s2'],fixed_unaccepted_probe=f['speed_deficit_mm'],
        gap_to_successful_baselines=j['speed_deficit_mm']-baseline,
        gap_to_unaccepted_probe=j['speed_deficit_mm']-f['speed_deficit_mm'],
        first_turn_inward=j['first_turn_30mm_audit_inside'])
    records.append(record)
report=dict(goal_complete=False,kind='recorded_full_lap_audit_no_new_simulation',runs=data,all34=records,
    failed_against_successful_baselines=[r['id'] for r in records if r['gap_to_successful_baselines']>1e-7],
    warning='Fixed seed922202 is unaccepted under existing projection check. Its local deficits remain diagnostic evidence, not a valid complete-time baseline.')
write_json(R/'fitted_turn_pair_audit.json',report)
print(json.dumps({k:{kk:v for kk,v in d.items() if kk not in ('corners',)} for k,d in data.items()},indent=2))
print(json.dumps(records,indent=2))
