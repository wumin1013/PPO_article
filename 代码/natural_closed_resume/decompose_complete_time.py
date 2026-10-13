"""Exact recorded-time decomposition from unmodified complete trajectories."""
from pathlib import Path
import sys,json
import numpy as np
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import sha,write_json
R=OUT/'all34_dominance_research'
binding=json.loads((BASE/'paper_evidence_review/verified_results/paper_result_binding.json').read_text(encoding='utf-8'))
traces={'S2':Path(binding['primary_methods']['S2']['trace']),
    'Joint':R/'joint_fit_seed922202_full/full_trace.npz','Fixed':R/'fixed_fit_seed922202_full/full_trace.npz'}
records={}
for name,path in traces.items():
    z=np.load(path);rows=z['rows'];speed=np.linalg.norm(rows[:,2:4],axis=1);dt=.001
    length=float(speed.sum()*dt);deficit=float((100-speed).sum()*dt)
    records[name]=dict(complete_recorded_time_s=len(rows)*dt,travelled_distance_mm=length,
        signed_speed_deficit_mm=deficit,at100_travel_time_s=length/100,deficit_time_s=deficit/100,
        identity_error_s=float(len(rows)*dt-(length+deficit)/100),trace_sha256=sha(path))
differences={}
for name in ['S2','Fixed']:
    differences[name]=dict(total_gain_s=records[name]['complete_recorded_time_s']-records['Joint']['complete_recorded_time_s'],
        distance_contribution_s=records[name]['at100_travel_time_s']-records['Joint']['at100_travel_time_s'],
        speed_deficit_contribution_s=records[name]['deficit_time_s']-records['Joint']['deficit_time_s'])
report=dict(goal_complete=False,kind='read_only_recorded_complete_time_identity',records=records,joint_gain_decomposition=differences,
    equation='T = L_executed / 100 + integral(100 - |v|)dt / 100; startup, full brake and rest included.',
    caveat='This is a decomposition of recorded1ms traces; it does not establish resolution-stable superiority or all34 dominance.')
write_json(R/'complete_time_decomposition.json',report)
print(json.dumps(report,indent=2))
