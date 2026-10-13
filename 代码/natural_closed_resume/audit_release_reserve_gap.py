"""Check the extra recursive release certificate on existing valid trajectories."""
from pathlib import Path
import sys,json,csv
import numpy as np
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from joint_runtime import projector
from study_io import write_json,sha

def main():
    root=OUT/'all34_dominance_research';evidence=BASE/'paper_evidence_review/verified_results'
    table=list(csv.DictReader((evidence/'all34_corner_statistics.csv').open(encoding='utf-8-sig')))
    sources={'S2':BASE/'paper_evidence_review/natural_closed_case/s2_complete/trace.npz',
             'Joint':OUT/'joint_actual_context_feedback_training/exact_continuation/full_trace.npz'}
    records=[]
    for method,path in sources.items():
        rows=np.load(path)['rows']
        for r in table:
            if r['method']!=method:continue
            rr=rows[int(r['first_index']):int(r['last_index'])+1]
            ends=np.array([projector.release_endpoint(q[2:4],q[4:6]) for q in rr])
            residual=np.linalg.norm(ends,axis=1)-100.
            rec=dict(method=method,corner=r['id'],angle=float(r['included_angle_deg']),samples=len(rr),
                maximum_release_endpoint_excess_mm_s=float(residual.max()),
                samples_failing_release_reserve=int((residual>1e-10).sum()),
                actual_vmax=float(np.linalg.norm(rr[:,2:4],axis=1).max()),
                actual_amax=float(np.linalg.norm(rr[:,4:6],axis=1).max()),
                actual_jmax=float(np.linalg.norm(rr[:,6:8],axis=1).max()))
            records.append(rec)
            if r['id'] in ('corner_00','corner_01','corner_06','corner_22','corner_33'):
                print('RESERVE_GAP',rec,flush=True)
    write_json(root/'release_reserve_gap.json',dict(kind='existing_trace_certificate_audit',physics_ticks=0,training_updates=0,
        records=records,source_sha256={str(p):sha(p) for p in sources.values()},
        projector_sha256=sha(Path(projector.__file__)),
        interpretation='The straight-direction acceleration-release certificate is sufficient but not necessary for the physical vector constraints. S2 is not required to satisfy this learned-runtime-specific certificate. No baseline or limit is changed by this audit.',
        next_hypothesis='If policy-only refinement stalls, investigate a rigorously checked turn-aware backup shared by Joint and Fixed; never simply remove recursive safety checks.'))

if __name__=='__main__':main()
