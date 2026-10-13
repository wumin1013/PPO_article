"""Identify actual complete-route constraints limiting an early joint update."""
import argparse
import numpy as np
from study_io import BASE,write_json,sha
from search_complete_return import read
from multi_corner_raw_metrics import KEYS


def names(records):
    result=[]
    shape=['in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
           'full_normalized_energy','full_spatial_peak_curvature_mm_inv','apex_speed_mm_s',
           'complete_time_s','cruise99_s','first_turn_inside','corridor','reverse_lateral_total_mm']
    for r in records:
        if 'shape' in r:result.extend(dict(path=r['path'],metric=k) for k in shape)
    for r in records:
        if 'shape' in r:continue
        result.extend(dict(path=r['path'],metric=k) for k in ('complete_time_s','cruise99_s','corridor','success'))
        for window in r['multi_corner_windows']:
            result.extend(dict(path=r['path'],vertex=window['vertex'],metric=k)
                for k in [key for key,_ in KEYS]+['apex_speed_mm_s','first_turn_inside','curvature_02mm','energy_02mm','reverse_lateral_significant_duration_s'])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);args=parser.parse_args()
    stage=BASE/args.stage;data=read(stage/'state_joint/results.json');rows=data['candidates'];zero=rows[0]
    labels=names(zero['records']);assert len(labels)==len(zero['measured_inequalities'])
    output=[]
    for c in rows[1:]:
        constraints=np.asarray(c['measured_inequalities']);assert len(constraints)==len(labels)
        if not np.any(np.asarray(c['coefficients'])):continue
        violations=np.flatnonzero(constraints < -1e-8)
        output.append(dict(tag=c['tag'],score=c['score'],loss_change=c['score']-zero['score'],
            minimum=float(constraints.min()),violation_count=len(violations),
            squared_violation=float(np.square(np.minimum(constraints,0.)).sum()),
            worst=[dict(labels[i],normalized_slack=float(constraints[i])) for i in sorted(violations,key=lambda i:constraints[i])[:8]]))
    lower=[c for c in output if c['loss_change']<-.001]
    closest=sorted(lower,key=lambda c:c['squared_violation'])[:5]
    write_json(stage/'constraint_activity_audit.json',dict(baseline_score=zero['score'],vectors=len(rows),
        result_sha256=sha(stage/'state_joint/results.json'),
        lower_loss_vectors=len(lower),qualified_lower_loss_vectors=sum(c['minimum']>=-1e-8 for c in lower),
        closest_lower_loss=closest,lowest_loss=sorted(lower,key=lambda c:c['score'])[:3],
        closest_new_lower_loss=sorted([c for c in lower if not c['tag'].startswith('reused')],key=lambda c:c['squared_violation'])[:3],
        constraint_names=labels,physical_reruns=0,interpretation='Observed guards only; no estimator-bias or global infeasibility claim.'))
    for row in closest[:3]:print('CLOSEST_MEASURED',row,flush=True)


if __name__=='__main__':main()
