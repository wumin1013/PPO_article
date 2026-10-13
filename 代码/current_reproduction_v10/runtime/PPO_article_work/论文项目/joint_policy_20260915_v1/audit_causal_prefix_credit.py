"""Compact measured shape changes for causal-prefix responses and learned proposals."""
import numpy as np
from study_io import BASE,write_json,sha
from search_complete_return import read


def totals(records):
    corners=[r for r in records if 'shape' in r]
    return dict(reverse_lateral_mm=sum(r['shape']['reverse_lateral_total_mm'] for r in corners),
        significant_reverse_duration_s=sum(r['shape']['reverse_lateral_significant_duration_s'] for r in corners),
        reverse_deg=sum(r['shape']['full_physical_reverse_deg'] for r in corners),
        exit_mm=sum(r['shape']['out_edge_outward_mm'] for r in corners),
        energy=sum(r['shape']['full_normalized_energy'] for r in corners),
        cruise99_s=sum(r['motion']['cruise_ge99_s'] for r in records),
        complete_time_s=sum(r['complete_time_s'] for r in records))


def main():
    stage=BASE/'canonical_gate_band_causal_prefix_joint_compensation_20260920'
    history=read(stage/'state_joint/results.json')['candidates'];zero=history[0];base=totals(zero['records'])
    rows=[]
    for c in history:
        if not c['physical_pass']:continue
        v=np.maximum(-np.array(c['augmented_measured_inequalities']),0.)
        t=totals(c['records'])
        rows.append(dict(tag=c['tag'],training_score=c['training_score'],total_violation=float(v.sum()),
            max_violation=float(v.max()),all_guard_pass=bool(v.max()<=1e-8),
            change={k:t[k]-base[k] for k in base},
            multi31_windows=[dict(vertex=w['vertex'],reverse_lateral_mm=w['shape']['reverse_lateral_total_mm'],
                energy=w['shape']['full_normalized_energy'],exit_mm=w['shape']['out_edge_outward_mm'])
                for r in c['records'] if r['path']==31 for w in r['multi_corner_windows']]))
    best_reverse=min(rows,key=lambda r:r['change']['reverse_lateral_mm'])
    best_violation=min(rows,key=lambda r:(r['total_violation'],r['max_violation']))
    report=dict(results_sha256=sha(stage/'state_joint/results.json'),candidates=len(history),
        physical_passes=sum(c['physical_pass'] for c in history),new_physical_runs=0,
        seed_totals=base,best_reverse=best_reverse,best_violation=best_violation,rows=rows,
        scope='Totals describe affected isolated corners; multicorner31 windows are retained separately. '
        'Changes are relative to the unadopted wide seed. All final guards remain relative to protected ded2.')
    write_json(stage/'prefix_credit_audit.json',report)
    print('PREFIX_AUDIT_COUNT',len(history),'PHYSICAL',report['physical_passes'],flush=True)
    print('PREFIX_BEST_REVERSE',best_reverse,flush=True)
    print('PREFIX_BEST_VIOLATION',best_violation,flush=True)


if __name__=='__main__':main()
