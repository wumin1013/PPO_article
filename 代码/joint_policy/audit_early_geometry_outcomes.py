"""Explain complete joint-return outcomes from saved raw evidence only."""
import argparse
import numpy as np
from duration_joint_metrics import guard_names
from localize_joint_shape_residual import characterize
from study_io import BASE,write_json,sha
from search_complete_return import read


def constraint_names(paths):
    original=['entry','exit','reverse','energy','curve','apex_speed','time_loose',
              'cruise_loose','first','corridor','lateral']
    return ([f'{p}.{k}' for p in paths for k in original]+
            [f'{p}.{k}' for p in paths for k in guard_names()]+
            [f'{p}.{k}' for p in paths for k in ['intrinsic_peak','intrinsic_energy']])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args();stage=BASE/args.stage
    history=read(stage/'state_joint/results.json');candidates=history['candidates']
    baseline=candidates[0];paths=[r['path'] for r in baseline['records']]
    names=constraint_names(paths)
    def describe(c):
        values=c['augmented_measured_inequalities'];assert len(values)==len(names)
        violations=[dict(name=k,normalized_margin=v) for k,v in zip(names,values) if v < -1e-8]
        target=next(r for r in c['records'] if r['path']==104)
        return dict(tag=c['tag'],score=c['score'],violations=violations,
            significant_lateral_mm=target['shape']['reverse_lateral_significant_mm'],
            significant_lateral_duration_s=target['shape']['reverse_lateral_significant_duration_s'],
            all_negative_lateral_mm=target['shape']['reverse_lateral_total_mm'],
            reverse_deg=target['shape']['full_physical_reverse_deg'],
            exit_mm=target['shape']['out_edge_outward_mm'],
            trace_sha256=sha(target['trace']))
    rows=[describe(c) for c in candidates]
    beneficial=[r for r in rows if r['score']<rows[0]['score']-1.]
    nearest=sorted(beneficial,key=lambda r:(len(r['violations']),
        max([-v['normalized_margin'] for v in r['violations']],default=0.),r['score']))[:3]
    chosen=next(c for c in candidates if c['tag']==history['best'])
    selected_tags={baseline['tag'],chosen['tag']}|{r['tag'] for r in nearest}
    local=[dict(tag=c['tag'],path=104,**characterize(next(r['trace'] for r in c['records'] if r['path']==104)))
           for c in candidates if c['tag'] in selected_tags]
    out=dict(receipt_sha256=sha(stage/'receipt.json'),baseline=rows[0],
        selected=describe(chosen),nearest_beneficial=nearest,
        feasible_points=sum(not r['violations'] for r in rows),stored_points=len(rows),
        reused_action_identity_route_records=sum(len(c.get('exact_zero_support_paths',[])) for c in candidates),
        local_raw_shape=local,new_physical_runs=0,adopted=False,goal_complete=False,
        method='Read completed closed-loop parameter learning; not PPO or a new experiment.')
    write_json(stage/'joint_outcome_audit.json',out)
    print('JOINT_OUTCOME_AUDIT',args.stage,'selected',out['selected'],'nearest',nearest,flush=True)


if __name__=='__main__':main()
