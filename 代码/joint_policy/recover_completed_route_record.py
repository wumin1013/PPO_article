"""Recover a proven completed route saved before an interrupted batch returned."""
import argparse
import numpy as np
from learned_action_pulse_policy import environment_factory
from analyze_nonlinear_joint_response import measure
from validate_actions import difference_audit
from learn_retained_joint_growth import smooth_cruise
from study_io import BASE,write_json,sha
from search_complete_return import read


def recover(trace,path,curved_path):
    with np.load(trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
    env=environment_factory(ref)
    for xy in rows[:,:2]:env.tracker.update(xy)
    audit=difference_audit(ref[0],rows)
    # The writer saves only after env.done. Rule out every other terminal reason
    # from the original runtime, without interpreting a failure prefix as success.
    assert len(rows)<env.max_ticks and rows[-1,12]==1
    assert env.tracker.inside and env.tracker.snapshot(1.25)['geometry_complete']
    assert audit['kinematic_pass'] and audit['rest_pass']
    assert np.max(np.linalg.norm(rows[-21:,2:4],axis=1))<=.001
    assert np.max(np.linalg.norm(rows[-21:,4:6],axis=1))<=.01
    env.success=True;env.reason='success'
    r=measure(ref,rows,env);assert r['success']
    r.update(path=path,trace=str(trace),curved_path=curved_path,smooth_cruise99_s=smooth_cruise(rows))
    return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True);args=parser.parse_args()
    stage=BASE/args.stage;history=read(stage/'state_joint/results.json')['candidates']
    candidate=next(c for c in history if c['tag'].startswith('late_response_'))
    checks=[]
    for old in candidate['records']:
        r=recover(old['trace'],old['path'],old['curved_path'])
        assert all(r[k]==old[k] for k in r),('Recovered measurement mismatch',old['path'])
        checks.append(dict(path=old['path'],trace_sha256=sha(old['trace'])))
    write_json(stage/'completed_route_recovery_check.json',dict(all_fields_exact=True,
        checked_records=checks,new_physical_runs=0,source_code_sha256=sha(__file__)))
    print('RECOVERED_ROUTE_RECORD_IDENTITY',len(checks),flush=True)


if __name__=='__main__':main()
