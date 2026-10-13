"""Retain and re-evaluate every previously observed complete-program outcome."""
from pathlib import Path
import argparse
import run_boundary_cases as engine
from common import sha,write_json,utc

GROUP='generalization'
SOURCE=engine.BASE/'evidence/frozen_main_generalization_v6.json'
engine.__file__=__file__


def register():
    target=engine.OUT/(GROUP+'_registration.json');assert not target.exists()
    old=engine.read(SOURCE)
    reg=engine.read(engine.OUT/'scaled_registration.json')
    reg.update(created_utc=utc(),group=GROUP,inputs=[],original_registration_sha256=sha(SOURCE),
        cohort='All five previously observed synthetic programs; no independent holdout claim.',
        all_ten_original_planning_outcomes_preserved=True)
    reg['source_sha256'][str(Path(__file__))]=sha(__file__)
    for spec in old['grid']:
        record=engine.BASE/spec['record'];trace=engine.BASE/spec['trace'];data=engine.read(record)
        assert data['registration_sha256']==sha(SOURCE)
        assert sha(trace)==data['trace_sha256'] and data['runtime_version']=='v5_physical_time'
        reg['inputs'].append(dict(case=spec['case'],method='JNN',dt_s=spec['dt_s'],
            record=str(record),record_sha256=sha(record),trace=str(trace),trace_sha256=sha(trace)))
    assert len(reg['inputs'])==10
    write_json(target,reg)
    print('REGISTERED',sha(target),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run','worker'))
    p.add_argument('--group',default=GROUP,choices=(GROUP,));p.add_argument('--index',type=int)
    p.add_argument('--jobs',type=int,default=2);a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='worker':engine.worker(GROUP,a.index)
    else:engine.matrix(GROUP,a.jobs)
