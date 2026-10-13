"""One-factor SCMO sensitivity on the two development validation paths only.

Tighter tolerances are numerical stress tests at 40% and 20% of 1.25 mm;
they are not attributed to an unprovided part tolerance or machining allowance.
"""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from baselines.motion_overlap import plan
from metrics import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default=str(ROOT/'results/sensitivity/scmo'))
    args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    conditions=[('nominal',1.25,(100.,2000.,62500.)),
                ('tolerance_0p5',.5,(100.,2000.,62500.)),
                ('tolerance_0p25',.25,(100.,2000.,62500.)),
                ('velocity_80',1.25,(80.,2000.,62500.)),
                ('velocity_120',1.25,(120.,2000.,62500.)),
                ('acceleration_1600',1.25,(100.,1600.,62500.)),
                ('acceleration_2400',1.25,(100.,2400.,62500.)),
                ('jerk_50000',1.25,(100.,2000.,50000.)),
                ('jerk_75000',1.25,(100.,2000.,75000.))]
    record_list=[]
    for path_id in ['validation_zigzag','validation_trapezoid']:
        pathfile=ROOT/'data'/(path_id+'.csv')
        reference=np.loadtxt(pathfile,delimiter=',',skiprows=1)
        for name,tolerance,limits in conditions:
            basename=path_id+'__'+name
            destination=out/(basename+'.json')
            if destination.exists():
                raise FileExistsError(str(destination)+' already exists; use a separate --output-dir for reruns')
            record=dict(method='independent_scurve_motion_overlap',path_id=path_id,
                        partition='validation',condition=name,tolerance_mm=tolerance,
                        vector_limits=dict(zip(['v','a','j'],limits)),dt_s=.001,
                        reference_sha256=hashlib.sha256(pathfile.read_bytes()).hexdigest(),
                        controller_reconfigured=True,learning_or_retraining=False,
                        tolerance_basis='0.5/0.25 mm are 40%/20% of nominal numerical stress tests; no part/process allowance supplied')
            try:
                start=perf_counter()
                r=plan(reference,tolerance=tolerance,limits=record['vector_limits'],dt=.001)
                record['planner_call_wall_s']=perf_counter()-start
                start=perf_counter()
                record['evaluation']=metrics(reference,r['positions'],dt=.001,tolerance=tolerance,limits=limits)
                record['evaluation_wall_s']=perf_counter()-start
                record['planner_metrics']=r['metrics']
                record['status']='success' if record['evaluation']['success'] else 'failed_evaluation'
                np.savez_compressed(out/(basename+'.npz'),reference=reference,
                    positions=r['positions'],time=r['time'],velocities=r['velocities'],
                    accelerations=r['accelerations'],jerk=r['jerk'],
                    corner_witness_times=r['corner_witness_times'])
            except Exception as error:
                record.update(status='exception',error=repr(error),traceback=traceback.format_exc())
            destination.write_text(json.dumps(record,indent=2),encoding='utf-8')
            record_list.append(record)
            print(json.dumps(dict(path_id=path_id,condition=name,status=record['status'],
                time_s=record.get('evaluation',{}).get('complete_time_s'),
                reason=record.get('evaluation',{}).get('reason',record.get('error')))),flush=True)
    summary=dict(scope='predeclared one-factor deterministic sensitivity; validation paths only',
        runs=len(record_list),successful=sum(r['status']=='success' for r in record_list),
        failed=sum(r['status']!='success' for r in record_list),
        method_version='1.1-finite-difference-rest',
        no_final_test_paths_accessed=True,tolerance_basis=record_list[0]['tolerance_basis'],
        conditions=[dict(name=n,tolerance_mm=t,limits=list(l)) for n,t,l in conditions],
        results=[dict(path_id=r['path_id'],condition=r['condition'],status=r['status'],
            complete_time_s=r.get('evaluation',{}).get('complete_time_s'),
            max_polyline_deviation_mm=r.get('evaluation',{}).get('max_polyline_deviation_mm'),
            vector_peaks=r.get('evaluation',{}).get('vector_peaks'),
            terminal_rest_s=r.get('evaluation',{}).get('terminal_rest_s'),
            planner_call_wall_s=r.get('planner_call_wall_s'),
            reason=r.get('evaluation',{}).get('reason',r.get('error'))) for r in record_list])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(runs=summary['runs'],successful=summary['successful'],failed=summary['failed'])),flush=True)


if __name__=='__main__': main()
