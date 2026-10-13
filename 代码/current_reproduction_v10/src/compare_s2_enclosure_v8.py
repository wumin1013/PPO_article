"""Versioned coordinate-frame repair of S2 bounds, with separate validation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import compare_traditional_g01_v7 as validation_driver
import compare_traditional_representative_v7 as base
from common import ROOT, sha, utc, write_json
from g01_ordered_simplification_v7 import install as install_geometry, simplify
from s2_local_enclosure_v8 import install as install_enclosure

OUTPUT = ROOT / 'results/representative_scaled_v8/s2_local_enclosure_v8'


def configure(stage):
    validation_driver.configure('validation' if stage=='validation' else 'formal')
    base.OUT = OUTPUT / stage
    base.REG = base.OUT / 'registration.json'
    base.METHODS = ('S2', 'S2R')
    if stage=='validation':
        base.SOURCE = base.OUT / 'validation_inputs.json'
    else:
        base.SOURCE = ROOT / 'evidence/scaled_representative_v8_registration.json'
        base.GLOBAL = base.SOURCE
        base.CASES = ('piecewise_curvature_coupon_scale10', 'asymmetric_neck_contour_scale10')


def register(stage):
    if base.REG.exists():
        return base.verify()
    if stage=='formal':
        receipt=base.read(OUTPUT / 'validation/summary.json')
        if receipt['successful_cells']!=8 or not receipt['all_registered_cells_present']:
            raise ValueError('Require all eight original-validation S2 cells')
        r=base.register()
    else:
        r=validation_driver.register('validation')
    r.update(study_id='s2_local_bernstein_enclosure_v8_'+stage,
        numerical_change='Same curvature and curvature-gradient Bernstein numerical enclosure in an orthonormal chord coordinate frame; no curve or threshold change.',
        unchanged='Positions/control points, speed-family/Ruckig scheduler, bounds1e-3 convergence tolerance, internal.999A/J, original-reference command acceptance.',
        prior_attempts='results/representative_scaled_v8/traditional',
        source_coordinate_invariance='Euclidean curvature and arc-length curvature derivative are rigid-rotation invariant; interval hull width can change with coordinates.',
        geometry_version='v7 fixeddelta0.1effective epsilon ordered G01 merging')
    for name in ('compare_s2_enclosure_v8.py','s2_local_enclosure_v8.py','g01_ordered_simplification_v7.py'):
        r['source_sha256'][name]=sha(ROOT/'src'/name)
    if stage=='formal':
        r['validation_receipt_sha256']=sha(OUTPUT/'validation/summary.json')
        for case in r['cases']:
            ref=np_load(ROOT/case['reference'])
            reduced,certificate=simplify(ref,case['evaluation_tolerance_mm'])
            write_json(base.OUT/(case['id']+'_geometry_certificate.json'),certificate)
    write_json(base.REG,r)
    return base.verify()


def np_load(path):
    import numpy as np
    return np.loadtxt(path,delimiter=',',skiprows=1)


def summarize():
    registration=base.verify()
    records=[]
    for cell in registration['grid']:
        path=base.OUT/(cell['run_id']+'.json')
        if path.exists():
            row=base.read(path)
            if row['registration_sha256']!=sha(base.REG):
                raise ValueError('Attempt registration differs')
            records.append({key:row.get(key) for key in ('run_id','case','method','dt_s','success',
                'complete_time_s','reason','error','wall_s','cpu_s','trace','trace_sha256',
                'vector_limit_ratios','max_polyline_deviation_mm','executed_length_mm','geometry','terminal_rest_s')})
    value=dict(created_utc=utc(),registration_sha256=sha(base.REG),expected_cells=len(registration['grid']),
        attempted_cells=len(records),successful_cells=sum(r['success'] for r in records),
        all_registered_cells_present=len(records)==len(registration['grid']),records=records)
    write_json(base.OUT/'summary.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('validation','formal'),required=True)
    p.add_argument('--register',action='store_true');p.add_argument('--all',action='store_true')
    p.add_argument('--case');p.add_argument('--method',choices=('S2','S2R'));p.add_argument('--dt',type=float,choices=(.001,.0005))
    args=p.parse_args();configure(args.stage)
    if args.register:
        r=register(args.stage)
        print(json.dumps(dict(registration=str(base.REG),sha256=sha(base.REG),cells=len(r['grid']))))
    elif args.all:
        base.verify()
        for case in base.CASES:
            for method in base.METHODS:
                for dt in base.PERIODS:
                    name=base.run_id(case,method,dt)
                    if (base.OUT/(name+'.json')).exists():continue
                    with (base.OUT/(name+'.log')).open('x',encoding='utf-8') as log:
                        result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--stage',args.stage,
                            '--case',case,'--method',method,'--dt',str(dt)],stdout=log,stderr=subprocess.STDOUT,check=False)
                    if result.returncode:raise RuntimeError('Preserved worker exception: '+name)
                    r=summarize();print(json.dumps(dict(attempted=r['attempted_cells'],successful=r['successful_cells'])),flush=True)
    elif args.case and args.method and args.dt:
        base.verify();install_geometry();install_enclosure()
        base.run(args.case,args.method,args.dt);summarize()
    else:p.error('Choose register, all or case/method/dt')
