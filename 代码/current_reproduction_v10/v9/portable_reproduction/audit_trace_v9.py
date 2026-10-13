"""Recompute one full command trace, including every interval and ordered gate."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from pathlib import Path
import sys
import time
from archive_tools import digest, read, write_new
from portable_paths import install_guard, resolve

CORE = ('success','complete_time_s','elapsed_prefix_s','samples','dt_s','tolerance_mm',
        'limits','vector_peaks','vector_limit_ratios','kinematic_pass','rest_pass',
        'terminal_rest_s','geometry','max_polyline_deviation_mm',
        'rms_polyline_deviation_mm','executed_length_mm')


def check_equal(actual, expected, name='metrics'):
    """Boolean/integer/status fields exact; numerical quantities at audit precision."""
    import numpy as np
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError('Missing '+name+'.'+key)
            check_equal(actual[key],value,name+'.'+key)
    elif isinstance(expected, (list,tuple)):
        if len(actual) != len(expected):
            raise AssertionError('Length differs: '+name)
        for i, value in enumerate(expected):
            check_equal(actual[i],value,name+'['+str(i)+']')
    elif isinstance(expected, float):
        if not np.isclose(actual,expected,rtol=1e-9,atol=2e-7):
            raise AssertionError(f'{name}: {actual!r} != {expected!r}')
    elif actual != expected:
        raise AssertionError(f'{name}: {actual!r} != {expected!r}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    guard=install_guard(workspace)
    started=time.perf_counter()
    import numpy as np
    spec=read(args.spec)
    row=spec['row']; expected=row['metrics']
    bindings={}
    for key in ('record','trace'):
        if row.get(key):
            path=resolve(row[key],workspace)
            if digest(path) != row[key+'_sha256']:
                raise AssertionError('Source checksum differs: '+key)
            bindings[key]=path
    if 'record' not in bindings:
        raise AssertionError('Every registered cell requires its retained record.')
    record=read(bindings['record'])
    if 'trace' not in bindings:
        assert expected['success'] is False and expected['complete_time_s'] is None
        result=dict(id=spec['id'],status='retained_planning_failure',trace_present=False,
            expected_metrics=expected,record_sha256=row['record_sha256'],
            audit_scope='No command trace exists; retained failed-planning record and null completion time verified.')
    else:
        with np.load(bindings['trace'],allow_pickle=False) as data:
            positions=data['positions'].copy(); reference=data['reference'].copy()
        definition=spec.get('definition')
        if definition:
            for key in ('reference','geometry'):
                path=resolve(definition[key],workspace)
                assert digest(path)==definition[key+'_sha256']
            source_reference=np.loadtxt(resolve(definition['reference'],workspace),delimiter=',',skiprows=1)
            assert np.array_equal(reference,source_reference)
        sys.path.insert(0,str(workspace/'src'))
        from runtime_no_target_override_v4b import install
        tolerance=expected.get('tolerance_mm',record.get('tolerance_mm',1.25))
        install(dt=row['dt_s'],tolerance=tolerance)
        from metrics_exact_index_v5 import metrics
        actual=metrics(reference,positions,dt=row['dt_s'],tolerance=tolerance,
            limits=expected.get('limits',[100.,2000.,62500.]),
            declared_success=expected['success'],reason=expected.get('reason','').split(';')[0],geometry=True)
        fields={key:expected[key] for key in CORE if key in expected}
        check_equal(actual,fields)
        result=dict(id=spec['id'],status='verified',trace_present=True,metrics=actual,
            recorded_reason=expected.get('reason'),trace_sha256=row['trace_sha256'],
            record_sha256=row['record_sha256'],compared_core_fields=list(fields),
            every_command_interval_audited=True,all_ordered_gates_audited=True,
            command_samples=len(positions),audit_scope='All command positions, planar backward differences, continuous interval corridor and ordered gates, initial and final rest.')
    result.update(wall_s=time.perf_counter()-started,guard=guard)
    write_new(args.output,result)
    print(spec['id'],result['status'],f"{result['wall_s']:.2f}s",flush=True)


if __name__=='__main__':
    main()
