"""Replay archived tracker prefixes, then test the importable terminal hook."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import sys
import numpy as np

OLD=Path('D:/PPO_article_work/JNN_dominance_20261011_v5')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(OLD/'src'))
from common import sha,write_json
from runtime_physical_time_v5 import install as install_base
from metrics_exact_index_v5 import metrics
from terminal_candidate import install as install_candidate


def main():
    target=HERE/'terminal_install_verification_v1.json'
    if target.exists():raise FileExistsError(target)
    records=[]
    for period_us in (1000,500):
        dt=period_us*1e-6
        install_base(dt=dt,tolerance=1.25)
        binding=install_candidate(vendor_dir=OLD/'literature/ruckig_vendor')
        assert install_candidate(vendor_dir=OLD/'literature/ruckig_vendor')==binding
        import joint_runtime
        from geometry_spec import Corridor,GateTracker
        path=OLD/'results/representative_scaled_v8/learned/main'/f'piecewise_curvature_coupon_scale10_JNN_dt{period_us}us.npz'
        with np.load(path) as bundle:
            reference,positions,rows=[bundle[key] for key in ('reference','positions','rows')]
        record=json.loads(path.with_suffix('.json').read_text('utf-8'))
        assert record['trace_sha256']==sha(path)
        start=int(np.flatnonzero(rows[:,12])[0])
        env=joint_runtime.JointRuntime.__new__(joint_runtime.JointRuntime)
        env.corridor=Corridor(reference,epsilon=1.25,closed=False)
        env.tracker=GateTracker(env.corridor,positions[0])
        for point in positions[1:start+1]:env.tracker.update(point)
        assert env.tracker.inside
        env.position=positions[start].copy();env.velocity=rows[start-1,2:4].copy()
        env.acceleration=rows[start-1,4:6].copy();env.brake=None;env.brake_index=0
        env._consider_terminal()
        assert env.brake is not None
        new_points=np.column_stack((env.brake['x_mm'],env.brake['y_mm']))
        joined=np.vstack((positions[:start+1],new_points[1:]))
        audited=metrics(reference,joined,dt=dt,tolerance=record['tolerance_mm'])
        assert audited['success']
        assert audited['terminal_rest_s']==record['terminal_rest_s']
        expected=np.load(HERE/f'scalar_scurve_diagnostic_v1/dt{period_us}us/selected_full_trace.npz')['positions']
        assert np.array_equal(joined,expected)
        # Force candidate geometry rejection on a separate hook invocation;
        # the already validated original cubic must remain byte-identical.
        import terminal_candidate
        native_check=terminal_candidate.geometry_pass
        terminal_candidate.geometry_pass=lambda *args:False
        env.brake=None;env.brake_index=0
        try:env._consider_terminal()
        finally:terminal_candidate.geometry_pass=native_check
        fallback=np.column_stack((env.brake['x_mm'],env.brake['y_mm']))
        assert np.max(np.abs(fallback-positions[start:]))<1e-8
        assert env.v9_terminal_diagnostics['selected']=='original_cubic'
        records.append(dict(dt_s=dt,binding=binding,source_trace_sha256=sha(path),
            complete_time_s=audited['complete_time_s'],terminal_rest_s=audited['terminal_rest_s'],
            full_audit=audited,exact_match_to_diagnostic=True,idempotent_install=True,
            geometry_rejection_fallback_verified=True,new_policy_rollout=False))
    write_json(target,dict(status='passed',formal_new_experiment=False,rows=records))
    print(json.dumps([dict(dt_s=row['dt_s'],complete_time_s=row['complete_time_s'],
        fallback=row['geometry_rejection_fallback_verified']) for row in records],indent=2))


if __name__=='__main__':main()
