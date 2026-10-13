"""Diagnose saved executed positions; no new physics or policy updates."""
import json
import numpy as np
from evaluate_historical_dense import ROOT,CASE,write,sha
from natural_case_audit import analytic_distance


def main():
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    paths=[ROOT/'direct14_farvector_local_feedback_fit_round21/eval_8000_dense.npz',
           ROOT/'direct14_farvector_local_feedback_fit_round21_dense_half_ms/trace.npz',
           ROOT/'direct14_two_action_initialization_round27/d_3/dense_12000.npz',
           ROOT/'direct14_two_action_initializer_dense_half_round32/trace.npz']
    paths+=sorted((ROOT/'direct14_main_mixed_heads_round34').glob('eval_*_dense_*.npz'))
    records=[]
    for path in paths:
        with np.load(path) as data:
            rows=data['rows'].copy();observations=data['observations'].copy()
            dt=float(data['dt_s']) if 'dt_s' in data.files else .001
        distance=analytic_distance(rows[:,:2],geometry);ids=np.flatnonzero(distance>1.2)
        records.append(dict(trace=str(path),sha256=sha(path),dt_s=dt,elapsed_prefix_or_complete_s=len(rows)*dt,
            maximum_sampled_distance_mm=float(distance.max()),time_above_1p2mm_s=float(len(ids)*dt),
            first_time_above_1p2mm_s=float((ids[0]+1)*dt) if len(ids) else None,
            time_above_1p24mm_s=float((distance>1.24).sum()*dt),
            maximum_abs_acceleration_observation=float(abs(observations[:,3:5]).max())))
    report=dict(scope='Diagnostic of saved physical samples only;not a new corridor limit or a complete-command certificate.',
                geometry_sha256=sha(CASE/'geometry.json'),records=records,new_physics_ticks=0,new_ppo_updates=0)
    write(ROOT/'dense_boundary_margin_round34.json',report)
    for item in records:print(json.dumps(item))


if __name__=='__main__':main()
