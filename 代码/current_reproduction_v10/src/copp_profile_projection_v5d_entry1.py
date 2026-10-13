"""Reusable controller for the frozen v5d command-projection rule.

Numerical scaling, limits, periods, reserve and budget are imported from the
immutable v5d experiment module. This controller supports new profiles without
changing that historical module or any of its registered evidence.
"""
from pathlib import Path
import time
import numpy as np
import copp_command_projection_v5d as rule

VERSION = 'v5d-fresh-entry-v1'


def project_profile(path_id, iteration, raw, curve, reference, tolerance, output, initial_runs):
    output = Path(output)
    dilation = initial_runs[0]['execution']['time_dilation']
    if len({run['execution']['time_dilation'] for run in initial_runs}) != 1:
        raise ValueError('Both periods require one common initial dilation')
    candidates = [dict(candidate_index=0,time_dilation=dilation,runs=initial_runs,
        new_sampling_and_audit_wall_s=0.)]
    runs = initial_runs
    for candidate_index in range(1,rule.MAX_CANDIDATES):
        if all(run['metrics']['success'] for run in runs):
            break
        if any(run['metrics']['reason'] not in ('success','vector_limit_violation') for run in runs):
            break
        factor = (1+rule.CORRECTION_RESERVE)*rule.common_factor(runs)
        dilation *= factor
        scaled = rule.core.copp.Profile3rd(raw['a']/dilation**2,raw['b']/dilation**2,
            tuple(int(x) for x in raw['num_stationary']))
        started = time.perf_counter()
        runs = []
        for dt in rule.PERIODS:
            clock = time.perf_counter()
            positions,times,progress,execution = rule.frozen._sample(curve,raw['s'],scaled,dt,rule.parent.SETTINGS)
            rule.configure_runtime('v3',tolerance=tolerance)
            metric = rule.metrics(reference,positions,dt=dt,tolerance=tolerance,limits=rule.LIMITS)
            filename = f'{path_id}_iteration{iteration}_candidate{candidate_index}_dt{round(dt*1e6)}us.npz'
            np.savez_compressed(output/filename,positions=positions,time=times,
                progress=progress,reference=reference,dt_s=dt)
            execution.update(time_dilation=dilation,time_projection_version='v5d-command-projection',
                controller_version=VERSION,correction_factor=factor,
                correction_reserve_fraction=rule.CORRECTION_RESERVE,wrapper_sha256=rule.core.sha(__file__),
                frozen_numeric_rule_sha256=rule.core.sha(rule.__file__))
            runs.append(dict(dt_s=dt,metrics=metric,execution=execution,
                sample_audit_save_wall_s=time.perf_counter()-clock,
                trace=filename,trace_sha256=rule.core.sha(output/filename),trace_reused=False,
                new_sampling_performed=True,independent_acceptance_reused=False,
                candidate_index=candidate_index))
        candidates.append(dict(candidate_index=candidate_index,time_dilation=dilation,
            correction_factor=factor,new_sampling_and_audit_wall_s=time.perf_counter()-started,runs=runs))
    return dict(projection_candidates=candidates,selected_projection_candidate=candidates[-1]['candidate_index'],
        runs=runs,joint_acceptance_pass=all(run['metrics']['success'] for run in runs))
