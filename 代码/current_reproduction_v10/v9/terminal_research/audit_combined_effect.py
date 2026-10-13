"""Audit every registered combined early replay, including any failure."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import audit_launch_effect as launch

ROOT, BASE = launch.ROOT, launch.BASE
STUDY = ROOT/'combined_study'
RESULTS = STUDY/'results/early_case_replay/omega60'
sys.path.insert(0, str(BASE/'src'))
from runtime_no_target_override_v4b import install as install_geometry_contract
from metrics_exact_index_v5 import metrics


def plain_timing(data, dt):
    positions = data['positions']
    velocity = np.diff(positions, axis=0)/dt
    acceleration = np.diff(np.vstack((np.zeros(2), velocity)), axis=0)/dt
    speed = np.linalg.norm(velocity, axis=1)
    distance = speed*dt
    deficit = (100.-speed)*dt
    rest = (speed <= .001) & (np.linalg.norm(acceleration, axis=1) <= .01)
    moving = np.flatnonzero(~rest)
    rest_start = int(moving[-1])+1 if len(moving) else 0
    high = np.flatnonzero(speed >= 99.999)
    result = dict(elapsed_prefix_s=len(speed)*dt, executed_length_mm=float(distance.sum()),
        signed_speed_deficit_mm=float(deficit.sum()), terminal_rest_s=(len(speed)-rest_start)*dt)
    if len(high):
        first, last = int(high[0]), int(high[-1])
        parts = {'pre_cruise': (0, first), 'between_first_and_last_cruise': (first,last+1),
                 'post_cruise_before_rest': (last+1, rest_start), 'final_rest': (rest_start, len(speed))}
        result['phase_partition'] = {name: dict(duration_s=(b-a)*dt,
            distance_mm=float(distance[a:b].sum()), signed_speed_deficit_mm=float(deficit[a:b].sum()))
            for name,(a,b) in parts.items()}
    assert abs(len(speed)*dt-(result['executed_length_mm']+result['signed_speed_deficit_mm'])/100.) < 1e-9
    return result


def recheck_metric(data, record):
    install_geometry_contract(dt=record['dt_s'], tolerance=record['tolerance_mm'])
    audit = metrics(data['reference'], data['positions'], dt=record['dt_s'],
        tolerance=record['tolerance_mm'], limits=record['limits'],
        declared_success=record['success'], reason=record.get('executor_reason',record['reason']))
    assert audit['success'] == record['success']
    assert audit['complete_time_s'] == record['complete_time_s']
    for field in ('terminal_rest_s','executed_length_mm','max_polyline_deviation_mm',
                  'rms_polyline_deviation_mm','vector_peaks','vector_limit_ratios'):
        assert np.allclose(audit[field], record[field], rtol=1e-11, atol=1e-8), field
    for field in ('corridor_inside','gates_passed','gate_count','ordered_complete','geometry_complete'):
        assert audit['geometry'][field] == record['geometry'][field], field
    return audit


def failed_actor_audit(data, record, checkpoint):
    observations, actions = data['observations'], data['actions']
    reconstructed = np.asarray([launch.independent_actor(obs, checkpoint) for obs in observations])
    exact = np.array_equal(reconstructed, actions)
    assert exact
    return dict(actor_calls=len(actions), independent_tensor_actions_bitwise_equal=exact,
        raw_action_max_abs_error=float(np.abs(reconstructed-actions).max(initial=0.)),
        success=False, all_failed_prefix_samples_retained=True,
        partial_decision_position_samples_retained=record.get('partial_decision_position_samples_retained'))


def difference(left, right):
    """Left minus comparator; never treat a failed prefix as a completion time."""
    result = dict(left_success=left['success'], comparator_success=right['success'],
                  left_time_s=left['complete_time_s'], comparator_time_s=right['complete_time_s'])
    if not (left['success'] and right['success']):
        return dict(result, time_difference_ms=None, time_reduction_percent=None,
                    reason='At least one complete execution failed; no success-only paired timing value.')
    dt = left['complete_time_s']-right['complete_time_s']
    dl = left['timing']['executed_length_mm']-right['timing']['executed_length_mm']
    dd = left['timing']['signed_speed_deficit_mm']-right['timing']['signed_speed_deficit_mm']
    assert abs(dt-(dl+dd)/100.) < 1e-9
    return dict(result, time_difference_ms=dt*1000.,
        time_reduction_percent=-dt/right['complete_time_s']*100.,
        distance_difference_mm=dl, length_contribution_ms=dl*10.,
        speed_deficit_difference_mm=dd, speed_deficit_contribution_ms=dd*10.,
        terminal_rest_difference_ms=(left['timing']['terminal_rest_s']-right['timing']['terminal_rest_s'])*1000.)


def stored_row(path):
    record = launch.read(path.with_suffix('.json'))
    assert launch.sha(path) == record['trace_sha256']
    data = launch.bundle(path)
    return dict(success=record['success'], complete_time_s=record['complete_time_s'],
        record=str(path.with_suffix('.json')), record_sha256=launch.sha(path.with_suffix('.json')),
        trace=str(path), trace_sha256=launch.sha(path), timing=plain_timing(data,record['dt_s'])), data, record


def main():
    target = Path(__file__).with_name('combined_effect_audit.json')
    assert not target.exists(), 'Preserve earlier receipt.'
    registration_path = STUDY/'early_case_replay_registration.json'
    jobs = launch.read(registration_path)['jobs']
    assert len(jobs) == 8
    for job in jobs:
        stem = f"{job['case']}_{job['method']}_dt{round(job['dt']*1e6)}us"
        assert (RESULTS/(stem+'.json')).exists(), 'Wait for all eight attempted runs.'
    reg_path = STUDY/'registration.json'
    reg = launch.read(reg_path)
    for field in ('source_files','dependency_files'):
        for path, digest in reg[field].items():
            assert launch.sha(path) == digest, path
    models = {m:torch.load(p, map_location='cpu',weights_only=False) for m,p in launch.MODELS.items()}
    source_comparison = BASE/'audit/scaled_representative_v8/comparison.json'
    s2 = {}
    for row in launch.read(source_comparison)['rows']:
        if row['method'] != 'S2R':
            continue
        record_path, trace_path = BASE/row['record'], BASE/row['trace']
        assert launch.sha(record_path) == row['record_sha256']
        assert launch.sha(trace_path) == row['trace_sha256']
        data = launch.bundle(trace_path)
        rechecked = recheck_metric(data, row['metrics'])
        s2[row['case'], row['dt_s']] = dict(success=rechecked['success'],
            complete_time_s=rechecked['complete_time_s'], record=str(record_path),
            record_sha256=launch.sha(record_path), trace=str(trace_path),trace_sha256=launch.sha(trace_path),
            timing=plain_timing(data,row['dt_s']), independent_full_interval_audit=rechecked)
    rows, lookup = [], {}
    for job in jobs:
        stem = f"{job['case']}_{job['method']}_dt{round(job['dt']*1e6)}us"
        path = RESULTS/(stem+'.npz')
        new, data, record = stored_row(path)
        assert record['registration_sha256'] == launch.sha(reg_path)
        assert record['model_sha256'] == launch.sha(launch.MODELS[job['method']])
        assert record['new_training_updates'] == 0
        full_audit = recheck_metric(data, record)
        actor = (launch.actor_audit if record['success'] else failed_actor_audit)(data,record,models[job['method']])
        launch_row, launch_data, launch_record = stored_row(launch.EARLY/(stem+'.npz'))
        old_path = launch.OLD/('main' if job['method']=='JNN' else 'fixed')/(stem+'.npz')
        old, olddata, oldrecord = stored_row(old_path)
        assert np.array_equal(data['reference'], olddata['reference'])
        assert np.array_equal(data['reference'], launch_data['reference'])
        assert record['tolerance_mm'] == oldrecord['tolerance_mm'] == launch_record['tolerance_mm']
        assert record['limits'] == oldrecord['limits'] == launch_record['limits']
        prefix = {}
        if record['success'] and launch_record['success']:
            terminal = actor['terminal_start_tick']
            launch_terminal = int(np.flatnonzero(launch_data['rows'][:,12])[0])
            prefix = dict(terminal_tick_equal=terminal==launch_terminal,
                positions_through_terminal_start_bitwise_equal=np.array_equal(data['positions'][:terminal+1], launch_data['positions'][:terminal+1]),
                nonterminal_rows_bitwise_equal=np.array_equal(data['rows'][:terminal],launch_data['rows'][:terminal]),
                all_observations_bitwise_equal=np.array_equal(data['observations'],launch_data['observations']),
                all_raw_actions_bitwise_equal=np.array_equal(data['actions'],launch_data['actions']))
            assert all(prefix.values())
            diagnostic = record['terminal_candidate_diagnostics']
            assert record['executor_setup']['terminal_binding']['source_sha256'] == launch.sha(ROOT/'terminal_research/terminal_candidate.py')
            assert abs(diagnostic['original_tail_s']-(launch_record['complete_time_s']-terminal*job['dt'])) < 1e-9
            assert abs(diagnostic['selected_tail_s']-(record['complete_time_s']-terminal*job['dt'])) < 1e-9
            assert abs(diagnostic['original_rest_ticks']*job['dt']-record['terminal_rest_s']) < 1e-9
            assert record['terminal_rest_s'] == launch_record['terminal_rest_s']
            prefix.update(original_rest_preserved=True,
                original_tail_s=diagnostic['original_tail_s'],selected_tail_s=diagnostic['selected_tail_s'],
                selected_prefix_ticks=diagnostic.get('selected_prefix_ticks'),
                candidate_count=len(diagnostic['candidates']),geometry_checks=diagnostic['geometry_checks'],
                terminal_planning_wall_s=diagnostic['planning_wall_s'])
            new['detailed_timing'] = launch.timing_audit(data,record)
        new.update(case=job['case'],method=job['method'],dt_s=job['dt'],reason=record['reason'],
            actor_identity=actor, independent_full_interval_audit=full_audit,
            original_launch_prefix_and_rest=prefix,
            versus_same_method_v8=difference(new,old),versus_same_method_launch=difference(new,launch_row))
        rows.append(new)
        lookup[job['case'],job['method'],job['dt']] = new
    comparisons = []
    for case in sorted({j['case'] for j in jobs}):
        for dt in (.001,.0005):
            current = lookup[case,'JNN',dt]
            shared_fixed = lookup[case,'FIXED',dt]
            old_stem = f'{case}_FIXED_dt{round(dt*1e6)}us.npz'
            old_fixed,_,_ = stored_row(launch.OLD/'fixed'/old_stem)
            launch_fixed,_,_ = stored_row(launch.EARLY/old_stem)
            comparisons.append(dict(case=case,dt_s=dt,
                jnn_vs_s2r=difference(current,s2[case,dt]),
                jnn_vs_shared_combined_fixed=difference(current,shared_fixed),
                jnn_vs_original_v8_fixed=difference(current,old_fixed),
                jnn_vs_launch_only_fixed=difference(current,launch_fixed),
                shared_combined_fixed_vs_s2r=difference(shared_fixed,s2[case,dt])))
    result = dict(status='passed',scope='All eight registered combined early replays, without choosing best versions or cells.',
        formal_new_experiment=False,additional_independent_random_samples=0,
        all_eight_attempted_cells_retained=True,successful_cells=sum(r['success'] for r in rows),
        failed_cells=[dict(case=r['case'],method=r['method'],dt_s=r['dt_s'],reason=r['reason']) for r in rows if not r['success']],
        generator_sha256=launch.sha(__file__), helper_sha256=launch.sha(launch.__file__),
        registration_sha256=launch.sha(registration_path),study_registration_sha256=launch.sha(reg_path),
        selection_sha256=launch.sha(STUDY/'selection.json'),
        s2_canonical_comparison_sha256=launch.sha(source_comparison),
        source_bindings_verified=reg['source_files'],dependency_bindings_verified=reg['dependency_files'],
        timing_definition='Exact full-trace T=(L+D)/100; D=sum((100-speed)*dt), including mandatory rest. Failed prefixes have null completion time.',
        interpretation='The two scaled programs are previously observed representative families, not independent holdout tests. Early and later formal replays of the same setting are deterministic duplicates, not extra random samples.',
        metadata_observation='Inherited executor_setup.terminal_family still says unchanged finite bounded vector cubic; terminal_brake_unchanged=false and terminal_binding correctly identify the changed family. Raw records were not edited.',
        rows=rows,s2r_rows=list(s2.values()),comparisons=comparisons)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(target),successes=result['successful_cells'],failures=result['failed_cells'],comparisons=comparisons),indent=2))


if __name__=='__main__':
    main()
