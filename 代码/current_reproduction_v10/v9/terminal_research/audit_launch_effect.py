"""Read-only retained-trace audit; no rollout, training, or result selection."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
EARLY = ROOT/'launch_study/results/early_case_replay/omega60'
OLD = BASE/'results/representative_scaled_v8/learned'
FORMAL = ROOT/'launch_study/results/formal/omega60'
REGISTRATION = ROOT/'launch_study/early_case_replay_registration.json'
MODELS = {'JNN': BASE/'baseline_records/supplementary_reproduction/models/main.pt',
          'FIXED': BASE/'evidence/frozen_main_v6/fixed_two.pt'}
V = 100.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def bundle(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@torch.no_grad()
def independent_actor(observation, saved):
    """Re-evaluate the published frozen tensor expression, one observation/call."""
    x = torch.as_tensor(observation, dtype=torch.float32)
    w = saved['policy']
    three = saved['interface'][1] == 3
    prefix = 'actor.backbone.backbone.' if three else 'actor.backbone.'
    value = x
    for layer in (0, 2, 4):
        value = F.linear(value, w[f'{prefix}{layer}.weight'], w[f'{prefix}{layer}.bias'])
        if layer != 4:
            value = torch.tanh(value)

    def localized(key):
        normalized = (x.unsqueeze(-2) - w[key+'centers']) / w[key+'feature_scale']
        radius = torch.sqrt(normalized.square().sum(-1).clamp_min(1e-20)) / w[key+'radius']
        return (1.-radius).clamp_min(0.).pow(4)*(4.*radius+1.)

    if three:
        assert saved['policy_class'] == 'LocalizedJointPolicy'
        assert saved.get('feed_gain', 1.) == 1.
        correction = .4*torch.tanh(F.linear(localized('actor.backbone.'),
                                            w['actor.backbone.correction.weight']))
        value = torch.cat([value[..., :2], value[..., 2:3]+correction], dim=-1)
    else:
        assert saved['policy_class'] == 'LocalizedTwoPolicy'
    correction = torch.tanh(F.linear(localized('actor.'), w['actor.correction.weight']))
    return (value+correction*w['actor.output_scale']).numpy()


def actor_audit(data, record, saved):
    observations, actions, rows = (data[name] for name in ('observations', 'actions', 'rows'))
    reconstructed = np.asarray([independent_actor(obs, saved) for obs in observations])
    error = float(np.abs(reconstructed-actions).max())
    exact = np.array_equal(reconstructed, actions)
    assert exact, ('Actor actions changed', error)
    dt = record['dt_s']
    macro = round(.02/dt)
    terminal_marks = np.flatnonzero(rows[:, 12])
    assert len(terminal_marks), 'Successful full trace must retain terminal rows.'
    terminal = int(terminal_marks[0])
    assert terminal % macro == 0
    active = terminal//macro
    assert len(actions) == active+1
    feed = 50.*(1.+np.clip(actions[:, 1].astype(float), -1.+1e-10, 1.-1e-10))
    expected_feed = np.repeat(feed[:active], macro)
    actual_feed = np.linalg.norm(rows[:terminal, 8:10], axis=1)
    feed_error = float(np.abs(expected_feed-actual_feed).max())
    assert feed_error < 1e-9
    assert len(actions) == record['actor_calls'] == record['submitted_actions']
    assert record.get('guide_calls', 0) == record.get('guide_interventions', 0) == 0
    return dict(actor_calls=len(actions), independent_tensor_actions_bitwise_equal=exact,
        raw_action_max_abs_error=error, active_tracking_decisions=active,
        terminal_initiating_actor_index=active, terminal_initiating_action_unused_by_tracking=True,
        decoded_feed_request_max_abs_error_mm_s=feed_error,
        active_max_feed_request_count=int(np.sum(feed[:active] >= 99.999999)),
        guide_calls=record.get('guide_calls', 0), guide_interventions=record.get('guide_interventions', 0),
        kcm_guard_ticks=int(np.sum(rows[:, 11])), terminal_start_tick=terminal,
        terminal_start_s=terminal*dt)


def timing_audit(data, record):
    positions, reference, rows = (data[name] for name in ('positions', 'reference', 'rows'))
    dt = record['dt_s']
    velocity = np.diff(positions, axis=0)/dt
    acceleration = np.diff(np.vstack((np.zeros(2), velocity)), axis=0)/dt
    jerk = np.diff(np.vstack((np.zeros(2), acceleration)), axis=0)/dt
    speed = np.linalg.norm(velocity, axis=1)
    amag, jmag = np.linalg.norm(acceleration, axis=1), np.linalg.norm(jerk, axis=1)
    distance = speed*dt
    deficit = (V-speed)*dt
    assert len(rows) == len(speed)
    assert np.array_equal(rows[:, :2], positions[1:])
    assert abs(record['complete_time_s']-len(speed)*dt) < 1e-10
    assert abs(record['executed_length_mm']-distance.sum()) < 1e-8
    cruise = np.flatnonzero(speed >= V-.001)
    assert len(cruise)
    first, last = int(cruise[0]), int(cruise[-1])
    rest = (speed <= .001) & (amag <= .01)
    nonrest = np.flatnonzero(~rest)
    rest_start = int(nonrest[-1])+1
    terminal = int(np.flatnonzero(rows[:, 12])[0])

    def segment(start, stop):
        return dict(start_tick=start, stop_tick_exclusive=stop, start_s=start*dt,
            stop_s=stop*dt, duration_s=(stop-start)*dt,
            distance_mm=float(distance[start:stop].sum()),
            signed_speed_deficit_mm=float(deficit[start:stop].sum()),
            peak_acceleration_mm_s2=float(amag[start:stop].max(initial=0.)),
            peak_jerk_mm_s3=float(jmag[start:stop].max(initial=0.)))

    parts = dict(pre_cruise=segment(0, first),
        between_first_and_last_cruise=segment(first, last+1),
        post_cruise_before_rest=segment(last+1, rest_start),
        final_rest=segment(rest_start, len(speed)))
    assert abs(sum(s['distance_mm'] for s in parts.values())-distance.sum()) < 1e-8
    assert abs(sum(s['signed_speed_deficit_mm'] for s in parts.values())-deficit.sum()) < 1e-8
    assert abs((distance.sum()+deficit.sum())/V-len(speed)*dt) < 1e-10
    assert abs((len(speed)-rest_start)*dt-record['terminal_rest_s']) < 1e-10
    tangent = reference[-1]-reference[-2]
    tangent = tangent/np.linalg.norm(tangent)
    endpoint_delta = positions[-1]-reference[-1]
    return dict(complete_time_s=len(speed)*dt, executed_length_mm=float(distance.sum()),
        signed_speed_deficit_mm=float(deficit.sum()), first_cruise_time_s=(first+1)*dt,
        last_cruise_time_s=(last+1)*dt, terminal_rest_s=(len(speed)-rest_start)*dt,
        phase_partition=parts, launch_120ms=segment(0, round(.12/dt)),
        terminal_brake_and_hold=segment(terminal, len(speed)),
        endpoint_error_mm=float(np.linalg.norm(endpoint_delta)),
        endpoint_along_last_tangent_mm=float(endpoint_delta@tangent),
        endpoint_transverse_last_tangent_mm=float(endpoint_delta@np.array([-tangent[1], tangent[0]])),
        position_recomputed_vector_peaks=[float(speed.max()), float(amag.max()), float(jmag.max())],
        row_velocity_reconstruction_max_abs_error=float(np.abs(velocity-rows[:, 2:4]).max()),
        row_acceleration_reconstruction_max_abs_error=float(np.abs(acceleration-rows[:, 4:6]).max()),
        row_jerk_reconstruction_max_abs_error=float(np.abs(jerk-rows[:, 6:8]).max()))


def analyse(path, saved, model_sha):
    record = read(path.with_suffix('.json'))
    assert record['success'], 'This registered replay cell failed; retain its failure and extend audit.'
    assert sha(path) == record['trace_sha256']
    assert model_sha == record['model_sha256']
    data = bundle(path)
    result = dict(record=str(path.with_suffix('.json')), record_sha256=sha(path.with_suffix('.json')),
        trace=str(path), trace_sha256=sha(path), model_sha256=model_sha,
        success=record['success'], reason=record['reason'],
        actor_identity=actor_audit(data, record, saved), timing=timing_audit(data, record))
    return result, data, record


def differences(old, new):
    oldt, newt = old['timing'], new['timing']
    dt = newt['complete_time_s']-oldt['complete_time_s']
    dl = newt['executed_length_mm']-oldt['executed_length_mm']
    dd = newt['signed_speed_deficit_mm']-oldt['signed_speed_deficit_mm']
    assert abs(dt-(dl+dd)/V) < 1e-10
    return dict(new_minus_v8_time_ms=dt*1000., length_contribution_ms=dl/V*1000.,
        deficit_contribution_ms=dd/V*1000., new_minus_v8_distance_mm=dl,
        new_minus_v8_signed_speed_deficit_mm=dd,
        phase_contributions_ms={key: dict(
            duration_delta_ms=(newt['phase_partition'][key]['duration_s']-oldt['phase_partition'][key]['duration_s'])*1000.,
            distance_delta_mm=newt['phase_partition'][key]['distance_mm']-oldt['phase_partition'][key]['distance_mm'],
            deficit_contribution_ms=(newt['phase_partition'][key]['signed_speed_deficit_mm']-oldt['phase_partition'][key]['signed_speed_deficit_mm'])/V*1000.)
            for key in oldt['phase_partition']})


def compare_bundles(left, right):
    assert set(left) == set(right)
    arrays = {}
    for key in sorted(left):
        same_shape = left[key].shape == right[key].shape
        arrays[key] = dict(left_shape=list(left[key].shape), right_shape=list(right[key].shape),
            bitwise_equal=np.array_equal(left[key], right[key]),
            max_abs_error=float(np.abs(left[key]-right[key]).max(initial=0.)) if same_shape else None)
    return dict(all_arrays_bitwise_equal=all(row['bitwise_equal'] for row in arrays.values()), arrays=arrays)


def formal_receipt(jobs):
    checks = []
    for job in jobs:
        stem = f"{job['case']}_{job['method']}_dt{round(job['dt']*1e6)}us"
        early, formal = EARLY/(stem+'.npz'), FORMAL/(stem+'.npz')
        if not formal.with_suffix('.json').exists():
            checks.append(dict(case=job['case'], method=job['method'], dt_s=job['dt'], status='pending'))
            continue
        record = read(formal.with_suffix('.json'))
        assert sha(formal) == record['trace_sha256']
        checks.append(dict(case=job['case'], method=job['method'], dt_s=job['dt'], status='checked',
            early_trace_sha256=sha(early), formal_trace_sha256=sha(formal),
            formal_record_sha256=sha(formal.with_suffix('.json')),
            content_comparison=compare_bundles(bundle(early), bundle(formal))))
    return dict(all_eight_available=all(c['status']=='checked' for c in checks), checks=checks,
        statistical_interpretation='Duplicate deterministic replay, not a new training seed or independent random sample.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal-only', action='store_true')
    args = parser.parse_args()
    jobs = read(REGISTRATION)['jobs']
    assert len(jobs) == 8
    target = Path(__file__).with_name('launch_formal_identity_audit.json' if args.formal_only else 'launch_effect_audit.json')
    assert not target.exists(), 'Preserve the previous audit receipt.'
    for job in jobs:
        stem = f"{job['case']}_{job['method']}_dt{round(job['dt']*1e6)}us"
        assert (EARLY/(stem+'.json')).exists(), 'Wait for all eight full replays.'
    if args.formal_only:
        comparison = formal_receipt(jobs)
        assert comparison['all_eight_available'], 'Wait for the corresponding formal matrix cells.'
        assert all(c['content_comparison']['all_arrays_bitwise_equal'] for c in comparison['checks'])
        result = dict(generator_sha256=sha(__file__), **comparison)
    else:
        saved = {method: torch.load(path, map_location='cpu', weights_only=False) for method, path in MODELS.items()}
        rows = []
        for job in jobs:
            stem = f"{job['case']}_{job['method']}_dt{round(job['dt']*1e6)}us"
            model_sha = sha(MODELS[job['method']])
            old_path = OLD/('main' if job['method']=='JNN' else 'fixed')/(stem+'.npz')
            old, olddata, oldrecord = analyse(old_path, saved[job['method']], model_sha)
            new, newdata, newrecord = analyse(EARLY/(stem+'.npz'), saved[job['method']], model_sha)
            assert np.array_equal(olddata['reference'], newdata['reference'])
            assert oldrecord['tolerance_mm'] == newrecord['tolerance_mm']
            assert oldrecord['limits'] == newrecord['limits'] == [100., 2000., 62500.]
            assert newrecord['executor_setup']['terminal_brake_unchanged']
            assert newrecord['omega'] == 60. and newrecord['new_training_updates'] == 0
            common = min(len(olddata['actions']), len(newdata['actions']))
            rows.append(dict(case=job['case'], method=job['method'], dt_s=job['dt'], v8=old, launch_v9=new,
                same_reference_and_limits=True, difference=differences(old, new),
                closed_loop_action_comparison=dict(
                    first_observation_bitwise_equal=np.array_equal(olddata['observations'][0], newdata['observations'][0]),
                    first_action_bitwise_equal=np.array_equal(olddata['actions'][0], newdata['actions'][0]),
                    common_actor_indices=common,
                    identical_raw_action_rows_at_common_indices=int(np.sum(np.all(olddata['actions'][:common] == newdata['actions'][:common], axis=1))),
                    interpretation='Same actor mapping does not imply the same subsequent actions: the changed initial execution changes subsequent observations.')))
        result = dict(status='passed', scope='Read-only full retained trace and tensor inference audit; no rollout, training, tuning, or trajectory editing.',
            formal_new_experiment=False, additional_independent_random_samples=0,
            generator_sha256=sha(__file__), replay_registration_sha256=sha(REGISTRATION),
            selection_sha256=sha(ROOT/'launch_study/selection.json'),
            source_bindings={str(p):sha(p) for p in [ROOT/'src/launch_tracking.py', BASE/'src/frozen_policy.py', *MODELS.values()]},
            speed_deficit_definition='D = sum((100 - norm(diff(position)/dt))*dt); exact T = (L+D)/100, including mandatory final rest.',
            phase_partition_definition='pre_cruise ends before first speed >=99.999 mm/s; middle includes first through last qualifying interval; post_cruise ends before final threshold-qualified rest. Phase boundaries may differ across traces.',
            actor_identity_definition='Every raw saved action is re-evaluated independently from the unchanged checkpoint at its own saved 14-feature observation, using one inference per recorded actor call.',
            result_interpretation='Eight deterministic early replays are duplicates of the selected formal configuration, not independent training seeds. Completion times remain the original full traces; no trimming or rest adjustment.',
            rows=rows, formal_replay_comparison=formal_receipt(jobs))
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(target), status=result.get('status','passed'),
        rows=[dict(case=r['case'], method=r['method'], dt_s=r['dt_s'],
            old_time_s=r['v8']['timing']['complete_time_s'], new_time_s=r['launch_v9']['timing']['complete_time_s'],
            **r['difference']) for r in result.get('rows', [])]), indent=2))


if __name__ == '__main__':
    main()
