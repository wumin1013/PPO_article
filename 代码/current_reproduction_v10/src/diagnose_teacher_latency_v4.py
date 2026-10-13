"""Second finite teacher diagnosis: account for known tracking/hold lag."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import json
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate import rollout
import runtime_configuration as runtime
from replay_no_target_override_v4b import no_target_override
from diagnose_teacher_taper_v4 import TaperTeacher, register as first_registration

OUT = ROOT/'results/teacher_latency_v4'
CANDIDATES = [dict(id='geometry_latency070', information='reference_geometry', beta=.75),
              dict(id='observation_latency070', information='fourteen_observations', beta=.75)]


def stopping_distance(speed):
    jerk, acceleration = 62500., 2000.
    if speed <= acceleration**2/jerk:
        return speed*np.sqrt(speed/jerk)
    return .5*speed*(speed/acceleration+acceleration/jerk)


def velocity_cap(distance):
    low, high = 0., 100.
    for _ in range(24):
        speed = .5*(low+high)
        if .07*speed+stopping_distance(speed) <= distance:
            low = speed
        else:
            high = speed
    return low


class LatencyTeacher(TaperTeacher):
    def action_for(self, env):
        super().action_for(env)
        distance, turn, geometry_feed, _, _ = self.decisions[-1]
        cap = velocity_cap(distance)
        feed = min(geometry_feed, cap)
        self.decisions[-1] = [distance, turn, geometry_feed, cap, feed]
        return np.array([0., feed/50.-1.])


def register():
    previous = first_registration()
    file = OUT/'registration.json'
    if not file.exists():
        tracking = ROOT/'runtime/PPO_article_work/论文项目/revision_20260913_preview_efficiency_v2/no_kcm_coverage_v2/tracking.py'
        write_json(file, dict(created_utc=utc(), scope='DEVELOPMENT ONLY, second finite teacher diagnosis',
            executor='v4b_no_target_override', nominal_tolerance_mm=1.25, dt_s=.001, candidates=CANDIDATES,
            cases=previous['cases'], expected_runs=12, fixed_preview_mm=3., steering_normalized=0.,
            preregistration_reason='All18 previous candidates finished; endpoint overshoot persisted. Residual velocity remained42-54mm/s under beta=.75.',
            previous_registration_sha256=sha(ROOT/'results/teacher_taper_v4/registration.json'),
            previous_summary_sha256=sha(ROOT/'results/teacher_taper_v4/summary.json'),
            tracking_source_sha256=sha(tracking), response_delay_s=.07,
            delay_derivation='Velocity tracking law jerk=1600*(desired-v)-80*a gives low-frequency group delay80/1600=.05s; add .02s maximum held-action interval. This estimates a teacher response allowance, not a safety certificate.',
            cap='Largest v in[0,100] with .07*v+s_stop(v)<=distance; s_stop=v*sqrt(v/J) for v<=A*A/J, else .5*v*(v/A+A/J); A=2000,J=62500.',
            candidate_information=previous['geometry_information']+'; '+previous['observation_information'],
            allocated_candidates='Exactly geometry and observation variants at the derived70ms allowance, no parameter grid.',
            limitations='No initial-acceleration term in heuristic. Full execution and independent vector/geometric/ordered/rest audits determine completion.',
            teacher_rule_unchanged=previous['geometric_angle_feed']+'; '+previous['observation_angle_feed'],
            source_sha256={**previous['source_sha256'], 'diagnose_teacher_latency_v4.py':sha(__file__)},
            all_previous_failures_retained=True, no_new_final_paths=True, no_model_training=True))
    reg = json.loads(file.read_text('utf-8'))
    for name, expected in reg['source_sha256'].items():
        assert sha(ROOT/'src'/name) == expected, name
    for row in reg['cases']:
        assert sha(ROOT/row['reference']) == row['sha256'], row['id']
    return reg


def run_all():
    torch.set_num_threads(1)
    reg = register()
    records = []
    started = time.perf_counter()
    for candidate in reg['candidates']:
        for row in reg['cases']:
            output = OUT/candidate['id']/(row['id']+'.npz')
            runtime._ordered_target = no_target_override
            teacher = LatencyTeacher(candidate)
            reference = np.loadtxt(ROOT/row['reference'], delimiter=',', skiprows=1)
            result = rollout(teacher, reference, output, fixed_d=3., dt=.001,
                             tolerance=1.25, wall_s=90., runtime_version='v3')
            with np.load(output) as trace:
                rows = trace['rows']
                diagnostics = dict(columns=['distance_proxy_mm', 'turn_rad', 'angle_feed_mm_s', 'latency_cap_mm_s', 'request_feed_mm_s'],
                    decisions=teacher.decisions, last_position_mm=trace['positions'][-1].tolist(),
                    endpoint_mm=reference[-1].tolist(), last_velocity_mm_s=rows[-1, 2:4].tolist(),
                    last_acceleration_mm_s2=rows[-1, 4:6].tolist(), last_actions=trace['actions'][-5:].tolist())
            write_json(output.with_name(output.stem+'_decisions.json'), diagnostics)
            result.update(scope='DEVELOPMENT ONLY', path=row['id'], teacher=candidate,
                runtime_version='v4b_no_target_override', fixed_preview_mm=3., reference_sha256=row['sha256'],
                registration_sha256=sha(OUT/'registration.json'), action_source='registered_teacher_tracking_latency_taper',
                teacher_decision_count=len(teacher.decisions), all_failed_prefixes_retained=True)
            write_json(output.with_suffix('.json'), result)
            records.append(dict(candidate=candidate['id'], path=row['id'], success=result['success'],
                complete_time_s=result['complete_time_s'], elapsed_prefix_s=result['elapsed_prefix_s'], reason=result['reason'],
                gates_passed=result['geometry']['gates_passed'], gate_count=result['geometry']['gate_count'],
                endpoint_error_mm=result['geometry']['endpoint_error_mm'], vector_peaks=result['vector_peaks'],
                result_file=str(output.with_suffix('.json').relative_to(ROOT))))
            write_json(OUT/'summary.json', dict(scope='DEVELOPMENT ONLY', expected_runs=12, records=records,
                candidates=[dict(id=c['id'], successes=sum(v['success'] for v in records if v['candidate']==c['id']),
                                 runs=sum(v['candidate']==c['id'] for v in records)) for c in CANDIDATES],
                wall_s=time.perf_counter()-started))
            print(candidate['id'], row['id'], result['success'], result['complete_time_s'], flush=True)


if __name__ == '__main__':
    run_all()
