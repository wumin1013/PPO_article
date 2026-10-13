"""Finite development-only comparison of local teacher terminal feed taper.

The executor stays v4b. Geometry teachers use only the public reference path and
progress query; the observation-only candidate needs exactly fourteen inputs.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate import rollout
import runtime_configuration as runtime
from replay_no_target_override_v4b import no_target_override

OUT = ROOT/'results/teacher_taper_v4'
CANDIDATES = [
    dict(id='geometry_end_beta075', information='reference_geometry', beta=.75),
    dict(id='geometry_end_beta090', information='reference_geometry', beta=.90),
    dict(id='observation_chord_beta075', information='fourteen_observations', beta=.75),
]
SOURCE_NAMES = ['diagnose_teacher_taper_v4.py', 'evaluate.py', 'metrics.py',
                'runtime_configuration.py', 'runtime_bootstrap.py',
                'execution_settings.py', 'corridor_v3.py', 'replay_no_target_override_v4b.py']


def register():
    file = OUT/'registration.json'
    if not file.exists():
        manifest = json.loads((ROOT/'data/manifest.json').read_text('utf-8'))
        rows = [dict(id=p['id'], previous_split=p['split'], reference='data/'+p['id']+'.csv',
                     sha256=sha(ROOT/'data'/(p['id']+'.csv')))
                for p in manifest['paths'] if p['split'] in ('train', 'validation')]
        assert len(rows) == 6
        write_json(file, dict(created_utc=utc(), scope='DEVELOPMENT ONLY, finite teacher diagnosis',
            executor='v4b_no_target_override', nominal_tolerance_mm=1.25, dt_s=.001,
            candidates=CANDIDATES, cases=rows, fixed_preview_mm=3., steering_normalized=0.,
            geometric_angle_feed='100/(1+max tangent-angle change over next12mm)',
            observation_angle_feed='100/(1+abs(wrap(atan2(p15_y-p2_y,p15_x-p2_x)-asin(obs6))))',
            cap='beta*(distance_mm*sqrt(62500))**(2/3), limited to100mm/s; zero distance permits zero feed',
            rationale='Sine and Bezier existing high-feed failures occur beyond open endpoints after every ordered gate; test local deceleration without reducing the full-path maximum feed.',
            geometry_information='Public env.progress() s plus corridor reference cumulative arclength and tangent arrays; no KCM, brake planner, tracker frontier, private method or path identifier in action computation.',
            observation_information='Only14-vector: far30 tangential/normal chord=30*hypot(obs5,obs13); p2=(obs7,obs8),p15=(obs11,obs12); units per original encoder. Chord is a local geometric proxy, not claimed to equal remaining arclength.',
            scope_of_cap='A heuristic teacher derived from zero-acceleration triangular jerk braking; actual vector acceleration, jerk and geometry are certified by unchanged executor and independent full-interval audit.',
            training_implication='Geometry-teacher actions require later fitting to the deployment observation; observation-only candidate is directly executable from14 inputs. Neither is a new runtime guard.',
            expected_runs=18, selection='No further candidates or parameter changes in this registration; retain all18 outcomes. Do not select paths by success.',
            no_new_final_paths=True, source_sha256={n:sha(ROOT/'src'/n) for n in SOURCE_NAMES}))
    reg = json.loads(file.read_text('utf-8'))
    for name, expected in reg['source_sha256'].items():
        assert sha(ROOT/'src'/name) == expected, name
    for row in reg['cases']:
        assert sha(ROOT/row['reference']) == row['sha256'], row['id']
    return reg


class TaperTeacher:
    def __init__(self, candidate):
        self.candidate = candidate
        self.decisions = []

    def action_for(self, env):
        x = env.observation()
        c = self.candidate
        if c['information'] == 'reference_geometry':
            progress, index = env.progress()
            arcs, tangents = env.corridor.arcs, env.corridor.tangents
            end = min(len(tangents)-1, int(np.searchsorted(arcs, progress+12., side='right')-1))
            turn = float(np.arccos(np.clip(tangents[index:end+1]@tangents[index], -1., 1.)).max(initial=0.))
            distance = max(0., float(arcs[-1]-progress))
        else:
            vector = x[11:13]-x[7:9]
            reference_angle = float(np.arcsin(np.clip(x[6], -1., 1.)))
            ahead_angle = float(np.arctan2(vector[1], vector[0])) if np.linalg.norm(vector) > 1e-9 else reference_angle
            turn = abs((ahead_angle-reference_angle+np.pi)%(2*np.pi)-np.pi)
            distance = 30.*float(np.hypot(x[5], x[13]))
        geometry_feed = 100./(1.+turn)
        cap = min(100., c['beta']*(distance*np.sqrt(62500.))**(2./3.))
        feed = min(geometry_feed, cap)
        self.decisions.append([distance, turn, geometry_feed, cap, feed])
        return np.array([0., feed/50.-1.])


def run_all():
    torch.set_num_threads(1)
    reg = register()
    records = []
    started = time.perf_counter()
    for candidate in reg['candidates']:
        for row in reg['cases']:
            output = OUT/candidate['id']/(row['id']+'.npz')
            if output.with_suffix('.json').exists():
                result = json.loads(output.with_suffix('.json').read_text('utf-8'))
            else:
                runtime._ordered_target = no_target_override
                teacher = TaperTeacher(candidate)
                reference = np.loadtxt(ROOT/row['reference'], delimiter=',', skiprows=1)
                result = rollout(teacher, reference, output, fixed_d=3., dt=.001,
                                 tolerance=1.25, wall_s=90., runtime_version='v3')
                with np.load(output) as trace:
                    rows = trace['rows']
                    position = trace['positions'][-1]
                    velocity = rows[-1, 2:4] if len(rows) else np.zeros(2)
                    acceleration = rows[-1, 4:6] if len(rows) else np.zeros(2)
                    actions = trace['actions']
                diagnostics = dict(columns=['distance_proxy_mm', 'turn_rad', 'angle_feed_mm_s', 'taper_cap_mm_s', 'request_feed_mm_s'],
                    decisions=teacher.decisions, last_position_mm=position.tolist(), endpoint_mm=reference[-1].tolist(),
                    last_velocity_mm_s=velocity.tolist(), last_acceleration_mm_s2=acceleration.tolist(),
                    last_actions=actions[-5:].tolist())
                write_json(output.with_name(output.stem+'_decisions.json'), diagnostics)
                result.update(scope='DEVELOPMENT ONLY', path=row['id'], teacher=candidate,
                    runtime_version='v4b_no_target_override', fixed_preview_mm=3., reference_sha256=row['sha256'],
                    registration_sha256=sha(OUT/'registration.json'), action_source='registered_local_geometric_teacher_taper',
                    teacher_decision_count=len(teacher.decisions), all_failed_prefixes_retained=True)
                write_json(output.with_suffix('.json'), result)
            records.append(dict(candidate=candidate['id'], path=row['id'], success=result['success'],
                complete_time_s=result['complete_time_s'], elapsed_prefix_s=result['elapsed_prefix_s'], reason=result['reason'],
                gates_passed=result['geometry']['gates_passed'], gate_count=result['geometry']['gate_count'],
                endpoint_error_mm=result['geometry']['endpoint_error_mm'], vector_peaks=result['vector_peaks'],
                result_file=str(output.with_suffix('.json').relative_to(ROOT))))
            write_json(OUT/'summary.json', dict(scope='DEVELOPMENT ONLY', expected_runs=18, records=records,
                candidates=[dict(id=c['id'], successes=sum(v['success'] for v in records if v['candidate']==c['id']),
                                 runs=sum(v['candidate']==c['id'] for v in records)) for c in CANDIDATES],
                wall_s=time.perf_counter()-started))
            print(candidate['id'], row['id'], result['success'], result['complete_time_s'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if args.register:
        print(json.dumps(register(), indent=2))
    else:
        run_all()
