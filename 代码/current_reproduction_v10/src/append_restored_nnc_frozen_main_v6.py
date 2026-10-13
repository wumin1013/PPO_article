"""Audit and append all four restored-NNC outcomes, including every failure."""
import copy
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json
from prepare_frozen_main_v6 import OUT, read, copy_checked
from evaluate_restored_nnc_v6 import OUT as NNC, MODEL_SHA, verify
from metrics_exact_index_v5 import metrics
from runtime_configuration import configure_runtime


def main():
    registration = verify()
    manifest_file = OUT / 'manifest.json'
    manifest = read(manifest_file)
    previous = copy.deepcopy(manifest)
    assert all(r['method'] != 'NNC' for r in manifest['runs']), 'NNC already appended'
    independent = []
    for spec in registration['runs']:
        source = NNC / (spec['id']+'.json')
        record = read(source)
        trace = source.with_suffix('.npz')
        assert sha(trace) == record['trace_sha256']
        assert record['model_sha256'] == MODEL_SHA
        assert record['registration_sha256'] == sha(NNC/'registration.json')
        assert record['original_project_caps_preserved'] and not record['original_enable_kcm']
        assert record['dt_s'] == spec['dt_s'] and record['tolerance_mm'] == spec['tolerance_mm']
        with np.load(trace) as data:
            p, reference = data['positions'], data['reference']
            assert np.array_equal(reference, np.loadtxt(OUT/'programs'/spec['case']/'reference.csv', delimiter=',', skiprows=1))
            assert data['observations'].shape[1] == 14 and data['actions'].shape[1] == 2
        configure_runtime('v3', tolerance=spec['tolerance_mm'])
        recomputed = metrics(reference, p, dt=spec['dt_s'], tolerance=spec['tolerance_mm'],
                             declared_success=record['success'], reason=record['reason'])
        for key in ('success', 'complete_time_s', 'elapsed_prefix_s', 'samples', 'geometry',
                    'vector_peaks', 'kinematic_pass', 'rest_pass', 'terminal_rest_s'):
            assert recomputed[key] == record[key], (spec['id'], key)
        if not record['success']: assert record['complete_time_s'] is None
        independent.append(dict(id=spec['id'], status=record['status'],
            source_record_sha256=sha(source), trace_sha256=sha(trace),
            metrics=recomputed, first_failure=record['first_failure'],
            all_peaks_scope=record['all_peaks_scope']))
    audit = dict(created_utc=utc(), source=registration['method'],
        registration_sha256=sha(NNC/'registration.json'), all_four_registered_outcomes_retained=True,
        independently_checked_intervals=sum(r['metrics']['samples']-1 for r in independent),
        successful_programs=sum(r['metrics']['success'] for r in independent),
        total_program_attempts=len(independent), records=independent,
        scope='Saved-command re-audit; all outcomes retained, no new motion or training; prefix peaks describe executed prefixes only.')
    audit_file = ROOT/'audit/restored_nnc_protected_v6_independent_review.json'
    write_json(audit_file, audit)
    backup = OUT/'source_records/manifest_before_nnc_append_v6.json'
    copy_checked(manifest_file, backup)
    audit_destination = OUT/'source_records/restored_nnc_protected_v6_independent_review.json'
    copy_checked(audit_file, audit_destination)
    # The portable evidence retains model, config, source, registration, logs,
    # preflights and all four outcomes together without relying on migration backups.
    for source in NNC.rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts:
            copy_checked(source, OUT/'nnc_runtime'/source.relative_to(NNC))
    rows = []
    for spec in registration['runs']:
        token = spec['id']; record_file = NNC/(token+'.json'); trace = record_file.with_suffix('.npz')
        record = read(record_file)
        copy_checked(trace, OUT/'runs'/trace.name)
        copy_checked(record_file, OUT/'source_records'/record_file.name)
        rows.append(dict(id=token, case=spec['case'], method='NNC', dt_s=spec['dt_s'],
            identity=dict(method='Recovered project NNC adapter with original feed protections',
                runtime='restored_project_NNC_protected_v6', model_sha256=MODEL_SHA,
                model_file='nnc_runtime/checkpoint.pth', original_enable_kcm=False,
                angular_feed_caps=True, turn_prebraking=True, endpoint_braking=True,
                policy_dt_s=.001, published_method_reproduction=False),
            trace='runs/'+trace.name, trace_sha256=sha(trace),
            source_record='source_records/'+record_file.name, source_record_sha256=sha(record_file),
            original_trace=str(trace), original_record=str(record_file),
            acceptance_evidence='source_records/restored_nnc_protected_v6_independent_review.json',
            acceptance_evidence_sha256=sha(audit_file), metrics=record))
    manifest['runs'].extend(rows)
    manifest.setdefault('comparator_models', {})['NNC'] = dict(file='nnc_runtime/checkpoint.pth',
        sha256=MODEL_SHA, inputs=14, outputs=2, hidden_width=256,
        original_enable_kcm=False, configuration='nnc_runtime/baseline_policy.yaml',
        registration='nnc_runtime/registration.json', source='nnc_runtime/source/PPO_project',
        identity='Project PPO-based NNC adapter; no claim of full published DDPG reproduction')
    manifest.setdefault('append_history', []).append(dict(created_utc=utc(),kind='restored_protected_nnc_all_outcomes',
        previous_manifest_sha256=sha(backup), run_ids=[r['id'] for r in rows], audit_sha256=sha(audit_file)))
    for p in OUT.rglob('*'):
        if p.is_file() and p != manifest_file:
            manifest['files'][p.relative_to(OUT).as_posix()] = sha(p)
    assert manifest['runs'][:len(previous['runs'])] == previous['runs']
    write_json(manifest_file, manifest)
    shared = ROOT/'shared_manuscript/frozen_main_v6'
    method = r'''The NNC comparator is a project adapter using a frozen PPO network with 14 observations and two outputs for angular-velocity and feed requests. Its angular-kinematic feed caps, turn prebraking and endpoint braking remain active. The checkpoint configuration disables the native kinematic projection. The policy is evaluated every 1 ms. At the 0.5 ms command period, each request is held for two commands, with cap-release rates and step-counted durations converted to the same physical time scales. The adapter receives the common reference and effective tolerance and is assessed from initial rest using the same ordered-coverage, command-interval, planar-vector and terminal-rest criteria. Each attempt ends at complete acceptance, the first infeasible command, or its fixed execution budget.
'''
    (shared/'nnc_method.tex').write_text(method,encoding='utf-8')
    count = len(independent); passed = audit['successful_programs']
    if passed == 0 and all(r['first_failure'] for r in independent):
        times = sorted({r['first_failure']['time_s'] for r in independent})
        text = (f'The project NNC adapter completed {passed} of the {count} path--resolution evaluations under the common acceptance criteria. '
                'Each attempt reached its first vector-constraint violation at the initial command; '
                f'the retained prefixes contain {" and ".join(f"{1000*t:g}" for t in times)} ms of executed motion. '
                'Completion time is therefore undefined for these attempts, and their constraint peaks refer to the retained prefixes.\n')
    else:
        text = (f'The project NNC adapter completed {passed} of the {count} path--resolution evaluations under the common acceptance criteria. '
                'All remaining attempts retain their executed prefixes and have no assigned completion time.\n')
    (shared/'nnc_results.tex').write_text(text,encoding='utf-8')
    write_json(ROOT/'audit/nnc_manuscript_binding_v6.json', dict(created_utc=utc(),
        source_audit_sha256=sha(audit_file), method_tex_sha256=sha(shared/'nnc_method.tex'),
        results_tex_sha256=sha(shared/'nnc_results.tex'), actual_counts=dict(completed=passed, attempted=count)))
    print('NNC_APPENDED', len(rows), 'COMPLETE', passed, 'MANIFEST_RUNS', len(manifest['runs']))


if __name__ == '__main__': main()
