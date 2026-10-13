"""Append the four preselected fixed-preview records to the portable evidence."""
import copy
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json
from prepare_frozen_main_v6 import OUT, read, copy_checked


def main():
    manifest_file = OUT / 'manifest.json'
    manifest = read(manifest_file)
    previous = copy.deepcopy(manifest)
    expected = '40af7ea30bec92d9cd42e060b2d71ee72919c4cc50c3b28281ff036a0790f469'
    assert all(row['method'] != 'FIXED' for row in manifest['runs']), 'Already appended'
    audit_file = ROOT / 'audit/fixed_comparator_complete_trace_audit_v5.json'
    audit = read(audit_file)
    assert audit['all_four_success'] and len(audit['runs']) == 4
    before_hash = sha(manifest_file)
    backup = OUT / 'source_records/manifest_before_fixed_append_v6.json'
    copy_checked(manifest_file, backup)
    copy_checked(Path(audit['model']['path']), OUT / 'fixed_two.pt')
    assert sha(OUT / 'fixed_two.pt') == expected
    copy_checked(audit_file, OUT / 'source_records/fixed_comparator_complete_trace_audit_v5.json')
    new_rows = []
    for run in audit['runs']:
        case, dt, metrics = run['case'], run['dt_s'], run['metrics']
        token = f'{case}_FIXED_dt{round(dt * 1e6)}us'
        original_trace = Path(run['trace'])
        assert sha(original_trace) == run['trace_sha256']
        assert metrics['success'] and metrics['kinematic_pass'] and metrics['rest_pass']
        assert metrics['geometry']['geometry_complete']
        trace_destination = OUT / 'runs' / f'{token}.npz'
        assert not trace_destination.exists()
        with np.load(original_trace) as data:
            arrays = {k: data[k] for k in data.files}
        reference = arrays['reference']
        assert np.array_equal(reference, np.loadtxt(OUT / 'programs' / case / 'reference.csv', delimiter=',', skiprows=1))
        arrays['positions'] = np.vstack([reference[0], arrays['rows'][:, :2]])
        assert abs((len(arrays['positions']) - 1) * dt - metrics['complete_time_s']) < 1e-9
        # Lossless schema addition only: every original array remains unchanged.
        np.savez_compressed(trace_destination, **arrays)
        with np.load(trace_destination) as standardized, np.load(original_trace) as original:
            assert all(np.array_equal(standardized[k], original[k]) for k in original.files)
        original_record = Path(run['original_result'])
        record_destination = OUT / 'source_records' / f'{token}.json'
        copy_checked(original_record, record_destination)
        new_rows.append(dict(id=token, case=case, method='FIXED', dt_s=dt,
            identity=dict(method='Specified independently developed fixed-preview neural comparator',
                runtime='historical_DirectTwoRuntime', model_sha256=expected,
                model_file='fixed_two.pt', fixed_preview_mm=3.,
                comparison_type='Independent developed comparator; unequal training refinement; not a paired ablation'),
            trace='runs/' + trace_destination.name, trace_sha256=sha(trace_destination),
            source_record='source_records/' + record_destination.name,
            source_record_sha256=sha(record_destination),
            original_trace=str(original_trace), original_record=str(original_record),
            original_trace_sha256=run['trace_sha256'],
            trace_transform='All original arrays retained bitwise; positions adds reference[0] followed by rows[:, :2].',
            acceptance_evidence='source_records/fixed_comparator_complete_trace_audit_v5.json',
            acceptance_evidence_sha256=sha(audit_file), metrics=metrics))
    manifest['runs'].extend(new_rows)
    manifest.setdefault('comparator_models', {})['FIXED'] = dict(file='fixed_two.pt', sha256=expected,
        outputs=2, inputs=14, architecture='14-64-64-2 with 192 compact-support Wendland features',
        fixed_preview_mm=3., training_identity='Selected historical developed comparator, not budget-paired with the main model')
    manifest.setdefault('append_history', []).append(dict(created_utc=utc(), kind='specified_fixed_comparator',
        previous_manifest_sha256=before_hash, run_ids=[r['id'] for r in new_rows],
        source_audit_sha256=sha(audit_file)))
    for p in (backup, OUT / 'fixed_two.pt', OUT / 'source_records/fixed_comparator_complete_trace_audit_v5.json'):
        manifest['files'][p.relative_to(OUT).as_posix()] = sha(p)
    for r in new_rows:
        for key, hash_key in (('trace', 'trace_sha256'), ('source_record', 'source_record_sha256')):
            manifest['files'][r[key]] = r[hash_key]
    assert manifest['runs'][:len(previous['runs'])] == previous['runs']
    assert manifest['model_sha256'] == previous['model_sha256']
    assert all(sha(OUT / relative) == expected_hash for relative, expected_hash in manifest['files'].items())
    write_json(manifest_file, manifest)
    print('APPENDED', len(new_rows), 'TOTAL', len(manifest['runs']), 'MANIFEST_SHA256', sha(manifest_file))


if __name__ == '__main__':
    main()
