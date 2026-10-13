"""Record completed holdout use without rewriting the immutable split snapshot."""
import json
from pathlib import Path
import time

import preview_direct_explore as candidate


def main():
    study = candidate.install()
    study.check_protocol()
    frozen = json.loads((study.OUT / 'freeze.json').read_text(encoding='utf-8'))
    expected = {f'formal/{m}_seed{s}' for m in study.FORMAL_VARIANTS for s in study.SEEDS}
    if set(frozen['models']) != expected:
        raise ValueError('Complete final model freeze is required')
    manifest = study.manifest()
    test_ids = {p['id'] for p in manifest['paths'] if p['split'] == 'test'}
    records = []
    for key in sorted(expected):
        run = study.OUT / key
        if study.backend.sha(run / 'checkpoint.pt') != frozen['models'][key]:
            raise ValueError(f'Frozen checkpoint changed: {key}')
        summary = run / 'evaluation_test_nominal/summary.json'
        rows = json.loads(summary.read_text(encoding='utf-8'))['rows']
        if {r['path_id'] for r in rows} != test_ids or len(rows) != len(test_ids):
            raise ValueError(f'Incomplete nominal holdout evaluation: {key}')
        for row in rows:
            if (row['checkpoint_sha256'] != frozen['models'][key]
                    or row['config_sha256'] != frozen['configs'][key]
                    or row['evaluation_source_sha256'] != frozen['source_sha256']
                    or row['split'] != 'test' or row['regime'] != 'nominal'):
                raise ValueError(f'Test provenance mismatch: {key}')
        records.append(dict(run=key, summary_sha256=study.backend.sha(summary),
                            evaluated_path_ids=sorted(test_ids)))
    destination = study.OUT / 'test_exposure_record.json'
    if destination.exists():
        raise FileExistsError('Preserve the recorded exposure event')
    study.backend.write_json(destination, dict(
        status='evaluated_after_final_freeze', records=records,
        statement='The original manifest and exposure ledger record split creation, not current exposure. '
                  'These test results have now been inspected. Any future tuning that uses them '
                  'must treat these paths as development data and reserve a new holdout.',
        recorded_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        final_freeze_timestamp_utc=frozen['timestamp_utc'],
        freeze_sha256=study.backend.sha(study.OUT / 'freeze.json'),
        original_manifest_sha256=study.backend.sha(study.DATA / 'manifest.json'),
        original_exposure_ledger_sha256=study.backend.sha(study.DATA / 'exposure_ledger.json'),
        script_sha256=study.backend.sha(__file__),
        command='python PPO_project/tools/preview_record_test_exposure.py'))
    (study.OUT / 'test_exposure_record_generator.py').write_bytes(Path(__file__).read_bytes())
    print(destination)


if __name__ == '__main__':
    main()
