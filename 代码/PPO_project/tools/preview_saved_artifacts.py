"""Verify archived experiment evidence for reports without executing its policy.

Unlike an evaluation gate, this read-only verifier does not require the active
working tree to match historical source bytes. Every archived source member,
model, configuration, and provenance must still match its registered freeze.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_archive(path, provenance):
    files = provenance['source_files']
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if digest != provenance['source_sha256']:
        raise ValueError('Archived source manifest digest mismatch')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError('Archived source membership differs from provenance')
        for name, expected in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Archived source member hash mismatch: {name}')
    return sha(path)


def verify_saved_freeze(base, required_keys, manifest, verify_protocol=False):
    base = Path(base)
    if not (base / 'freeze.json').exists():
        raise ValueError(f'Final/baseline freeze is required for archived reporting: {base}')
    frozen = read(base / 'freeze.json')
    if set(frozen['models']) != set(required_keys) or set(frozen['configs']) != set(required_keys):
        raise ValueError('Archived freeze must contain the complete registered model/config roster')
    if frozen['path_manifest_sha256'] != sha(manifest):
        raise ValueError('Archived frozen path manifest mismatch')
    if verify_protocol:
        if frozen['protocol_sha256'] != sha(base / 'protocol.json') or frozen['selection_sha256'] != sha(base / 'selection.json'):
            raise ValueError('Archived frozen protocol/selection mismatch')
    runs = {}
    for key in sorted(required_keys):
        run = base / key
        if sha(run / 'checkpoint.pt') != frozen['models'][key] or sha(run / 'config.yaml') != frozen['configs'][key]:
            raise ValueError(f'Archived frozen checkpoint/config mismatch: {key}')
        provenance = read(run / 'provenance.json')
        training = read(run / 'training_summary.json')
        if (provenance['source_sha256'] != frozen['source_sha256']
                or provenance['config_sha256'] != frozen['configs'][key]
                or provenance['path_manifest_sha256'] != frozen['path_manifest_sha256']
                or training['checkpoint_sha256'] != frozen['models'][key]):
            raise ValueError(f'Archived training/freeze provenance mismatch: {key}')
        archive_sha = verify_archive(run / 'source.zip', provenance)
        runs[key] = dict(run=run, provenance=provenance, training=training,
                         checkpoint_sha256=frozen['models'][key], config_sha256=frozen['configs'][key],
                         source_archive_sha256=archive_sha, baseline_freeze_sha256=sha(base / 'freeze.json'))
    return frozen, runs


def final_saved(candidate, manifest):
    if not (candidate.OUT / 'freeze.json').exists():
        raise ValueError(f'Final/baseline freeze is required for archived reporting: {candidate.OUT}')
    protocol = read(candidate.OUT / 'protocol.json')
    variants, seeds = protocol['formal_variants'], protocol['formal_seeds']
    if not variants or not seeds or len(set(variants)) != len(variants) or len(set(seeds)) != len(seeds):
        raise ValueError('Archived formal protocol requires nonempty unique variants/seeds')
    keys = {f'formal/{variant}_seed{seed}' for variant in variants for seed in seeds}
    return verify_saved_freeze(candidate.OUT, keys, manifest, verify_protocol=True)


def legacy_saved(v2_output, seeds, manifest):
    v2_keys = {f'finetune/{variant}_seed{seed}' for variant in ['fixed', 'residual', 'absolute'] for seed in seeds}
    locations = [('absolute', v2_output, 'finetune', v2_keys),
                 ('nnc', v2_output / 'nnc_auxiliary', 'runs', {f'runs/nnc_seed{seed}' for seed in seeds})]
    result = []
    for baseline, base, stage, keys in locations:
        _, verified = verify_saved_freeze(base, keys, manifest)
        for seed in seeds:
            result.append(dict(baseline=baseline, seed=seed, **verified[f'{stage}/{baseline}_seed{seed}']))
    return result
