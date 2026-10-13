"""Archived evidence stays reproducible while tampering is rejected."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_saved_artifacts as evidence


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def saved(tmp_path):
    base = tmp_path / 'saved'; base.mkdir()
    run = base / 'formal/method_seed11'; run.mkdir(parents=True)
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}', encoding='utf-8')
    raw = b'\xef\xbb\xbfprint(1)\n'
    files = {'src/example.py': hashlib.sha256(raw).hexdigest()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (run / 'config.yaml').write_text('seed: 11\n', encoding='utf-8')
    (run / 'checkpoint.pt').write_bytes(b'opaque-model-fixture')
    provenance = dict(source_files=files, source_sha256=digest,
                      config_sha256=evidence.sha(run / 'config.yaml'),
                      path_manifest_sha256=evidence.sha(manifest))
    write(run / 'provenance.json', provenance)
    write(run / 'training_summary.json', dict(checkpoint_sha256=evidence.sha(run / 'checkpoint.pt')))
    with zipfile.ZipFile(run / 'source.zip', 'w') as archive:
        archive.writestr('src/example.py', raw)
    write(base / 'freeze.json', dict(models={'formal/method_seed11': evidence.sha(run / 'checkpoint.pt')},
          configs={'formal/method_seed11': evidence.sha(run / 'config.yaml')},
          source_sha256=digest, path_manifest_sha256=evidence.sha(manifest)))
    return base, run, manifest


def test_live_tree_encoding_cleanup_does_not_change_archived_evidence(saved, tmp_path):
    base, _, manifest = saved
    live = tmp_path / 'src'; live.mkdir()
    (live / 'example.py').write_bytes(b'print(1)\n')  # Archive deliberately retains its historical BOM.
    frozen, runs = evidence.verify_saved_freeze(base, {'formal/method_seed11'}, manifest)
    assert len(frozen['models']) == len(runs) == 1


def test_changed_archived_source_member_is_rejected(saved):
    base, run, manifest = saved
    with zipfile.ZipFile(run / 'source.zip', 'w') as archive:
        archive.writestr('src/example.py', b'print(999)\n')
    with pytest.raises(ValueError, match='source member hash mismatch'):
        evidence.verify_saved_freeze(base, {'formal/method_seed11'}, manifest)


@pytest.mark.parametrize('target', ['checkpoint.pt', 'config.yaml'])
def test_changed_frozen_model_or_configuration_is_rejected(saved, target):
    base, run, manifest = saved
    (run / target).write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint/config mismatch'):
        evidence.verify_saved_freeze(base, {'formal/method_seed11'}, manifest)


def test_missing_archive_member_is_rejected(saved):
    base, run, manifest = saved
    with zipfile.ZipFile(run / 'source.zip', 'w'):
        pass
    with pytest.raises(ValueError, match='membership differs'):
        evidence.verify_saved_freeze(base, {'formal/method_seed11'}, manifest)


def test_changed_data_manifest_is_rejected(saved):
    base, _, manifest = saved
    manifest.write_text('{"changed":true}', encoding='utf-8')
    with pytest.raises(ValueError, match='path manifest mismatch'):
        evidence.verify_saved_freeze(base, {'formal/method_seed11'}, manifest)


def test_incomplete_registered_model_roster_is_rejected(saved):
    base, _, manifest = saved
    with pytest.raises(ValueError, match='complete registered'):
        evidence.verify_saved_freeze(base, {'formal/method_seed11', 'formal/method_seed22'}, manifest)
