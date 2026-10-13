"""Archived replay never silently changes bytes or escapes its new directory."""
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_frozen_replay as replay


def zip_bytes(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        for name,value in files.items():archive.writestr(name,value)
    return stream.getvalue()


def test_source_archive_preserves_original_bom_bytes():
    files={'PPO_project/main.py':b'\xef\xbb\xbfprint("original")\n'}
    hashes={name:replay.sha(value) for name,value in files.items()}
    assert replay.archived_files(zip_bytes(files),hashes)==files
    changed={'PPO_project/main.py':files['PPO_project/main.py'][3:]}
    with pytest.raises(ValueError,match='do not match'):
        replay.archived_files(zip_bytes(changed),hashes)


def test_archive_rejects_unregistered_files_and_path_traversal():
    with pytest.raises(ValueError,match='members differ'):
        replay.archived_files(zip_bytes({'extra.py':b'pass'}),{})
    with pytest.raises(ValueError,match='Unsafe relative'):
        replay.archived_files(zip_bytes({'../main.py':b'pass'}),{'../main.py':replay.sha(b'pass')})
    for path in ('C:/main.py','x\\main.py','/main.py','.git/config'):
        with pytest.raises(ValueError,match='Unsafe relative'):replay.safe_relative(path)


def test_target_cannot_overwrite_existing_or_original_study(tmp_path):
    target=tmp_path/'论文项目/revision_20260908/replay'
    assert replay.new_target(tmp_path,target)==target.resolve()
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError,match='Preserve'):replay.new_target(tmp_path,target)
    with pytest.raises(ValueError,match='separate directory'):
        replay.new_target(tmp_path,tmp_path/replay.STUDY/'nested')
    with pytest.raises(ValueError,match='separate directory'):
        replay.new_target(tmp_path,tmp_path/'outside_revision')


def test_verify_rejects_bom_removal_in_restored_frozen_source(tmp_path):
    content=b'\xef\xbb\xbfpass\n';name='PPO_project/main.py'
    path=tmp_path/name;path.parent.mkdir();path.write_bytes(content)
    helper=tmp_path/'replay_inputs/prepare_frozen_replay.py';helper.parent.mkdir();helper.write_bytes(b'helper')
    hashes={name:replay.sha(content)}
    meta=dict(copied_files=hashes,preparation_tool_sha256=replay.sha(b'helper'),
        runtime_source_files=hashes,runtime_source_sha256=replay.source_digest(hashes),copied_model_runs=['formal/direct_seed11'])
    (tmp_path/'replay_manifest.json').write_text(json.dumps(meta),encoding='utf-8')
    assert replay.verify(tmp_path)['source_bytes_verified'] is True
    path.write_bytes(content[3:])
    with pytest.raises(ValueError,match='Replay bytes changed'):replay.verify(tmp_path)
