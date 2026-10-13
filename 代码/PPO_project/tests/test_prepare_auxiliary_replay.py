"""Additive archive preparation must preserve every existing replay byte."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import prepare_auxiliary_replay as supplement


def test_identical_existing_bytes_are_reused_without_rewriting(tmp_path):
    old = tmp_path / 'old.py'; old.write_bytes(b'\xef\xbb\xbfprint(1)\n')
    stamp = old.stat().st_mtime_ns
    added, reused = supplement.additive_copy(tmp_path, {'old.py': old.read_bytes(), 'models/new.bin': b'model'})
    assert added == ['models/new.bin']
    assert reused == ['old.py']
    assert old.stat().st_mtime_ns == stamp
    assert (tmp_path / 'models/new.bin').read_bytes() == b'model'


def test_conflict_refuses_before_creating_any_planned_file(tmp_path):
    (tmp_path / 'existing.py').write_bytes(b'original')
    with pytest.raises(FileExistsError, match='no files overwritten'):
        supplement.additive_copy(tmp_path, {'new.py': b'new', 'existing.py': b'changed'})
    assert not (tmp_path / 'new.py').exists()
    assert (tmp_path / 'existing.py').read_bytes() == b'original'


def test_unsafe_destination_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unsafe relative'):
        supplement.additive_copy(tmp_path, {'../escape.py': b'no'})


def test_differing_archive_members_cannot_collide():
    contents = {}
    supplement.add_bytes(contents, 'src/shared.py', b'one')
    with pytest.raises(ValueError, match='Conflicting archived bytes'):
        supplement.add_bytes(contents, 'src/shared.py', b'two')
    assert contents == {'src/shared.py': b'one'}


def test_identical_archive_members_are_deduplicated():
    contents = {}
    supplement.add_bytes(contents, 'src/shared.py', b'one')
    supplement.add_bytes(contents, 'src/shared.py', b'one')
    assert len(contents) == 1
