"""Scheduling must preserve the ten-model freeze boundary and failed costs."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_formal_batch as batch


@pytest.fixture
def fake_study(tmp_path, monkeypatch):
    directory = tmp_path / 'PPO_project/tools'
    directory.mkdir(parents=True)
    for name in ('preview_formal_batch.py', 'preview_formal_continuation.py', 'preview_formal_holdout.py'):
        (directory / name).write_text('# immutable source\n', encoding='utf-8')
    monkeypatch.setattr(batch, 'ROOT', tmp_path)
    monkeypatch.setattr(batch, '__file__', str(directory / 'preview_formal_batch.py'))
    return tmp_path / '论文项目/revision_20260910_continued/formal_v1'


def test_all_ten_chains_complete_before_freeze_and_no_test_access(fake_study, monkeypatch):
    calls = []
    def execute(argv, directory, label):
        if label == 'freeze':
            assert len(calls) == 11
        calls.append((label, argv))
        return dict(returncode=0, process_wall_s=2, argv=argv)
    monkeypatch.setattr(batch, 'execute', execute)
    result = batch.run('train', fake_study, 3)
    assert calls[0][0] == 'prepare' and calls[-1][0] == 'freeze'
    assert {x[0] for x in calls[1:-1]} == {f'{m}_seed{s}' for s in batch.SEEDS for m in batch.METHODS}
    assert not any('evaluate' in argv for _, argv in calls)
    assert result['status'] == 'complete' and result['command_process_wall_s'] == 24


def test_failed_chain_retained_and_prevents_freeze(fake_study, monkeypatch):
    calls = []
    def execute(argv, directory, label):
        calls.append(label)
        return dict(returncode=1 if label == 'adaptive_seed22' else 0, process_wall_s=3, argv=argv)
    monkeypatch.setattr(batch, 'execute', execute)
    with pytest.raises(RuntimeError, match='jobs failed'):
        batch.run('train', fake_study, 2)
    assert len(calls) == 11 and 'freeze' not in calls
    result = json.loads(next(fake_study.glob('batch_invocations/*/summary.json')).read_text(encoding='utf-8'))
    assert result['status'] == 'failed_or_interrupted'
    assert len(result['commands']) == 11 and result['command_process_wall_s'] == 33


def test_no_holdout_subprocess_before_freeze(fake_study, monkeypatch):
    monkeypatch.setattr(batch, 'execute', lambda *a: pytest.fail('Must not launch subprocess'))
    with pytest.raises(ValueError, match='freeze all ten'):
        batch.run('holdout', fake_study, 1)


def test_real_subprocess_failure_records_exit_output_and_cost(tmp_path):
    result = batch.execute([sys.executable, '-c', "print('retained diagnostic'); raise SystemExit(7)"], tmp_path, 'failure')
    assert result['returncode'] == 7 and result['status'] == 'failed'
    assert result['process_wall_s'] > 0
    assert (tmp_path / 'failure.log').read_text(encoding='utf-8').strip() == 'retained diagnostic'
    assert json.loads((tmp_path / 'failure.json').read_text(encoding='utf-8')) == result
