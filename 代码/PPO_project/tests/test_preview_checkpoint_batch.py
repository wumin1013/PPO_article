import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from preview_checkpoint_batch import execute, sha


def test_process_log_binds_literal_command_and_unicode_output(tmp_path):
    argv = [sys.executable, '-c', "import sys; sys.stdout.reconfigure(encoding='utf-8'); print('完成')"]
    record = execute(argv, tmp_path, 0, 'runner-source')
    assert record['argv'] == argv
    assert record['returncode'] == 0 and record['process_wall_s'] > 0
    assert '完成' in (tmp_path / '000_stdout.log').read_text(encoding='utf-8')
    assert record['output_sha256'] == sha(tmp_path / '000_stdout.log')
    assert json.loads((tmp_path / '000_command.json').read_text(encoding='utf-8')) == record


def test_failed_process_retains_log_and_returncode(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        execute([sys.executable, '-c', "print('diagnostic'); raise SystemExit(7)"], tmp_path, 0, 'source')
    record = json.loads((tmp_path / '000_command.json').read_text(encoding='utf-8'))
    assert record['returncode'] == 7
    assert (tmp_path / '000_stdout.log').read_text(encoding='utf-8').strip() == 'diagnostic'
    with pytest.raises(FileExistsError):
        execute([sys.executable, '-c', "print('new')"], tmp_path, 0, 'source')
