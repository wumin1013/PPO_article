"""Read-only input/probe verification and a manifest for the cause analysis."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '论文项目/revision_20260909_cause_analysis'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    expected = {
        '论文项目/main.tex': '6437c5f3234c94d9ef748e7aa65841b4c30dce58f8f0690ab414eab8007be70c',
        '论文项目/main.pdf': '8a0f342ddd86a14e90ace15e8fb190a88df178c265713787159a90a9fae82b37',
        '论文项目/revision_20260908/before_main_d8ff2b5.zip': 'be26cd2400c2d8e8f236bbfbd2af76cedcbcaec74b83a94f27eef16513af4cf5',
    }
    for name, digest in expected.items():
        assert sha(ROOT / name) == digest, name
    original_inputs = {}
    validations = []
    for folder in sorted((OUT / 'sensitivity').iterdir()):
        protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        assert protocol['source_sha256'] == '5e601663d463b0c916c75ec2a5df9fdb70a7a952841a546c83fcdf4f00aebc79'
        for name, digest in protocol['inputs_sha256'].items():
            assert sha(name) == digest, name
            original_inputs[name] = digest
        for name, digest in json.loads((folder / 'hash_manifest.json').read_text(encoding='utf-8')).items():
            assert sha(folder / name) == digest, name
        assert summary['no_optimizer_called'] and summary['frozen_source_unchanged'] and summary['input_files_unchanged']
        assert len(summary['replay_checks']) == 3
        assert all(row['exact_position_replay'] and row['clone_obs_max_difference'] == 0 for row in summary['replay_checks'])
        assert len(summary['branch_comparisons']) == 72
        assert all(row['matched_steps'] == 128 and not row['early_end'] for row in summary['branch_comparisons'])
        validations.append(dict(run=folder.name, replay_steps=sum(row['steps'] for row in summary['replay_checks']),
            one_step_count=summary['one_step_count'], branch_transition_count=summary['branch_transition_count']))
    assert len(validations) == 2
    sources = [ROOT / 'PPO_project/tools' / name for name in (
        'preview_cause_sensitivity.py', 'preview_cause_summary.py',
        'preview_ppo_cause_diagnostic.py', 'verify_preview_cause_delivery.py')]
    sources += list(OUT.rglob('*.py'))
    for path in sources:
        raw = path.read_bytes()
        assert not raw.startswith(b'\xef\xbb\xbf'), str(path)
        ast.parse(raw.decode('utf-8'), filename=str(path))
    documents = [ROOT / '论文项目' / name for name in (
        '固定前瞻与自适应前瞻原因分析.md', '修改说明.md', '实验复现说明.md')]
    for path in documents:
        assert not path.read_bytes().startswith(b'\xef\xbb\xbf')
        path.read_bytes().decode('utf-8')
    files = {path.relative_to(ROOT).as_posix(): sha(path) for path in OUT.rglob('*')
             if path.is_file() and '__pycache__' not in path.parts and path.name != 'final_delivery_audit.json'}
    files.update({path.relative_to(ROOT).as_posix(): sha(path) for path in [*sources, *documents]})
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        unchanged_manuscript_and_archive=expected, unchanged_sensitivity_inputs=original_inputs,
        sensitivity_checks=validations, checked_python_files=len(sources), files_sha256=files,
        passed=True, new_training_steps=0, scope='Diagnostic scripts/reports only; no algorithm, checkpoint or paper performance changes')
    (OUT / 'final_delivery_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('branch', 'head', 'passed', 'sensitivity_checks', 'checked_python_files')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
