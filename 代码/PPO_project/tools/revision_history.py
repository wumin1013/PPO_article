"""Preserve and rerun the manuscript's historical controllers on HEAD source."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / '论文项目' / 'revision_20260908'
LOCAL = ROOT / 'trajectory_autoresearch' / '_local_results'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    destination = AUDIT / 'historical'
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUDIT / 'before_main_d8ff2b5.zip') as archive:
        for name in archive.namelist():
            if name.startswith('PPO_project/') and name.endswith('.py'):
                target = destination / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(archive.read(name))
    sources = {
        'full_method_snapshot': LOCAL / 'runs/20260328_140725_0028_teacher_progress_stable',
        'abl_fixed_lookahead': LOCAL / 'paper_runs/20260329_222832_20260329_222831_paper_matrix_24h_v1_abl_fixed_lookahead/abl_fixed_lookahead',
        'baseline_policy': LOCAL / 'paper_runs/20260329_222832_20260329_222831_paper_matrix_24h_v1_baseline_policy/baseline_policy',
    }
    records = {}
    for name, source in sources.items():
        records[name] = {'source': str(source), 'files': [], 'missing': []}
        selected = [source / 'checkpoints/best_model.pth', source / 'checkpoints/tracking_model_final.pth',
                    ROOT / '论文项目/generated/configs' / (name + '.yaml')]
        for path in selected:
            if path.exists():
                target = destination / name / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(path, target)
                records[name]['files'].append({'source': str(path), 'copy': str(target),
                                                'sha256': digest(path)})
            else:
                records[name]['missing'].append(str(path))
    (destination / 'inventory.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['full_method_snapshot', 'abl_fixed_lookahead', 'baseline_policy'])
    parser.add_argument('--path', choices=['square', 'circle', 'butterfly'])
    args = parser.parse_args()
    destination = prepare()
    if not args.variant:
        print('Historical source and model copies:', destination, flush=True)
        return
    sys.path.insert(0, str(destination / 'source/PPO_project'))
    import numpy as np
    import torch
    import yaml
    from tools.phase32_export_best_trajectories import _rollout_with_model
    torch.set_num_threads(1)
    cfg = yaml.safe_load((destination / args.variant / (args.variant + '.yaml')).read_text(encoding='utf-8'))
    paths = cfg.get('training', {}).get('path_curriculum', {}).get('paths', [])
    if not paths:
        full_cfg = yaml.safe_load((destination / 'full_method_snapshot/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        paths = full_cfg['training']['path_curriculum']['paths']
    records = []
    for spec in paths:
        path_id = spec.get('name', spec['type'])
        if args.path and path_id != args.path:
            continue
        out = destination / args.variant / ('rerun_' + path_id)
        out.mkdir(exist_ok=True)
        if (out / 'result.json').exists():
            print('Preserving existing run', out, flush=True)
            continue
        torch.manual_seed(42)
        np.random.seed(42)
        started = time.perf_counter()
        model = destination / args.variant / 'best_model.pth'
        result = _rollout_with_model(cfg, spec, model, torch.device('cpu'))
        rows = result.pop('rows')
        np.savez_compressed(out / 'trajectory.npz', positions=result.pop('trajectory'),
                            reference=result.pop('reference'), velocity=result.pop('velocities'))
        with (out / 'trace.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result.update(variant=args.variant, path_id=path_id, code_head='d8ff2b505fd6a0dd16349a734869dc609c0a9c53',
                      source_archive_sha256=digest(AUDIT / 'before_main_d8ff2b5.zip'),
                      model_sha256=digest(model), seed=42, wall_seconds=time.perf_counter()-started,
                      command='python ' + ' '.join(sys.argv),
                      execution_time_s=result['steps'] * cfg['environment']['interpolation_period'],
                      max_contour_error_mm=max(abs(r['contour_error']) for r in rows),
                      role='historical regression; not an independent training replicate')
        (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        records.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
