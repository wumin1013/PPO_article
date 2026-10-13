"""Schedule immutable legacy adapters by disjoint models; never alter evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import preview_unified_baselines as adapter


def shard_keys(items, shard, shards):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError('Require 0 <= shard < shards and shards >= 1')
    return [(item['baseline'], item['seed']) for index, item in enumerate(items) if index % shards == shard]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=3)
    args = parser.parse_args()
    actual_command = 'python ' + ' '.join(sys.argv)
    candidate = adapter.final_study('direct-explore')
    adapter.validate_final(candidate)
    items = adapter.baseline_runs()
    selected = shard_keys(items, args.shard, args.shards)
    out = candidate.OUT / 'unified_baselines' / adapter.OUTPUT_VERSION / 'evaluate_regression_nominal'
    schedule = out / 'scheduling'
    schedule.mkdir(parents=True, exist_ok=True)
    source_hash = adapter.sha(Path(__file__))
    meta = dict(shard=args.shard, shards=args.shards, models=selected,
        launcher_sha256=source_hash, adapter_sha256=adapter.sha(Path(adapter.__file__)),
        actual_command=actual_command,
        effective_adapter_command='python PPO_project/tools/preview_unified_baselines.py --study direct-explore evaluate --split regression --regime nominal',
        scheduling_only=True,
        note='Only baseline_runs selection and the top-level partial summary destination are changed in memory. All model evaluation, physics, path, protocol and trace code is the unmodified adapter. Each model belongs to exactly one shard.')
    provenance = schedule / f'shard{args.shard}_provenance.json'
    if provenance.exists() and adapter.read(provenance) != json.loads(json.dumps(meta)):
        raise ValueError('Preserve existing shard provenance')
    adapter.write(provenance, meta)
    (schedule / f'shard{args.shard}_launcher.py').write_bytes(Path(__file__).read_bytes())
    lock = schedule / f'shard{args.shard}.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(datetime.now(timezone.utc).isoformat())
    original_runs, original_write = adapter.baseline_runs, adapter.write
    try:
        adapter.baseline_runs = lambda: [item for item in items if (item['baseline'], item['seed']) in selected]
        target_summary = out / 'summary.json'

        def write(path, value):
            if Path(path) == target_summary:
                path = schedule / f'shard{args.shard}_summary.json'
                value = {**value, 'partial_model_shard': meta}
            original_write(path, value)

        adapter.write = write
        sys.argv = ['PPO_project/tools/preview_unified_baselines.py', '--study', 'direct-explore',
                    'evaluate', '--split', 'regression', '--regime', 'nominal']
        adapter.main()
        original_write(schedule / f'shard{args.shard}_completion.json',
                       dict(completed_utc=datetime.now(timezone.utc).isoformat(), models=selected,
                            launcher_sha256=source_hash))
    finally:
        adapter.baseline_runs, adapter.write = original_runs, original_write
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
