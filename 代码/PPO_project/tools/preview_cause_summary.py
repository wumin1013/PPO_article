"""Summarize saved validation causal probes; never evaluate or update a model."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '论文项目/revision_20260909_cause_analysis'


def main():
    summaries = []
    inputs = {}
    for path in sorted((OUT / 'sensitivity').glob('*/summary.json')):
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = json.loads(path.read_text(encoding='utf-8'))
        for entry in result['by_path']:
            path_id = entry['path_id']
            branches = [row for row in result['branch_comparisons'] if row['path_id'] == path_id]
            modes = {}
            for kind in ('hold', 'impulse'):
                values = [row for row in branches if row['mode'].startswith(kind)]
                errors = [row['final_position_difference_mm'] for row in values]
                modes[kind] = dict(branch_count=len(values),
                    final_position_difference_mm_mean=float(np.mean(errors)),
                    final_position_difference_mm_median=float(np.median(errors)),
                    final_position_difference_mm_max=float(np.max(errors)),
                    reward_delta_min=float(min(row['cumulative_reward_difference'] for row in values)),
                    reward_delta_max=float(max(row['cumulative_reward_difference'] for row in values)),
                    early_ends=sum(row['early_end'] for row in values))
            summaries.append(dict(variant=entry['variant'], path_id=path_id,
                one_step_pairs=entry['pairs'], short_branches=modes))
    target = OUT / 'sensitivity_summary.json'
    payload = dict(inputs_sha256=inputs, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=summaries, inference='Descriptive within-path counterfactuals for seed 11 in each beta group; states are not independent training replicates',
        intervention='Same original physical action sequence; compare D=3 mm baseline with +/-0.19 mm either for 1 ms or for 128 ms')
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if target.exists() and target.read_text(encoding='utf-8') != encoded:
        raise FileExistsError('Preserve a changed prior diagnosis summary')
    target.write_text(encoded, encoding='utf-8')
    lines = ['|配置（seed 11）|验证路径|改变意图后当拍角速度相同|1 ms 扰动后128 ms位置差均值/mm|持续128 ms位置差均值/mm|',
             '|---|---|---:|---:|---:|']
    for row in summaries:
        pair = row['one_step_pairs'][0]
        modes = row['short_branches']
        lines.append(f"|{row['variant']}|{row['path_id']}|{pair['identical_omega_despite_changed_intent_count']}/{pair['intent_changed_count']} ({100*pair['identical_omega_given_changed_intent_fraction']:.2f}%)|{modes['impulse']['final_position_difference_mm_mean']:.6g}|{modes['hold']['final_position_difference_mm_mean']:.6g}|")
    (OUT / 'sensitivity_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
