"""Generate validation clip and training-budget tables from audited receipts."""
from pathlib import Path
import json
from generate_fresh_tables import table, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    budget = json.loads((ROOT / 'results/summary/training_budget_audit.json').read_text(encoding='utf-8'))
    names = {'main': 'Bounded pilot, main', 'clip_0.1': 'Bounded pilot, clip 0.1',
             'clip_0.3': 'Bounded pilot, clip 0.3', 'fullrange': 'Full-range main',
             'fullrange_clip_0.1': 'Full-range clip 0.1', 'fullrange_clip_0.3': 'Full-range clip 0.3'}
    rows = []
    for row in budget['training_by_tag']:
        c = row['cost']
        rows.append(f"{names[row['tag']]} & {row['completed_runs']} & {c['training_interactions']:,} & {c['supervised_optimizer_steps']:,} & {c['ppo_optimizer_steps']:,} & {c['total_wall_s']:.1f}" + r'\\')
    write('generated_training_budget.tex', table(
        'Measured new-training budget, including the bounded calibration pilot and unselected clip settings. Each registered run completed its budget.',
        'tab:training_budget', 'lrrrrr',
        r'Run group & Runs & Decisions & Fit steps & PPO steps & Wall time (s)', rows,
        r'Wall time sums internal process timers on a shared host; it excludes launch/import overhead and differs from elapsed project time when jobs overlap. Teacher construction, validation search and evaluation are additional costs. No historical learned checkpoint was inherited.'))
    clip = json.loads((ROOT / 'results/summary/validation_sensitivity.json').read_text(encoding='utf-8'))
    values = {(r['clip'], r['arm'], r['path']): r for r in clip['records']}
    rows = []
    for ratio in (0.1, 0.2, 0.3):
        numbers = []
        for path in ('validation_trapezoid', 'validation_zigzag'):
            for arm in ('adaptive', 'fixed'):
                r = values[ratio, arm, path]
                numbers.append(f"{r['complete_time_s']:.3f}" if r['success'] else 'Failed')
        rows.append(f'{ratio:.1f} & ' + ' & '.join(numbers) + r'\\')
    write('generated_clip_sensitivity.tex', table(
        'Validation-only PPO clip sensitivity for seed 101. All twelve executions completed; entries are complete times (s). The main clip 0.2 was retained by the predeclared rule.',
        'tab:clip_sensitivity', 'rrrrr',
        r'$\epsilon_{\mathrm{PPO}}$ & Trapezoid adaptive & Trapezoid fixed & Zigzag adaptive & Zigzag fixed', rows))


if __name__ == '__main__':
    main()
