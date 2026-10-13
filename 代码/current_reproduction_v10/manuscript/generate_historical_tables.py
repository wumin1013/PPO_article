"""Generate manuscript tables from independently recalculated retained records."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript'


def write_table(name, caption, label, columns, header, rows, note=''):
    text = '\\begin{table}[htbp]\n\\centering\n'
    text += f'\\caption{{{caption}}}\n\\label{{{label}}}\n'
    text += '\\small\n\\setlength{\\tabcolsep}{3.5pt}\n'
    text += f'\\begin{{tabular}}{{{columns}}}\n\\toprule\n{header}\\\\\n\\midrule\n'
    text += '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n'
    if note:
        text += '\\par\\smallskip\n\\begin{minipage}{0.98\\linewidth}\\footnotesize\n' + note + '\n\\end{minipage}\n'
    text += '\\end{table}\n'
    (OUT / name).write_text(text, encoding='utf-8')


def main():
    result = json.loads((ROOT / 'results/historical_verified.json').read_text(encoding='utf-8'))
    binding = json.loads((ROOT / 'audit/manuscript_baseline_binding.json').read_text(encoding='utf-8'))
    records = {(r['case'], r['method'], r['dt_s']): r for r in result['records']}
    labels = {'J-NNC': 'J-NNC', 'Two-action': 'Two-action', 'S2': 'Two-step planner'}
    old_names = {'J-NNC': 'J-NNC', 'Two-action': 'two_action', 'S2': 'two_step'}
    deviations = {(r['path'], r['method']): r['maximum_contour_deviation_mm'] for r in binding['main_results_1ms']}
    rows, utilization = [], []
    for case, case_label in [('original', 'Long path'), ('dense', 'Dense case')]:
        if rows:
            rows.append('\\midrule')
            utilization.append('\\midrule')
        for index, method in enumerate(labels):
            r = records[case, method, .001]
            name = case_label if index == 0 else ''
            d = deviations['long' if case == 'original' else case, old_names[method]]
            rows.append(f"{name} & {labels[method]} & {r['complete_time_s']:.3f} & {d:.6f}\\\\")
            utilization.append(f"{name} & {labels[method]} & {r['mean_velocity_use_percent']:.3f} & {r['acceleration_95pct_percent']:.3f} & {r['jerk_95pct_percent']:.3f}\\\\")
    write_table('generated_historical_main.tex',
                'Complete execution results on the two development paths at 1 ms.', 'tab:main_results', 'llrr',
                'Case & Method & Time (s) & \\shortstack{Maximum contour\\\\deviation (mm)}', rows,
                'Time is recalculated from retained complete trajectories, including startup, braking and terminal rest. Contour deviation and ordered traversal use the corresponding original geometric evaluations. The NNC adaptation has no feasible complete result and is reported separately.')
    write_table('generated_historical_utilization.tex',
                'Kinematic-limit utilization recalculated from complete retained trajectories at 1 ms (percent).',
                'tab:utilization', 'llrrr',
                'Case & Method & \\shortstack{Mean $V$\\\\utilization} & \\shortstack{95th-percentile $A$\\\\utilization} & \\shortstack{95th-percentile $J$\\\\utilization}', utilization,
                'Statistics include all equally spaced samples from startup through terminal rest. Percentiles describe the distribution, not peak utilization.')
    resolution = []
    for case, label in [('original', 'Long path'), ('dense', 'Dense case')]:
        values = [records[case, method, .0005]['complete_time_s'] for method in labels]
        resolution.append(label + ' & ' + ' & '.join(f'{value:.4f}' for value in values) + '\\\\')
    write_table('generated_historical_resolution.tex', 'Complete execution times at 0.5 ms on the development paths.',
                'tab:resolution', 'lrrr', 'Case & J-NNC (s) & Two-action (s) & Two-step (s)', resolution)
    manuscript = OUT / 'manuscript.tex'
    text = manuscript.read_text(encoding='utf-8')
    for label, filename in [('tab:main_results', 'generated_historical_main.tex'),
                            ('tab:utilization', 'generated_historical_utilization.tex'),
                            ('tab:resolution', 'generated_historical_resolution.tex')]:
        for match in re.finditer(r'\\begin\{table\}.*?\\end\{table\}', text, flags=re.S):
            if '\\label{' + label + '}' in match.group():
                text = text[:match.start()] + '\\input{' + filename + '}' + text[match.end():]
                break
    manuscript.write_text(text, encoding='utf-8')
    print('Generated three historical tables from results/historical_verified.json.')


if __name__ == '__main__':
    main()
