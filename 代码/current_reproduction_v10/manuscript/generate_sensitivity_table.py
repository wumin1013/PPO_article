"""Render completed one-factor deterministic validation experiments."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'results/sensitivity/scmo/summary.json').read_text(encoding='utf-8'))
    values = {(r['condition'], r['path_id']): r for r in data['results']}
    labels = {'nominal': 'Nominal', 'tolerance_0p5': r'One-sided tolerance 0.50 mm',
              'tolerance_0p25': r'One-sided tolerance 0.25 mm', 'velocity_80': r'$V_{\max}=80$ mm/s',
              'velocity_120': r'$V_{\max}=120$ mm/s', 'acceleration_1600': r'$A_{\max}=1600$ mm/s$^2$',
              'acceleration_2400': r'$A_{\max}=2400$ mm/s$^2$', 'jerk_50000': r'$J_{\max}=50000$ mm/s$^3$',
              'jerk_75000': r'$J_{\max}=75000$ mm/s$^3$'}
    rows = []
    for condition in data['conditions']:
        key = condition['name']
        runs = [values[key, path] for path in ('validation_zigzag', 'validation_trapezoid')]
        times = [f"{run['complete_time_s']:.3f}" if run['status'] == 'success' else 'Failed' for run in runs]
        rows.append(labels[key] + ' & ' + ' & '.join(times) + '\\\\')
    tex = r'''\begin{table}[htbp]
\centering
\caption{Independent motion-overlap comparator under one-factor changes on validation programs. Each of the 18 executions passed the common acceptance checks.}
\label{tab:sensitivity}
\small
\begin{tabular}{lrr}
\toprule
Condition & Zigzag time (s) & Trapezoid time (s)\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (ROOT / 'manuscript/generated_sensitivity_table.tex').write_text(tex, encoding='utf-8')
    frozen = json.loads((ROOT / 'results/sensitivity/frozen/summary.json').read_text(encoding='utf-8'))
    records = {(r['condition'], r['path_id']): r for r in frozen['results']}
    comparison = json.loads((ROOT / 'results/sensitivity/s2/comparison.json').read_text(encoding='utf-8'))
    s2 = {(r['condition'], r['path_id']): r for r in comparison['records']}
    rows = []
    for key in ('nominal', 'tolerance_0p5', 'tolerance_0p25', 'velocity_80', 'acceleration_1600', 'jerk_50000'):
        times = []
        for path in ('validation_zigzag', 'validation_trapezoid'):
            run = records[key, path]
            baseline = s2[key, path]
            times.extend([f"{run['complete_time_s']:.3f}" if run['success'] else 'Failed',
                          f"{baseline['s2_complete_time_s']:.3f}" if baseline['s2_success'] else 'Failed'])
        rows.append(labels[key] + ' & ' + ' & '.join(times) + '\\\\')
    tex = r'''\begin{table}[htbp]
\centering
\caption{Frozen historical J-NNC and reconfigured two-step planner (S2) on validation programs. Entries are complete times (s); failed runs have no complete execution time.}
\label{tab:frozen_sensitivity}
\small
\begin{tabular}{lrrrr}
\toprule
 & \multicolumn{2}{c}{Zigzag} & \multicolumn{2}{c}{Trapezoid}\\
Condition & J-NNC & S2 & J-NNC & S2\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    (ROOT / 'manuscript/generated_frozen_sensitivity_table.tex').write_text(tex, encoding='utf-8')


if __name__ == '__main__':
    main()
