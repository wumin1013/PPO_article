"""Create measured runtime table without dropping the slower retained run."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    engineering = ROOT / 'engineering'
    first = json.loads((engineering / 'runtime_benchmark_first_run_summary.json').read_text(encoding='utf-8'))
    second = json.loads((engineering / 'runtime_benchmark.json').read_text(encoding='utf-8'))
    rows = []
    for key, label in [('policy_batch_one', 'Policy inference'),
                       ('kcm_saved_reachable_states', 'One KCM projection'),
                       ('full_decision_regular', 'Ordinary decision'),
                       ('full_decision_terminal_tail', 'Terminal decision')]:
        a, b = first[key], second[key]
        assert a['n'] == b['n']
        values = [f"{a[field]:.3f}/{b[field]:.3f}" for field in ('mean_ms', 'p99_ms', 'max_observed_ms')]
        rows.append(f"{label} & {a['n']} & " + ' & '.join(values) + '\\\\')
    text = r'''\begin{table}[htbp]
\centering
\caption{CPU wall-clock latency in two retained benchmark runs (run 1/run 2, in ms).}
\label{tab:runtime}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrrr}
\toprule
Component & $n$ per run & Mean & 99th percentile & Maximum observed\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.98\linewidth}\footnotesize
An ordinary decision integrates twenty 1 ms commands. A terminal decision consumes a variable braking/rest tail. Run 1 survives as a captured summary; its raw timing arrays were overwritten during a diagnostic-count correction. Run 2 retains individual samples. Both runs are reported to avoid selecting only the faster measurement.
\end{minipage}
\end{table}
'''
    (ROOT / 'manuscript/generated_runtime_table.tex').write_text(text, encoding='utf-8')
    print('Generated runtime table from both retained runs.')


if __name__ == '__main__':
    main()
