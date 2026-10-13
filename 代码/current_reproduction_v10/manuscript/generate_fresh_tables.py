"""Create manuscript tables solely from the completed frozen test summary."""
from pathlib import Path
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript'
LABELS = {'test_short_segments': 'Short blocks', 'test_acute_turns': 'Acute turns',
          'test_curvature': 'Varying curvature', 'test_line_arc': 'Mixed line--arc',
          'test_irregular_pocket': 'Closed pocket'}
METHODS = {'adaptive': 'Adaptive', 'fixed': 'Fixed', 'legacy_adaptive': 'Historical J-NNC',
           'legacy_fixed': 'Historical two-action', 'SCMO': 'Motion overlap', 'S2': 'Two-step'}


def write(name, text):
    (OUT / name).write_text(text + '\n', encoding='utf-8')


def table(caption, label, columns, header, rows, note=''):
    return '\n'.join([r'\begin{table}[htbp]', r'\centering',
                      r'\caption{' + caption + '}', r'\label{' + label + '}', r'\small',
                      r'\begin{tabular}{' + columns + '}', r'\toprule', header + r'\\',
                      r'\midrule', *rows, r'\bottomrule', r'\end{tabular}',
                      r'\par\smallskip\begin{minipage}{0.98\linewidth}\footnotesize ' + note + r'\end{minipage}' if note else '',
                      r'\end{table}'])


def main():
    data = json.loads((ROOT / 'results/summary/all_results.json').read_text(encoding='utf-8'))
    ledger = json.loads((ROOT / 'results/summary/expected_grid_ledger.json').read_text(encoding='utf-8'))
    if ledger['expected'] != ledger['recorded']:
        raise RuntimeError('Refuse to publish a partial frozen test matrix.')
    rows = []
    for r in data['per_path']:
        mean, sd = r['complete_time_mean_s'], r['complete_time_sd_s']
        timing = '--' if mean is None else f'{mean:.3f}' + (f' ({sd:.3f})' if sd is not None else ' (--)')
        interval = f"[{100*r['success_exact95_low']:.1f}, {100*r['success_exact95_high']:.1f}]"
        rows.append(f"{LABELS[r['path']]} & {METHODS[r['method']]} & {r['successes']}/{r['n']} & {interval} & {timing}" + r'\\')
    write('generated_fresh_results.tex', table(
        'Fresh independently initialized policies on five held-out synthetic programs at 1 ms. Time is conditional on successful complete traversal.',
        'tab:fresh_results', 'llrrr',
        r'Program & Preview & Successes & Exact 95\% interval (\%) & Mean time (SD), s', rows,
        r'Seeds: 101, 211, 307, 401 and 503. Intervals concern per-program success across independent training seeds; different programs run by the same seed are repeated measurements. SD is the sample standard deviation. Failed prefixes have no complete time.'))
    rows = []
    by_key = {(r['path'], r['method']): r for r in data['records']}
    for name in data['test_paths']:
        times = []
        for method in ('legacy_adaptive', 'legacy_fixed', 'S2', 'SCMO'):
            r = by_key[name, method]
            times.append(f"{r['time_s']:.3f}" if r['success'] else 'Failed')
        rows.append(LABELS[name] + ' & ' + ' & '.join(times) + r'\\')
    write('generated_comparator_results.tex', table(
        'Frozen historical systems and deterministic comparators on the same held-out programs at 1 ms. Entries are complete times in seconds; each deterministic policy or planner is executed once per program.',
        'tab:comparator_results', 'lrrrr',
        r'Program & Historical J-NNC & Historical two-action & Two-step & Motion overlap', rows))
    rows = []
    pairs = {r['path']: r for r in data['paired_differences']}
    for name in data['test_paths']:
        r = pairs.get(name)
        if r is None:
            rows.append(LABELS[name] + r' & 0 & -- & --\\')
            continue
        interval = '--' if r['t95_low_s'] is None else f"[{r['t95_low_s']:.3f}, {r['t95_high_s']:.3f}]"
        rows.append(f"{LABELS[name]} & {r['n_complete_pairs']} & {r['fixed_minus_adaptive_mean_s']:.3f} & {interval}" + r'\\')
    write('generated_paired_differences.tex', table(
        'Paired time difference (fixed minus adaptive preview). Positive values favor adaptive preview within the fresh training procedure.',
        'tab:paired_differences', 'lrrr',
        r'Program & Paired $n$ & Mean difference (s) & 95\% interval (s)', rows,
        r'Student intervals assume approximately normal paired seed differences and are conditional on both arms succeeding. The small sample cannot establish normality. They do not compare the historical optimized system with the fresh policies.'))
    rows = []
    for r in data['resolution']:
        times = [f"{r[k]:.4f}" if r[k] is not None else 'Failed' for k in ('time_1ms_s', 'time_half_ms_s')]
        delta = '--' if r['delta_half_minus_1ms_s'] is None else f"{r['delta_half_minus_1ms_s']:+.4f}"
        rows.append(f"{LABELS[r['path']]} & {METHODS[r['method']]} & " + ' & '.join(times) + ' & ' + delta + r'\\')
    write('generated_fresh_resolution.tex', '\n'.join([
        r'\small', r'\begin{longtable}{llrrr}',
        r'\caption{Held-out-program resolution check. Fresh policies use seed 101. Other methods are frozen historical systems or deterministic planners. All entries include the full terminal hold.}\label{tab:fresh_resolution}\\',
        r'\toprule Program & Method & 1 ms time (s) & 0.5 ms time (s) & Change (s)\\\midrule\endfirsthead',
        r'\toprule Program & Method & 1 ms time (s) & 0.5 ms time (s) & Change (s)\\\midrule\endhead',
        r'\midrule\multicolumn{5}{r}{Continued on next page}\\\endfoot',
        r'\bottomrule\endlastfoot', *rows, r'\end{longtable}', r'\normalsize']))
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    paths = {r['id']: r for r in manifest['paths']}
    rows = [f"{LABELS[name]} & {paths[name]['vertices']-1} & {paths[name]['length_mm']:.3f} & " + ('Closed' if paths[name]['closed'] else 'Open') + r'\\' for name in data['test_paths']]
    write('generated_test_programs.tex', table(
        'Frozen synthetic final-test programs. The line--arc case is evaluated as its declared chordal representation.',
        'tab:test_programs', 'lrrl', r'Program group & Blocks & Reference length (mm) & Topology', rows))
    for name in ('fresh_unseen_results', 'fresh_training_curves'):
        shutil.copy2(ROOT / 'figures' / (name + '.pdf'), OUT / (name + '.pdf'))
    print(f'Generated tables from {ledger["recorded"]} frozen records.')


if __name__ == '__main__':
    main()
