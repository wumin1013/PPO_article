"""Generate manuscript tables/figures only from a complete frozen v2 evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript'
NAMES = {'reserve_raster': 'Rounded raster', 'reserve_keyhole': 'Keyhole',
         'reserve_notched_profile': 'Notched profile',
         'reserve_asymmetric_profile': 'Asymmetric profile',
         'reserve_three_lobe': 'Three-lobe profile'}
SEEDS = [101, 211, 307, 401, 503]
COLORS = {'adaptive': '#246D91', 'fixed': '#AA6247'}


def write(name, contents):
    (OUT / name).write_text(contents, encoding='utf-8')


def table(caption, label, columns, header, rows, note=''):
    text = ('\\begin{table}[htbp]\n\\centering\n\\caption{' + caption + '}\n'
            '\\label{' + label + '}\n\\small\n\\begin{tabular}{' + columns + '}\n'
            '\\toprule\n' + header + r'\\' + '\n\\midrule\n')
    text += '\n'.join(' & '.join(row) + r'\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    if note:
        text += '\\par\\smallskip\\begin{minipage}{0.98\\linewidth}\\footnotesize ' + note + '\\end{minipage}\n'
    return text + '\\end{table}\n'


def fnum(x, digits=3):
    return '--' if x is None else f'{x:.{digits}f}'


def cp(k, n):
    return (0 if k == 0 else beta.ppf(.025, k, n-k+1),
            1 if k == n else beta.ppf(.975, k+1, n-k))


def main():
    source = ROOT / 'results/reserved_v2/summary.json'
    if not source.exists():
        raise RuntimeError('The reserved evaluation summary does not yet exist; no artifacts were written.')
    result = json.loads(source.read_text(encoding='utf-8'))
    if (result.get('valid_for_manuscript') is not True or result['recorded'] != result['expected']
            or result['expected'] != 80):
        raise RuntimeError('The evaluation is incomplete or inconsistent; no artifacts were written.')
    registration = json.loads((ROOT / 'prospective/reserved_programs/registration.json').read_text(encoding='utf-8'))
    paths = [r['id'] for r in registration['paths']]
    rows = result['results']
    if len(paths) != 5 or any(p not in NAMES for p in paths):
        raise RuntimeError('Unexpected program registration.')
    lookup = {(r['path'], r['method']): r for r in result['per_path']}
    for path in paths:
        for arm in ['adaptive', 'fixed']:
            if lookup[path, arm]['attempts'] != 5:
                raise RuntimeError('Each learning arm must retain five registered attempts per program.')
    receipts = []
    for seed in SEEDS:
        group = 'learning_001' if seed == 101 else 'learning_002'
        for arm in ['fixed', 'adaptive']:
            directory = ROOT / f'results/development_v2/{group}/paired/seed_{seed}/{arm}'
            r = json.loads((directory / 'receipt.json').read_text(encoding='utf-8'))
            if not r.get('complete') or r.get('error'):
                raise RuntimeError('A registered training run is incomplete.')
            receipts.append((seed, arm, directory, r))
    by_cell = {(r['path'], r['method'], r['seed'], r['dt_s']): r for r in rows}
    execution_records = {r['result']: json.loads((ROOT / r['result'].replace('\\', '/')).read_text(encoding='utf-8'))
                         for r in rows}
    resolution, changes, classification_same = [], [], True
    for path in paths:
        for method in ['adaptive', 'fixed', 'S2', 'SCMO']:
            candidates = [r for r in rows if r['path'] == path and r['method'] == method and r['dt_s'] == .0005]
            if len(candidates) != 1:
                raise RuntimeError('Expected one resolution record per program and method.')
            half = candidates[0]; full = by_cell[path, method, half['seed'], .001]
            classification_same &= full['success'] == half['success']
            delta = half['time_s'] - full['time_s'] if full['success'] and half['success'] else None
            if delta is not None:
                changes.append(abs(delta))
            resolution.append([NAMES[path], method.capitalize() if method in COLORS else method,
                               fnum(full['time_s'], 4) if full['success'] else 'Incomplete',
                               fnum(half['time_s'], 4) if half['success'] else 'Incomplete',
                               fnum(delta, 4)])
    # All checks above precede output mutations.
    programs = [[NAMES[r['id']], str(r['vertices'] - 1), fnum(r['length_mm']),
                 fnum(r['min_segment_mm'])] for r in registration['paths']]
    write('generated_reserved_programs.tex', table(
        'Five reserved synthetic program groups, evaluated as the stored polylines.',
        'tab:reserved_programs', 'lrrr',
        'Program & Blocks & Length (mm) & Shortest block (mm)', programs,
        'All segments and direct variants remain in one group. No production-CAM origin is asserted.'))
    learned = []
    for path in paths:
        for arm in ['adaptive', 'fixed']:
            r = lookup[path, arm]; lo, hi = cp(r['completed'], r['attempts'])
            learned.append([NAMES[path], arm.capitalize(), f"{r['completed']}/{r['attempts']}",
                            f'[{lo*100:.1f}, {hi*100:.1f}]', f"{fnum(r['mean_s'])} ({fnum(r['sd_s'])})"])
    write('generated_reserved_results.tex', table(
        'Evaluation of independently trained policies on reserved programs at 1 ms. Times are conditional on successful complete traversal.',
        'tab:reserved_results', 'llrrr',
        'Program & Preview & Successes & 95\\% interval (\\%) & Mean time (SD), s', learned,
        'Exact Clopper--Pearson success intervals use five independent training seeds per arm. Programs evaluated with the same seed are repeated measurements. '
        'SD is the sample standard deviation; incomplete attempts have no complete time.'))
    paired = {r['path']: r for r in result['paired']}
    difference = [[NAMES[path], str(paired[path]['n_complete_pairs']),
                   fnum(paired[path]['fixed_minus_adaptive_s']),
                   '[' + fnum(paired[path]['t95_low_s']) + ', ' + fnum(paired[path]['t95_high_s']) + ']'] for path in paths]
    write('generated_reserved_paired.tex', table(
        'Paired time difference on reserved programs (fixed minus adaptive preview).',
        'tab:reserved_paired', 'lrrr',
        'Program & Paired $n$ & Mean difference (s) & 95\\% interval (s)', difference,
        'Positive values favor adaptive preview. Student intervals are conditional on both arms completing and assume approximately '
        'normal paired differences. Five seeds cannot establish that assumption.'))
    baselines = []
    for path in paths:
        row = [NAMES[path]]
        for method in ['S2', 'SCMO']:
            r = lookup[path, method]
            row.extend([f"{r['completed']}/{r['attempts']}", fnum(r['mean_s'])])
        baselines.append(row)
    write('generated_reserved_baselines.tex', table(
        'Deterministic planners on the same reserved programs at 1 ms.',
        'tab:reserved_baselines', 'lrrrr',
        'Program & S2 $n$ & S2 time (s) & SCMO $n$ & SCMO time (s)', baselines,
        'Each planner is executed once per program under the common limits and complete-traversal rules.'))
    combined = []
    for path in paths:
        row = [NAMES[path]]
        for method in ['adaptive', 'fixed', 'S2', 'SCMO']:
            r = lookup[path, method]
            value = fnum(r['mean_s'])
            if method in COLORS and r['completed']:
                value += ' (' + fnum(r['sd_s']) + ')'
            row.append(r'\shortstack{' + value + r'\\' + f"{r['completed']}/{r['attempts']}" + '}')
        combined.append(row)
    write('generated_reserved_main.tex', table(
        'Complete execution on the five reserved synthetic programs at 1 ms.',
        'tab:reserved_main', 'lcccc',
        'Program & Adaptive preview & Fixed preview & S2 & SCMO', combined,
        'Each cell gives successful time in seconds (sample SD for learned policies), followed by completed/attempted runs. '
        'Five independent training seeds are evaluated per learning arm; deterministic planners run once. '
        'An incomplete attempt has no complete time.'))
    write('generated_reserved_resolution.tex', table(
        'Reserved-program resolution check. Learned policies use seed 101 with unchanged parameters.',
        'tab:reserved_resolution', 'llrrr',
        'Program & Method & 1 ms time (s) & 0.5 ms time (s) & Change (s)', resolution))
    costs = []
    for method in ['adaptive', 'fixed', 'S2', 'SCMO']:
        selected = [r for r in rows if r['method'] == method]
        durations = [execution_records[r['result']]['wall_s'] for r in selected]
        costs.append([method.capitalize() if method in COLORS else method, str(len(selected)),
                      fnum(sum(durations), 2), fnum(np.mean(durations), 2), fnum(max(durations), 2)])
    write('generated_reserved_execution_cost.tex', table(
        'Recorded command-generation and verification wall time across both resolutions, including incomplete attempts.',
        'tab:reserved_execution_cost', 'lrrrr',
        'Method & Attempts & Sum (s) & Mean (s) & Maximum (s)', costs,
        'Each timer covers command generation, record saving and independent acceptance checks on the shared host. '
        'Durations are neither pure offline planning time nor isolated online decision latency. '
        'Partly concurrent durations are not elapsed project time.'))
    failed = [r for r in rows if not r['success']]
    if not failed:
        write('generated_reserved_failures.tex', 'All 80 reserved-program evaluations completed under the common acceptance rules.\n')
    else:
        text = (r'\begin{longtable}{p{0.20\linewidth}llrp{0.36\linewidth}}' + '\n'
                r'\caption{All incomplete attempts in the reserved-program matrix. No prefix is assigned a complete time.}' + '\n'
                r'\label{tab:reserved_failures}\\' + '\n' + r'\toprule' + '\n'
                r'Program & Method & Seed & $h$ (ms) & Recorded cause\\' + '\n'
                r'\midrule\endfirsthead' + '\n' + r'\toprule' + '\n'
                r'Program & Method & Seed & $h$ (ms) & Recorded cause\\' + '\n'
                r'\midrule\endhead' + '\n')
        for r in failed:
            detail = execution_records[r['result']]
            cause = str(detail.get('error', r['reason'])).replace('_', ' ').replace(';', '; ')
            cause = cause.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')
            text += ' & '.join([NAMES[r['path']], r['method'], str(r['seed']) if r['seed'] is not None else '--',
                                f"{r['dt_s']*1000:g}", cause]) + r'\\' + '\n'
        write('generated_reserved_failures.tex', text + r'\bottomrule\end{longtable}' + '\n')
    budget = [[str(seed), arm.capitalize(), str(r['cost']['training_interactions']),
               str(r['cost']['ppo_optimizer_steps']), fnum(r['cost']['total_wall_s'], 1)]
              for seed, arm, _, r in receipts]
    write('generated_reserved_training_budget.tex', table(
        'Training-run receipts for the reserved-program experiment.',
        'tab:reserved_budget', 'llrrr',
        'Seed & Preview & Interactions & PPO steps & Run wall time (s)', budget,
        'Each run additionally uses 2000 fitting steps. Run wall time includes initialization, fitting, PPO and checkpoint validation. '
        'Durations are summed across partly concurrent processes and are not elapsed project time. Shared teacher search is separate.'))
    manifest = {'source': source.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'records': len(rows), 'per_arm_success': {arm: sum(lookup[path,arm]['completed'] for path in paths) for arm in COLORS},
                'run_wall_s': sum(r['cost']['total_wall_s'] for _, _, _, r in receipts),
                'resolution_classifications_equal': bool(classification_same), 'max_resolution_change_s': max(changes) if changes else None,
                'paired': result['paired']}
    write('reserved_artifact_receipt.json', json.dumps(manifest, indent=2) + '\n')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'], 'font.size':8,
                         'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False})
    labels = ['Raster', 'Keyhole', 'Notched', 'Asymmetric', 'Three-lobe']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={'width_ratios':[1.3,1]})
    for arm, offset in [('adaptive', -.13), ('fixed', .13)]:
        for i, path in enumerate(paths):
            records = [r for r in rows if r['path'] == path and r['method'] == arm and r['dt_s'] == .001]
            values = [r['time_s'] for r in records if r['success']]
            if values:
                axes[0].scatter(i + offset + np.linspace(-.04,.04,len(values)), values, s=14, color=COLORS[arm], alpha=.8)
                axes[0].plot([i+offset-.07,i+offset+.07],[np.mean(values)]*2,color=COLORS[arm],lw=2)
            axes[1].plot(i+offset, len(values)/len(records), 'o', color=COLORS[arm], label=arm.capitalize() if i==0 else None)
            axes[1].annotate(f'{len(values)}/{len(records)}', (i+offset, len(values)/len(records)),
                             xytext=(0,5),textcoords='offset points',ha='center',fontsize=7,color=COLORS[arm])
    for ax in axes:
        ax.set_xticks(range(5),labels,rotation=28,ha='right'); ax.grid(axis='y',alpha=.16)
    axes[0].set_ylabel('Complete time among successes (s)')
    axes[1].set_ylabel('Success fraction'); axes[1].set_ylim(-.04,1.2); axes[1].set_yticks(np.arange(0,1.01,.2)); axes[1].legend(loc='lower center',fontsize=7)
    axes[0].set_title('a  Individual seeds and means',loc='left',fontsize=9)
    axes[1].set_title('b  All five attempts per program',loc='left',fontsize=9)
    fig.tight_layout()
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / ('reserved_unseen_results.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2,2.55))
    for arm in COLORS:
        curves = [json.loads((folder/'training_curve.json').read_text(encoding='utf-8')) for _, a, folder, _ in receipts if a==arm]
        x = np.array([r['step'] for r in curves[0]])
        for ax, key in zip(axes,['mean_batch_reward','clip_fraction']):
            y = np.array([[r[key] for r in curve] for curve in curves])
            for values in y:
                ax.plot(x,values,color=COLORS[arm],lw=.6,alpha=.2)
            ax.plot(x,y.mean(0),color=COLORS[arm],lw=1.5,label=arm.capitalize())
    axes[0].set_ylabel('Mean rollout reward'); axes[1].set_ylabel('PPO clip fraction')
    for ax in axes:
        ax.set_xlabel('Environment decisions');ax.grid(alpha=.15);ax.legend(fontsize=7)
    axes[0].set_title('a  Five independent training runs',loc='left',fontsize=9)
    axes[1].set_title('b  Empirical update behavior',loc='left',fontsize=9)
    fig.tight_layout()
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / ('reserved_training_curves.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
