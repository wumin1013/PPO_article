"""Read-only gate-study report from saved results; never trains or evaluates.

Python/matplotlib, editable 183-mm vector figures, existing paper style.
Evidence contract: show completion/failures before common-success time effects.
Statistics use independent training seeds within each reference path. Trace
samples describe one run only and are never treated as independent experiments.
No external visual precedent is claimed in this offline, existing-style task.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

import revision_plot as common

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT/'论文项目'
DEFAULT_RUNS = PAPER/'revision_20260908/preview_gate'
DEFAULT_DATA = ROOT/'PPO_project/data/preview_v2'
LABELS = {'fixed_short': 'Fixed 1.2 mm', 'fixed_mid': 'Fixed 2.3 mm', 'fixed_center': 'Fixed 3.0 mm',
          'fixed_long': 'Fixed 4.0 mm', 'selected_fixed': 'Selected fixed',
          'gate_only': 'Gate only', 'gate_residual': 'Gate + residual'}
COLORS = {'fixed_short': '#999999', 'fixed_mid': '#7a7a7a', 'fixed_center': '#505860',
          'fixed_long': '#343b42', 'selected_fixed': '#60666c',
          'gate_only': '#d78332', 'gate_residual': '#1f77b4'}


def mean_sd(values):
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return (float(np.mean(valid)), float(np.std(valid, ddof=1)) if len(valid)>1 else None) if valid else (None, None)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict,list)) else value
                             for key, value in row.items()})


class Inputs:
    """Hash the exact bytes read, including a snapshot of a still-running study."""
    def __init__(self):
        self.hashes = {}

    def name(self, path):
        try:
            return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return str(Path(path).resolve())

    def read(self, path):
        content = Path(path).read_bytes()
        self.hashes[self.name(path)] = hashlib.sha256(content).hexdigest()
        return content

    def sha(self, path):
        key = self.name(path)
        if key not in self.hashes:
            self.read(path)
        return self.hashes[key]

    def json(self, path):
        return json.loads(self.read(path).decode('utf-8'))

    def trace(self, path):
        with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as stored:
            return dict(zip(stored['columns'].tolist(), stored['data'].T))


def verify_frozen(base, data, protocol, inputs):
    frozen = inputs.json(base/'freeze.json')
    if frozen['protocol_sha256'] != inputs.sha(base/'protocol.json'):
        raise ValueError('Frozen protocol hash mismatch')
    if frozen['selection_sha256'] != inputs.sha(base/'selection.json'):
        raise ValueError('Frozen selection hash mismatch')
    if frozen['path_manifest_sha256'] != inputs.sha(data/'manifest.json'):
        raise ValueError('Frozen reference manifest hash mismatch')
    if frozen['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Frozen runtime source hash mismatch')
    expected = {f'formal/{method}_seed{seed}' for method in protocol['formal_variants'] for seed in protocol['formal_seeds']}
    if set(frozen['models']) != expected or set(frozen['configs']) != expected:
        raise ValueError('Freeze does not contain every registered formal model/configuration')
    for relative in sorted(expected):
        run = base/relative
        if inputs.sha(run/'checkpoint.pt') != frozen['models'][relative]:
            raise ValueError(f'Frozen checkpoint mismatch: {relative}')
        if inputs.sha(run/'config.yaml') != frozen['configs'][relative]:
            raise ValueError(f'Frozen configuration mismatch: {relative}')
        provenance = inputs.json(run/'provenance.json')
        training = inputs.json(run/'training_summary.json')
        if (provenance['source_sha256'] != frozen['source_sha256']
            or provenance['path_manifest_sha256'] != frozen['path_manifest_sha256']
            or training['steps'] != protocol['formal_steps']
            or training['checkpoint_sha256'] != frozen['models'][relative]):
            raise ValueError(f'Frozen training provenance mismatch: {relative}')
    return frozen


def discover(base, data, include_test=False):
    inputs = Inputs()
    protocol, path_manifest = inputs.json(base/'protocol.json'), inputs.json(data/'manifest.json')
    if protocol['manifest_sha256'] != inputs.sha(data/'manifest.json'):
        raise ValueError('Study reference manifest changed')
    selection = inputs.json(base/'selection.json') if (base/'selection.json').exists() else None
    if include_test:
        verify_frozen(base, data, protocol, inputs)
    rows, budgets, pending, expected = [], [], [], []
    for stage in ('pilot', 'formal'):
        methods = protocol[f'{stage}_variants']
        seeds = [protocol['pilot_seed']] if stage == 'pilot' else protocol['formal_seeds']
        budget = protocol[f'{stage}_steps']
        for method in methods:
            for seed in seeds:
                run = base/stage/f'{method}_seed{seed}'
                training_path = run/'training_summary.json'
                training = inputs.json(training_path) if training_path.exists() else None
                budgets.append(dict(stage=stage, method=method, seed=seed, planned_steps=budget,
                    completed_summary_steps=training['steps'] if training else None,
                    training_wall_s=training.get('training_wall_s') if training else None,
                    complete=bool(training and training['steps']==budget), run_dir=inputs.name(run)))
                splits = ['validation'] + (['test'] if include_test and stage=='formal' else [])
                for split in splits:
                    ids = [p['id'] for p in path_manifest['paths'] if p['split']==split]
                    regimes = ['nominal']
                    if run.exists():
                        regimes += [value for value in ('tight', 'low_jerk') if (run/f'evaluation_{split}_{value}').exists()]
                    for regime in regimes:
                        for path_id in ids:
                            key = dict(stage=stage, method=method, seed=seed, path_id=path_id, split=split, regime=regime)
                            expected.append(key)
                            result_path = run/f'evaluation_{split}_{regime}'/(path_id+'.json')
                            if not result_path.exists():
                                pending.append(dict(key, status='evaluation_pending'))
                                continue
                            try:
                                row = inputs.json(result_path)
                            except json.JSONDecodeError:
                                if split=='test':
                                    raise ValueError(f'Incomplete frozen result: {result_path}')
                                pending.append(dict(key, status='result_write_in_progress'))
                                continue
                            provenance = inputs.json(run/'provenance.json')
                            checkpoint_hash, config_hash = inputs.sha(run/'checkpoint.pt'), inputs.sha(run/'config.yaml')
                            effective = result_path.parent/'effective_config.yaml'
                            if (row['checkpoint_sha256'] != checkpoint_hash or row['config_sha256'] != config_hash
                                or provenance['config_sha256'] != config_hash
                                or row['code_sha256'] != protocol['source_sha256']
                                or provenance['source_sha256'] != protocol['source_sha256']
                                or provenance['path_manifest_sha256'] != inputs.sha(data/'manifest.json')
                                or row['effective_config_sha256'] != inputs.sha(effective)
                                or (row['variant'], row['seed'], row['path_id'], row['split'], row['regime'])
                                   != (method, seed, path_id, split, regime)):
                                raise ValueError(f'Stale or mismatched evaluation provenance: {result_path}')
                            if not training or row['training_steps'] != budget or training['steps'] != budget:
                                pending.append(dict(key, status='unregistered_or_partial_training_budget', source=inputs.name(result_path)))
                                continue
                            trace_path = result_path.with_suffix('.npz')
                            if not trace_path.exists():
                                raise ValueError(f'Missing raw trace for saved result: {result_path}')
                            rows.append(dict(row, **{k:v for k,v in key.items() if k not in row},
                                source=inputs.name(result_path), trace_file=inputs.name(trace_path),
                                _trace_path=str(trace_path), source_sha256=inputs.sha(result_path)))
    return dict(rows=rows, expected=expected, pending=pending, budgets=budgets,
                protocol=protocol, selection=selection, inputs=inputs)


def aggregate(rows, expected):
    output = []
    keys = sorted({(r['stage'],r['split'],r['regime'],r['method'],r['path_id']) for r in expected})
    for stage,split,regime,method,path_id in keys:
        match = lambda r: (r['stage'],r['split'],r['regime'],r['method'],r['path_id']) == (stage,split,regime,method,path_id)
        subset = [r for r in rows if match(r)]
        desired = sorted(r['seed'] for r in expected if match(r))
        received = sorted(r['seed'] for r in subset)
        if len(received) != len(set(received)):
            raise ValueError('Duplicate training seed/path result')
        successful = [r for r in subset if r['success']]
        entry = dict(stage=stage, split=split, regime=regime, method=method, path_id=path_id,
            n_evaluated=len(subset), n_expected=len(desired), evaluated_seeds=received,
            missing_seeds=sorted(set(desired)-set(received)), complete=received==desired,
            successes=len(successful), failures=len(subset)-len(successful),
            success_rate=len(successful)/len(subset) if subset else None,
            time_n_success=len(successful),
            failure_counts=dict(Counter(reason for row in subset for reason in row['failure_reasons'])),
            termination_counts=dict(Counter(row['termination_reason'] for row in subset)))
        metrics = {'success_time_s': [r['elapsed_time_s'] for r in successful],
            'ordered_progress': [r['ordered_progress'] for r in subset],
            'peak_error_mm': [r['contour_error_max_mm'] for r in subset],
            'mean_error_mm': [r.get('contour_error_mean_mm') for r in subset],
            'rmse_error_mm': [r.get('contour_error_rmse_mm') for r in subset],
            'actor_inference_median_us': [r.get('inference_median_us') for r in subset],
            'actor_inference_p95_us': [r.get('inference_p95_us') for r in subset]}
        for metric, values in metrics.items():
            entry[metric+'_mean'], entry[metric+'_sd'] = mean_sd(values)
        output.append(entry)
    return output


def paired_comparisons(rows, expected, selection=None):
    """Strict pairwise success; whole-suite effects only for complete seed suites."""
    summaries, paired_rows, suites = [], [], []
    groups = sorted({(r['stage'],r['split'],r['regime']) for r in expected})
    for stage,split,regime in groups:
        group = [r for r in rows if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        plan = [r for r in expected if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        paths, seeds = sorted({r['path_id'] for r in plan}), sorted({r['seed'] for r in plan})
        baselines = ['selected_fixed','gate_only'] if stage=='formal' else [
            selection['selected_fixed'] if selection else 'selected_fixed', 'gate_only']
        lookup = {(r['method'],r['seed'],r['path_id']):r for r in group}
        if len(lookup)!=len(group):
            raise ValueError('Duplicate method/seed/path pair')
        for baseline in baselines:
            complete_suite = []
            for path_id in paths:
                local, both, failures = [], [], Counter()
                for seed in seeds:
                    reference = lookup.get((baseline,seed,path_id))
                    candidate = lookup.get(('gate_residual',seed,path_id))
                    if reference is None or candidate is None:
                        continue
                    both.append(seed)
                    if not reference['success'] or not candidate['success']:
                        failures['both_failed' if not reference['success'] and not candidate['success']
                                 else 'baseline_failed' if not reference['success'] else 'candidate_failed'] += 1
                        continue
                    delta = float(reference['elapsed_time_s']-candidate['elapsed_time_s'])
                    pair = dict(stage=stage, split=split, regime=regime, baseline=baseline,
                        candidate='gate_residual', path_id=path_id, seed=seed,
                        baseline_time_s=reference['elapsed_time_s'], candidate_time_s=candidate['elapsed_time_s'],
                        baseline_minus_candidate_s=delta,
                        relative_reduction_percent=100*delta/reference['elapsed_time_s'] if reference['elapsed_time_s']>0 else None)
                    local.append(pair)
                    paired_rows.append(pair)
                delta_mean, delta_sd = mean_sd([r['baseline_minus_candidate_s'] for r in local])
                rel_mean, rel_sd = mean_sd([r['relative_reduction_percent'] for r in local])
                summaries.append(dict(stage=stage, split=split, regime=regime, baseline=baseline,
                    candidate='gate_residual', path_id=path_id, n_expected=len(seeds),
                    n_both_evaluated=len(both), n_common_success=len(local),
                    common_success_seeds=[r['seed'] for r in local], missing_pair_seeds=sorted(set(seeds)-set(both)),
                    noncommon_outcomes=dict(failures), baseline_minus_candidate_mean_s=delta_mean,
                    baseline_minus_candidate_sd_s=delta_sd, relative_reduction_mean_percent=rel_mean,
                    relative_reduction_sd_percent=rel_sd))
            for seed in seeds:
                subset = [r for r in paired_rows if (r['stage'],r['split'],r['regime'],r['baseline'],r['seed'])
                          ==(stage,split,regime,baseline,seed)]
                if {r['path_id'] for r in subset}==set(paths):
                    complete_suite.append(dict(seed=seed, path_ids=paths,
                        baseline_total_s=sum(r['baseline_time_s'] for r in subset),
                        candidate_total_s=sum(r['candidate_time_s'] for r in subset),
                        baseline_minus_candidate_total_s=sum(r['baseline_minus_candidate_s'] for r in subset)))
            mean,sd = mean_sd([r['baseline_minus_candidate_total_s'] for r in complete_suite])
            suites.append(dict(stage=stage, split=split, regime=regime, baseline=baseline,
                candidate='gate_residual', n_complete_suite_seeds=len(complete_suite),
                complete_suite_seed_details=complete_suite, baseline_minus_candidate_total_mean_s=mean,
                baseline_minus_candidate_total_sd_s=sd,
                unit='independent seed; both methods succeed on every planned path'))
    return summaries, paired_rows, suites


TRACE_FIELDS = {'D0_mm':'control_status__gate_prior_mm', 'preview_mm':'lookahead_mm',
    'residual_mm':'control_status__gate_residual_mm', 'geometry_decision_us':'control_status__gate_decision_us',
    'observation_features_us':'control_status__gate_observation_features_us',
    'q':'control_status__gate_action_q', 'range_over_pi':'control_status__gate_action_range_over_pi',
    'kcm_intervention':'kcm_intervention'}


def summarize_traces(rows, inputs):
    summaries = []
    for row in rows:
        trace = inputs.trace(Path(row['_trace_path']))
        result = {k:row[k] for k in ('stage','split','regime','method','seed','path_id','success','trace_file')}
        result['trace_sha256'] = inputs.sha(Path(row['_trace_path']))
        result['samples'] = len(trace['timestamp_s'])
        executed = np.asarray(trace['timestamp_s']) > 0.
        for metric, field in TRACE_FIELDS.items():
            values = trace.get(field, np.full(len(executed), np.nan))[executed]
            values = values[np.isfinite(values)]
            for suffix, function in [('mean',np.mean), ('min',np.min), ('max',np.max),
                                     ('median',np.median), ('p95',lambda v:np.percentile(v,95))]:
                result[metric+'_'+suffix] = float(function(values)) if len(values) else None
        result['reward_total'] = float(np.nansum(trace['reward']))
        for field, values in trace.items():
            if field.startswith('reward_components__r_'):
                result[field+'_sum'] = float(np.nansum(values))
        selected = trace.get('control_status__gate_selected_mm')
        result['preview_log_mismatch_max_mm'] = float(np.nanmax(np.abs(trace['lookahead_mm']-selected))) if selected is not None else None
        result['statistical_role'] = 'within-run descriptive summaries; time samples are not independent repeats'
        summaries.append(result)
    return summaries


def aggregate_trace_summaries(rows):
    results = []
    keys = sorted({(r['stage'],r['split'],r['regime'],r['method'],r['path_id']) for r in rows})
    for key in keys:
        group = [r for r in rows if tuple(r[k] for k in ('stage','split','regime','method','path_id'))==key]
        result = dict(zip(('stage','split','regime','method','path_id'),key))
        result['n_independent_seeds'] = len(group)
        fields = sorted({k for r in group for k,v in r.items() if k.endswith(('_mean','_sum','_median','_p95')) and (v is None or isinstance(v,(int,float)))})
        for field in fields:
            result[field+'_across_seeds_mean'],result[field+'_across_seeds_sd']=mean_sd([r.get(field) for r in group])
        results.append(result)
    return results


def short_path(path):
    for family,label in common.FAMILY_LABEL.items():
        if path.startswith(family):
            return label
    return path


def save_figure(fig, figures, name, products):
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf','svg','png'):
        output = figures/f'{name}.{suffix}'
        fig.savefig(output, dpi=240, bbox_inches='tight', facecolor='white')
        products.append(output)
    plt.close(fig)


def performance_figures(aggregates, figures, products):
    available = sorted({(r['stage'],r['split'],r['regime']) for r in aggregates if r['n_evaluated']})
    names = []
    for stage,split,regime in available:
        group=[r for r in aggregates if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        methods=list(dict.fromkeys(r['method'] for r in group))
        paths=sorted({r['path_id'] for r in group})
        lookup={(r['method'],r['path_id']):r for r in group}
        matrix=np.array([[lookup[(m,p)]['success_rate'] if lookup[(m,p)]['success_rate'] is not None else np.nan for p in paths] for m in methods])
        fig,axes=plt.subplots(1,2,figsize=(7.2,max(3.3,.4*len(methods)+1.2)),gridspec_kw={'width_ratios':[1,1.25]},constrained_layout=True)
        cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#eeeeee')
        axes[0].imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=1,aspect='auto')
        for i,m in enumerate(methods):
            for j,p in enumerate(paths):
                row=lookup[(m,p)]
                label=f"{row['successes']}/{row['n_evaluated']}" if row['n_evaluated'] else 'pending'
                if row['n_evaluated'] and not row['complete']:label+='*'
                axes[0].text(j,i,label,ha='center',va='center',color='white' if matrix[i,j]>.65 else '#222222',fontsize=7)
        axes[0].set_xticks(range(len(paths)),[short_path(p).replace(' ','\n',1) for p in paths])
        axes[0].set_yticks(range(len(methods)),[LABELS.get(m,m) for m in methods])
        axes[0].set_title('a  Strict success / evaluated seeds',loc='left',fontsize=9)
        markers=('o','s','^','D')
        for j,p in enumerate(paths):
            offset=(j-(len(paths)-1)/2)*.18
            for i,m in enumerate(methods):
                row=lookup[(m,p)]
                if row['success_time_s_mean'] is not None:
                    axes[1].errorbar(row['success_time_s_mean'],i+offset,xerr=row['success_time_s_sd'],
                        fmt=markers[j%len(markers)],color=COLORS.get(m,'#555555'),ms=4,capsize=2)
                elif row['n_evaluated']:
                    axes[1].scatter(0,i+offset,marker='x',s=16,color=COLORS.get(m,'#555555'))
        for j,p in enumerate(paths):
            axes[1].plot([],[],marker=markers[j%len(markers)],ls='',color='#444444',label=short_path(p))
        axes[1].set_ylim(len(methods)-.5,-.5);axes[1].set_yticks([])
        axes[1].set_xlabel('Successful execution time (s): mean ± seed SD')
        axes[1].set_title('b  Successes only; x = no successful run',loc='left',fontsize=9)
        axes[1].legend(loc='best',fontsize=6);axes[1].grid(axis='x',alpha=.15,ls=':')
        fig.suptitle(f'Gate study: {stage}, {split}, {regime}; * incomplete seed set',fontsize=9)
        name=f'preview_gate_{stage}_{split}_{regime}_performance'
        save_figure(fig,figures,name,products);names.append(name)
    return names


def paired_figures(summaries, pairs, figures, products):
    names=[]
    for split,regime in sorted({(r['split'],r['regime']) for r in summaries if r['stage']=='formal' and r['n_both_evaluated']}):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),constrained_layout=True)
        for ax,baseline in zip(axes,('selected_fixed','gate_only')):
            group=[r for r in summaries if (r['stage'],r['split'],r['regime'],r['baseline'])==('formal',split,regime,baseline)]
            for i,row in enumerate(group):
                selected=[p for p in pairs if (p['stage'],p['split'],p['regime'],p['baseline'],p['path_id'])==('formal',split,regime,baseline,row['path_id'])]
                offsets=np.linspace(-.08,.08,len(selected)) if len(selected)>1 else [0.]
                for p,offset in zip(selected,offsets):
                    ax.scatter(p['baseline_minus_candidate_s'],i+offset,s=12,color='#a8c4dc')
                if row['n_common_success']:
                    ax.errorbar(row['baseline_minus_candidate_mean_s'],i,xerr=row['baseline_minus_candidate_sd_s'],fmt='o',color='#1f77b4',capsize=2,ms=4)
                else:
                    ax.annotate('no common success',(0,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=6)
            ax.set_yticks(range(len(group)),[f"{short_path(r['path_id'])} (n={r['n_common_success']})" for r in group])
            ax.axvline(0,color='#999999',ls='--',lw=.7);ax.set_title(LABELS[baseline]+' minus residual')
            ax.set_xlabel('Paired time difference (s); positive favours residual')
        fig.suptitle(f'Formal common-success seed pairs: {split}, {regime}',fontsize=9)
        name=f'preview_gate_formal_{split}_{regime}_paired'
        save_figure(fig,figures,name,products);names.append(name)
    return names


def tex_report(aggregates, pairs, budgets, protocol, selection, names):
    text=['% Generated only from saved gate-study logs by preview_gate_report.py.',
          r'\section{Geometry-prior preview study}',r'\label{sec:preview_gate}', '',
          'The gate study retains the V2 physical-time reward, ordered coverage checks and kinematic module. '
          'All configurations receive the same two additional geometry features, giving 20 state inputs. '
          'The fixed 4.6 mm geometry window yields a turn-cancellation measure and the unwrapped tangent range divided by pi. '
          'The gate uses a 135-degree threshold; the residual policy adjusts its geometry prior by at most 0.3 mm. '
          'The smaller actor learning rate and GAE parameter are common training settings and are not attributed to preview adaptation.', '',
          f"The pilot budget is {protocol['pilot_steps']:,} steps for each of {len(protocol['pilot_variants'])} configurations, "
          f"using seed {protocol['pilot_seed']}. Formal runs use {protocol['formal_steps']:,} fresh-training steps "
          f"for each method and each of {len(protocol['formal_seeds'])} independent seeds. "
          'Statistics below use seeds within paths. Missing evaluations are pending and are not counted as either successes or failures. '
          'Standard deviations are sample standard deviations; they are undefined for fewer than two observations.', '']
    if selection:
        text += [f"Validation selected a uniform fixed distance of {selection['fixed_distance_mm']:g} mm. "
                 'This choice is shared by all formal paths.', '']
    for stage,split,regime in sorted({(r['stage'],r['split'],r['regime']) for r in aggregates if r['n_evaluated']}):
        subset=[r for r in aggregates if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        def formatted(mean,sd):
            return '--' if mean is None else f'${mean:.3f}$' if sd is None else f'${mean:.3f}\\pm{sd:.3f}$'
        body=[[short_path(r['path_id']),LABELS.get(r['method'],r['method']),
               f"{r['n_evaluated']}/{r['n_expected']}",f"{r['successes']}/{r['n_evaluated']}" if r['n_evaluated'] else 'pending',
               formatted(r['success_time_s_mean'],r['success_time_s_sd']),
               formatted(r['peak_error_mm_mean'],r['peak_error_mm_sd'])] for r in subset]
        text += common.table(f'Gate study: {stage}, {split}, {common.escaped(regime)}. Success requires the saved strict evaluator. '
            'Time uses successful runs only; peak error retains successful and failed runs. Evaluated/planned counts identify partial results.',
            f'tab:preview_gate_{stage}_{split}_{regime}', ['Path','Method','Evaluated/planned','Success','Success time/s','Peak error/mm'],body)
    formal=[r for r in aggregates if r['stage']=='formal']
    if not any(r['n_evaluated'] for r in formal):
        text += ['Formal five-seed evaluation is pending. No five-seed advantage is established by the available pilot results.', '']
    for split,regime in sorted({(r['split'],r['regime']) for r in pairs if r['stage']=='formal' and r['n_both_evaluated']}):
        group=[r for r in pairs if (r['stage'],r['split'],r['regime'])==('formal',split,regime)]
        body=[[short_path(r['path_id']),LABELS[r['baseline']],str(r['n_common_success']),
               '--' if r['baseline_minus_candidate_mean_s'] is None else f"{r['baseline_minus_candidate_mean_s']:.4f}",
               '--' if r['baseline_minus_candidate_sd_s'] is None else f"{r['baseline_minus_candidate_sd_s']:.4f}"] for r in group]
        text+=common.table('Formal paired time differences: baseline minus gate residual. Each pair uses the same independent seed and path, '
            'and both runs must succeed. Positive values favour the residual method. No common success is reported as unavailable, not zero.',
            f'tab:preview_gate_pairs_{split}_{regime}',['Path','Baseline','Common seeds','Mean/s','SD/s'],body)
    failed=[r for r in aggregates if r['failures']]
    if failed:
        body=[[common.escaped(r['stage']),short_path(r['path_id']),LABELS.get(r['method'],r['method']),
               common.escaped(r['split']+'/'+r['regime']),str(r['failures']),
               common.escaped('; '.join(f'{reason}: {count}' for reason,count in r['failure_counts'].items()))] for r in failed]
        text+=common.table('All failed evaluations remain. Multiple reasons may apply to one run; reason counts must not be summed as a failure rate.',
            'tab:preview_gate_failures',['Stage','Path','Method','Evaluation','Failed runs','Recorded reasons'],body)
    for name in names:
        text += [r'\begin{figure}[htbp]',r'\centering',
            f'\\includegraphics[width=0.98\\linewidth]{{figures/generated/{name}.pdf}}',
            r'\caption{Gate-study results from saved logs. Pilot results use one training seed; formal error bars use independent seeds. '
            r'Failure records and pending evaluations are retained. Time summaries on different success sets are descriptive; method effects use common-success pairs.}',
            f'\\label{{fig:{name}}}',r'\end{figure}','']
    text += ['Per-run preview priors, learned residuals, reward components and geometry computation costs are exported separately. '
             'These time-series summaries describe each run; interpolation samples are not statistical replicates. '
             'Actor inference cost and geometry/feature computation costs are recorded separately.', '']
    return '\n'.join(text)


def build_report(base=DEFAULT_RUNS, data=DEFAULT_DATA, output=None, figures=None, tex=None, include_test=False):
    output=output or base/'report'
    figures=figures or PAPER/'figures/generated'
    tex=tex or PAPER/'generated/preview_gate_results.tex'
    snapshot=discover(base,data,include_test)
    rows,expected,inputs=snapshot['rows'],snapshot['expected'],snapshot['inputs']
    aggregates=aggregate(rows,expected)
    paired,pair_rows,suites=paired_comparisons(rows,expected,snapshot['selection'])
    traces=summarize_traces(rows,inputs)
    trace_aggregate=aggregate_trace_summaries(traces)
    clean=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]
    output.mkdir(parents=True,exist_ok=True)
    products=[]
    tables={'raw_evaluations':clean,'performance_by_path_method':aggregates,'paired_by_path':paired,
            'common_success_pairs':pair_rows,'complete_suite_pairs':suites,'trace_descriptions':traces,
            'trace_across_training_seeds':trace_aggregate,'training_budget':snapshot['budgets'],'pending_evaluations':snapshot['pending']}
    for name,values in tables.items():
        for suffix in ('json','csv'):
            path=output/f'{name}.{suffix}'
            write_json(path,values) if suffix=='json' else write_csv(path,values)
            products.append(path)
    contract=dict(core_conclusion='Report completion and failures before assessing paired time effects; no assumed adaptive advantage',
        evidence_chain=['strict success over evaluated independent seeds','successful time mean and sample SD',
                        'paired same-seed same-path effects conditional on both succeeding'],
        archetype='quantitative grid',backend='Python/matplotlib only',export='183 mm SVG/PDF with editable text and PNG preview',
        visual_style='existing manuscript quantitative-panel style; no external precedent claimed in this offline task',
        n_definition='independent training seeds within each path; no interpolation-point iid assumption',
        partial_data='pending is neither failure nor success; empty pairs and n<2 SD are null',
        source_data='raw_evaluations, performance_by_path_method, common_success_pairs, trace_descriptions',
        default_scope='development validation only',frozen_test_included=include_test)
    write_json(output/'figure_contract.json',contract);products.append(output/'figure_contract.json')
    common.style()
    names=performance_figures(aggregates,figures,products)
    names+=paired_figures(paired,pair_rows,figures,products)
    tex.parent.mkdir(parents=True,exist_ok=True)
    tex.write_text(tex_report(aggregates,paired,snapshot['budgets'],snapshot['protocol'],snapshot['selection'],names),encoding='utf-8')
    products.append(tex)
    summary=dict(scope='frozen test plus development validation' if include_test else 'development validation only',
        evaluated_runs=len(rows),pending_evaluations=len(snapshot['pending']),strict_successes=sum(r['success'] for r in rows),
        failed_evaluations=sum(not r['success'] for r in rows),formal_complete=all(r['complete'] for r in aggregates if r['stage']=='formal'),
        no_significance_test='Only mean/sample SD and strict paired deltas are reported',
        paired_delta_sign='baseline minus gate_residual; positive favours gate_residual',figures=names)
    write_json(output/'summary.json',summary);products.append(output/'summary.json')
    lines=['# Gate-study saved-log report','',f"Scope: {summary['scope']}.",
        f"Evaluated: {len(rows)}; successes: {summary['strict_successes']}; failures: {summary['failed_evaluations']}; pending: {summary['pending_evaluations']}.",
        '', 'All means/SD use training seeds within paths. Empty paired sets are null; one-seed SD is null.',
        'Time effects use the same seed and path and require both methods to succeed. Complete-suite summaries require every planned path to succeed for both methods.',
        'Trace samples only describe a run. Reward-component sums, D0/residual summaries, actor inference and geometry/feature overhead remain separate.',
        '', '| Stage | Method | Path | Success / evaluated | Evaluated / planned | Success time mean / s | SD / s |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in aggregates:
        if r['n_evaluated']:
            lines.append(f"| {r['stage']} | {r['method']} | {r['path_id']} | {r['successes']}/{r['n_evaluated']} | "
                f"{r['n_evaluated']}/{r['n_expected']} | {r['success_time_s_mean']} | {r['success_time_s_sd']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');products.append(output/'report.md')
    inputs.read(Path(__file__));inputs.read(Path(common.__file__))
    write_json(output/'hash_manifest.json',dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        inputs=inputs.hashes,outputs={inputs.name(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in products},
        command='python PPO_project/tools/preview_gate_report.py'+(' --include-frozen-test' if include_test else ''),
        study_run_directory=inputs.name(base),read_only_inputs=True))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=DEFAULT_RUNS)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--figures',type=Path)
    parser.add_argument('--tex',type=Path)
    parser.add_argument('--include-frozen-test',action='store_true')
    args=parser.parse_args()
    print(json.dumps(build_report(args.runs,args.data,args.output,args.figures,args.tex,args.include_frozen_test),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
