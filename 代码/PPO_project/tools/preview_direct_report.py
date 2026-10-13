"""Read-only direct-distance PPO reporting, with frozen-test access opt-in.

Reuses saved-result provenance checks and seed/path statistics from the gate
report; it never imports a policy runner, trains, or executes a controller.
No geometry-rule runs are included. Direct policy learning and performance are
reported separately: nonzero gradients do not establish an efficiency benefit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

import preview_gate_report as saved

ROOT,PAPER,DATA=saved.ROOT,saved.PAPER,saved.DEFAULT_DATA
DEFAULT_RUNS=PAPER/'revision_20260908/preview_direct'
LABELS={**saved.LABELS,'direct':'Direct distance','direct_no_anchor':'Direct, no anchor',
    'selected_fixed_no_anchor':'Selected fixed, no anchor',
    **{f'fixed_{name}_no_anchor':f'Fixed {distance} mm, no anchor' for name,distance in [('short','1.2'),('mid','2.3'),('center','3.0'),('long','4.0')]}}
COLORS={**saved.COLORS,'direct':'#1f77b4','direct_no_anchor':'#d78332',
    'selected_fixed_no_anchor':'#967658',**{f'fixed_{name}_no_anchor':'#967658' for name in ('short','mid','center','long')}}
DIRECT_TRACE_FIELDS={'preview_mm':'lookahead_mm','feed_mm_s':'v_mm_s','q':'control_status__direct_observation_q',
    'range_over_pi':'control_status__direct_observation_range_over_pi','mapping_us':'control_status__direct_mapping_us',
    'geometry_features_us':'control_status__direct_observation_features_us','kcm_intervention':'kcm_intervention'}


def direct_pairs(rows,expected,selection=None):
    # Reuse only the generic numeric calculation, with one explicit candidate /
    # baseline pair at a time. Geometry-rule data are never read or substituted.
    result=([],[],[])
    for stage in sorted({r['stage'] for r in expected}):
        baseline='selected_fixed' if stage=='formal' else selection['selected_fixed'] if selection else 'selected_fixed'
        specs=[('direct',baseline,'matched_beta1_preview'),('direct','direct_no_anchor','physical_anchor_ablation')]
        if any(r['method']=='selected_fixed_no_anchor' or r['method'].startswith('fixed_') and r['method'].endswith('_no_anchor') for r in expected):
            baseline0='selected_fixed_no_anchor' if stage=='formal' else selection.get('selected_fixed_no_anchor','selected_fixed_no_anchor') if selection else 'selected_fixed_no_anchor'
            specs.append(('direct_no_anchor',baseline0,'matched_beta0_preview'))
        for candidate,reference,role in specs:
            def adapt(items):
                return [dict(r,method='gate_residual' if r['method']==candidate else 'selected_fixed') for r in items
                        if r['stage']==stage and r['method'] in (candidate,reference)]
            paired=saved.paired_comparisons(adapt(rows),adapt(expected),{'selected_fixed':'selected_fixed'})
            for output,collection in zip(result,paired):
                for row in collection:
                    if row['baseline']!='selected_fixed':continue
                    output.append(dict(row,baseline=reference,candidate=candidate,comparison_role=role))
    return result


def add_suite_confidence_intervals(suites):
    """Independent seeds are replicates; each seed first totals its fixed path set."""
    for row in suites:
        details=row['complete_suite_seed_details'];n=len(details)
        values=[r['baseline_minus_candidate_total_s'] for r in details]
        mean,sd=saved.mean_sd(values)
        margin=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
        row['baseline_minus_candidate_total_95ci_low_s']=mean-margin if margin is not None else None
        row['baseline_minus_candidate_total_95ci_high_s']=mean+margin if margin is not None else None
        percentages=[100*r['baseline_minus_candidate_total_s']/r['baseline_total_s'] for r in details if r['baseline_total_s']>0]
        row['relative_reduction_total_mean_percent'],row['relative_reduction_total_sd_percent']=saved.mean_sd(percentages)
        pmean,psd=row['relative_reduction_total_mean_percent'],row['relative_reduction_total_sd_percent']
        pmargin=float(student_t.ppf(.975,len(percentages)-1)*psd/np.sqrt(len(percentages))) if len(percentages)>1 else None
        row['relative_reduction_total_95ci_low_percent']=pmean-pmargin if pmargin is not None else None
        row['relative_reduction_total_95ci_high_percent']=pmean+pmargin if pmargin is not None else None
        row['confidence_unit']='Independent training seed after summing the predeclared path suite; both methods must succeed on every path'
        row['confidence_scope']='95% Student t interval conditional on complete-success suites; fixed tested geometries, not uncertainty over arbitrary unseen paths'
    return suites


def read_json_lines(path,inputs):
    content=inputs.read(path).decode('utf-8')
    records=[];partial=False
    lines=content.splitlines()
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:records.append(json.loads(line))
        except json.JSONDecodeError:
            if index==len(lines)-1 and not content.endswith('\n'):
                partial=True
            else:raise ValueError(f'Corrupt completed JSONL record: {path}:{index+1}')
    return records,partial


def append_saved_regression(snapshot,base,data):
    """Read separately labelled historical regression only when saved runs exist."""
    protocol,inputs=snapshot['protocol'],snapshot['inputs']
    manifests=inputs.json(data/'manifest.json')
    historical=manifests.get('historical_regression_paths',[])
    active_regimes=[regime for regime in ('nominal','tight','low_jerk')
        if any((base/'formal'/f'{method}_seed{seed}'/f'evaluation_regression_{regime}').exists()
               for method in protocol['formal_variants'] for seed in protocol['formal_seeds'])]
    if not active_regimes:return
    historical_config=PAPER/'generated/configs/full_method_snapshot.yaml'
    inputs.read(historical_config)
    registration_path=base/'regression_reference_provenance.json'
    registration=inputs.json(registration_path) if registration_path.exists() else None
    if registration:
        if (registration['protocol_sha256']!=inputs.sha(base/'protocol.json')
            or registration['source_sha256']!=protocol['source_sha256']
            or registration['historical_config_sha256']!=inputs.sha(historical_config)
            or {r['path_id'] for r in registration['references']}!=set(historical)):
            raise ValueError('Historical regression registration does not match frozen study/configuration')
        for reference in registration['references']:
            path=(base/reference['file']).resolve();path.relative_to(base.resolve())
            if inputs.sha(path)!=reference['sha256']:raise ValueError('Registered historical reference points changed')
    for method in protocol['formal_variants']:
        for seed in protocol['formal_seeds']:
            run=base/'formal'/f'{method}_seed{seed}'
            for regime in active_regimes:
                for path_id in historical:
                    key=dict(stage='formal',method=method,seed=seed,path_id=path_id,split='regression',regime=regime)
                    snapshot['expected'].append(key)
                    result_path=run/f'evaluation_regression_{regime}'/f'{path_id}.json'
                    if not result_path.exists():
                        snapshot['pending'].append(dict(key,status='evaluation_pending'));continue
                    try:row=inputs.json(result_path)
                    except json.JSONDecodeError:
                        snapshot['pending'].append(dict(key,status='result_write_in_progress'));continue
                    provenance=inputs.json(run/'provenance.json');training=inputs.json(run/'training_summary.json')
                    checkpoint=inputs.sha(run/'checkpoint.pt');config=inputs.sha(run/'config.yaml')
                    effective=result_path.parent/'effective_config.yaml'
                    if (row['checkpoint_sha256']!=checkpoint or row['config_sha256']!=config
                        or provenance['config_sha256']!=config or row['code_sha256']!=protocol['source_sha256']
                        or provenance['source_sha256']!=protocol['source_sha256']
                        or provenance['path_manifest_sha256']!=inputs.sha(data/'manifest.json')
                        or row['effective_config_sha256']!=inputs.sha(effective)
                        or (row['variant'],row['seed'],row['path_id'],row['split'],row['regime'])!=(method,seed,path_id,'regression',regime)):
                        raise ValueError(f'Stale or mismatched historical regression provenance: {result_path}')
                    if training['steps']!=protocol['formal_steps'] or row['training_steps']!=protocol['formal_steps']:
                        snapshot['pending'].append(dict(key,status='unregistered_or_partial_training_budget'));continue
                    trace=result_path.with_suffix('.npz')
                    if not trace.exists():raise ValueError(f'Missing saved historical trace: {trace}')
                    snapshot['rows'].append(dict(row,**{k:v for k,v in key.items() if k not in row},
                        source=inputs.name(result_path),trace_file=inputs.name(trace),_trace_path=str(trace),
                        source_sha256=inputs.sha(result_path),regression_reference_config=inputs.name(historical_config),
                        regression_reference_config_sha256_at_reporting=inputs.sha(historical_config),
                        regression_registration_sha256=inputs.sha(registration_path) if registration else None,
                        regression_registered_at_utc=registration['registered_at_utc'] if registration else None,
                        regression_scope='Previously developed historical path; not unseen evidence. Historical configuration is hashed at reporting time, not retroactively asserted as an evaluation-time hash.'))


def output_replacement_diagnostics(snapshot,base,data):
    """Separate counterfactual validation traces, never an independently trained method."""
    protocol,inputs=snapshot['protocol'],snapshot['inputs']
    selection=snapshot['selection'];pairs=[];summaries=[]
    validation_ids=[p['id'] for p in inputs.json(data/'manifest.json')['paths'] if p['split']=='validation']
    original={(r['method'],r['seed'],r['path_id']):r for r in snapshot['rows']
              if (r['stage'],r['split'],r['regime'])==('formal','validation','nominal')}
    for method in protocol['formal_variants']:
        if not method.startswith('direct'):continue
        for seed in protocol['formal_seeds']:
            run=base/'formal'/f'{method}_seed{seed}'
            for directory in sorted(run.glob('evaluation_validation_nominal_fixed_*_diagnostic')):
                registration=inputs.json(directory/'replacement_protocol.json')
                if (registration['role']!='Output replacement only; no retraining'
                    or registration['checkpoint_sha256']!=inputs.sha(run/'checkpoint.pt')
                    or registration['selection_sha256']!=inputs.sha(base/'selection.json')
                    or registration['study_source_sha256']!=protocol['source_sha256']
                    or registration['script_sha256']!=inputs.sha(directory/'preview_direct_swap.py')
                    or registration['executed_preview_mm']!=selection[registration['selection_distance_key']]):
                    raise ValueError(f'Stale output-replacement registration: {directory}')
                local=[]
                for path_id in validation_ids:
                    path=directory/f'{path_id}.json'
                    if not path.exists():continue
                    row=inputs.json(path);reference=original.get((method,seed,path_id))
                    if reference is None:raise ValueError('Output replacement requires its original validated policy run')
                    if (row['checkpoint_sha256']!=reference['checkpoint_sha256']
                        or row['config_sha256']!=reference['config_sha256']
                        or row['effective_config_sha256']!=inputs.sha(directory/'effective_config.yaml')
                        or row['code_sha256']!=protocol['source_sha256']
                        or (row['variant'],row['seed'],row['path_id'],row['split'],row['regime'])!=(method,seed,path_id,'validation','nominal')
                        or row['comparison_role']!='output replacement only; not independent fixed training'):
                        raise ValueError(f'Stale output-replacement result: {path}')
                    trace_path=path.with_suffix('.npz');trace=inputs.trace(trace_path)
                    executed=trace['timestamp_s']>0
                    distance_error=float(np.nanmax(np.abs(trace['lookahead_mm'][executed]-registration['executed_preview_mm'])))
                    if distance_error>1e-12:raise ValueError('Diagnostic did not execute its registered fixed replacement distance')
                    common=bool(reference['success'] and row['success'])
                    pair=dict(method=method,seed=seed,path_id=path_id,original_success=reference['success'],replacement_success=row['success'],
                        original_failure_reasons=reference['failure_reasons'],replacement_failure_reasons=row['failure_reasons'],
                        original_time_s=reference['elapsed_time_s'],replacement_time_s=row['elapsed_time_s'],common_success=common,
                        replacement_minus_original_s=float(row['elapsed_time_s']-reference['elapsed_time_s']) if common else None,
                        replaced_distance_mm=registration['executed_preview_mm'],distance_log_error_mm=distance_error,
                        original_source=reference['source'],replacement_source=inputs.name(path),replacement_trace=inputs.name(trace_path),
                        replacement_trace_sha256=inputs.sha(trace_path),diagnostic_protocol_sha256=inputs.sha(directory/'replacement_protocol.json'),
                        comparison_role='Same trained policy with preview output replaced; no retraining, not an independent fixed baseline')
                    pairs.append(pair);local.append(pair)
                common=[r for r in local if r['common_success']]
                summaries.append(dict(method=method,seed=seed,replaced_distance_mm=registration['executed_preview_mm'],
                    expected_paths=len(validation_ids),evaluated_paths=len(local),original_successes=sum(r['original_success'] for r in local),
                    replacement_successes=sum(r['replacement_success'] for r in local),common_success_paths=len(common),
                    original_common_time_sum_s=sum(r['original_time_s'] for r in common) if common else None,
                    replacement_common_time_sum_s=sum(r['replacement_time_s'] for r in common) if common else None,
                    replacement_minus_original_common_sum_s=sum(r['replacement_minus_original_s'] for r in common) if common else None,
                    complete_success_suite=len(common)==len(validation_ids),comparison_role='Output replacement only; no retraining',
                    statistics='One trained seed; path totals are descriptive and are not independent seed replicates'))
    return pairs,summaries


def serial_actor_benchmark(snapshot,base,data):
    """Validate saved serial timing against the complete model freeze and raw samples."""
    directory=base/'inference_benchmark';path=directory/'summary.json'
    if not path.exists():return None,[]
    inputs,protocol=snapshot['inputs'],snapshot['protocol']
    frozen=saved.verify_frozen(base,data,protocol,inputs)
    summary=inputs.json(path)
    if (summary['freeze_sha256']!=inputs.sha(base/'freeze.json')
        or summary['script_sha256']!=inputs.sha(directory/'preview_inference_benchmark.py')
        or summary['states_sha256']!=inputs.sha(directory/'shared_training_states.npy')
        or summary['raw_sha256']!=inputs.sha(directory/'latency_samples.npz')
        or summary['device']!='cpu' or summary['torch_threads']!=1):
        raise ValueError('Serial actor benchmark provenance or execution mode mismatch')
    expected={f'formal/{method}_seed{seed}':(method,seed) for method in protocol['formal_variants'] for seed in protocol['formal_seeds']}
    if len(summary['rows'])!=len(expected) or {r['run'] for r in summary['rows']}!=set(expected):
        raise ValueError('Serial benchmark requires each independently trained frozen model exactly once')
    with np.load(io.BytesIO(inputs.read(directory/'latency_samples.npz')),allow_pickle=False) as raw:
        for row in summary['rows']:
            run=row['run'];samples=np.asarray(raw[run.replace('/','__')])
            if ((row['method'],row['seed'])!=expected[run]
                or row['checkpoint_sha256']!=frozen['models'][run] or row['config_sha256']!=frozen['configs'][run]
                or len(samples)!=row['calls'] or len(samples)<100 or not np.isfinite(samples).all() or np.any(samples<=0)):
                raise ValueError(f'Invalid frozen serial timing samples: {run}')
            for key,value in [('median_us',np.median(samples)),('p95_us',np.percentile(samples,95)),('mean_us',np.mean(samples))]:
                if not np.isclose(row[key],value,rtol=0,atol=1e-9):raise ValueError(f'Serial timing summary differs from raw samples: {run}')
    costs=[]
    for method in protocol['formal_variants']:
        group=[r for r in summary['rows'] if r['method']==method]
        row=dict(method=method,n_independent_models=len(group),seeds=[r['seed'] for r in group],
            scope=summary['scope'],scheduling=summary['scheduling'],warmup_calls_per_policy=summary['warmup_calls_per_policy'])
        for metric in ('median_us','p95_us','mean_us'):
            row[metric+'_across_seeds_mean'],row[metric+'_across_seeds_sd']=saved.mean_sd([r[metric] for r in group])
        costs.append(row)
    return summary,costs


def evaluation_trace_summaries(rows,inputs):
    descriptions=[]
    for row in rows:
        trace=inputs.trace(Path(row['_trace_path']))
        actual=np.asarray(trace['timestamp_s'])>0.
        entry={k:row[k] for k in ('stage','split','regime','method','seed','path_id','success','trace_file')}
        for name,column in DIRECT_TRACE_FIELDS.items():
            values=trace.get(column,np.full(len(actual),np.nan))[actual]
            values=values[np.isfinite(values)]
            for statistic,op in [('mean',np.mean),('min',np.min),('max',np.max),('median',np.median),('p95',lambda x:np.percentile(x,95))]:
                entry[name+'_'+statistic]=float(op(values)) if len(values) else None
        entry['reward_total']=float(np.nansum(trace['reward']))
        for field,values in trace.items():
            if field.startswith('reward_components__r_'):entry[field+'_sum']=float(np.nansum(values))
        reward_fields=[field for field in trace if field.startswith('reward_components__r_')]
        reconstructed=np.nansum(np.stack([trace[field] for field in reward_fields]),axis=0) if reward_fields else np.full(len(actual),np.nan)
        entry['reward_component_reconstruction_max_error']=float(np.nanmax(np.abs(reconstructed[actual]-trace['reward'][actual]))) if actual.any() and reward_fields else None
        timestamps=np.asarray(trace['timestamp_s'])
        entry['sample_interval_min_s']=float(np.min(np.diff(timestamps))) if len(timestamps)>1 else None
        entry['sample_interval_max_s']=float(np.max(np.diff(timestamps))) if len(timestamps)>1 else None
        entry['elapsed_time_log_error_s']=float(abs((timestamps[-1]-timestamps[0])-row['elapsed_time_s']))
        entry['geometry_feature_time_alignment']='q/range at s_(t+1); D and reward from the transition ending at timestamp t+1'
        selected=trace.get('control_status__direct_selected_mm')
        entry['distance_mapping_log_error_max_mm']=float(np.nanmax(np.abs(trace['lookahead_mm']-selected))) if selected is not None else None
        entry['trace_sha256']=inputs.sha(Path(row['_trace_path']))
        entry['statistical_unit']='One independently trained seed/path run; trace samples are descriptive only'
        descriptions.append(entry)
    return descriptions


def training_diagnostics(base,protocol,inputs):
    epochs,exploration,block_rows,issues=[],[],[],[]
    for stage in ('pilot','formal'):
        seeds=[protocol['pilot_seed']] if stage=='pilot' else protocol['formal_seeds']
        for method in protocol[f'{stage}_variants']:
            for seed in seeds:
                run=base/stage/f'{method}_seed{seed}'
                update_path=run/'actor_update_diagnostics.jsonl'
                if not update_path.exists():continue
                provenance=inputs.json(run/'provenance.json')
                if (provenance['source_sha256']!=protocol['source_sha256']
                    or provenance['config_sha256']!=inputs.sha(run/'config.yaml')
                    or provenance['path_manifest_sha256']!=protocol['manifest_sha256']):
                    raise ValueError(f'Training diagnostic provenance mismatch: {run}')
                launcher_path=run/'launcher_provenance.json'
                launcher=None
                if launcher_path.exists():
                    launcher=inputs.json(launcher_path)
                    if (launcher['runtime_source_sha256']!=protocol['source_sha256']
                        or launcher['protocol_sha256']!=inputs.sha(base/'protocol.json')
                        or launcher['launcher_sha256']!=inputs.sha(run/'launcher_source.py')):
                        raise ValueError(f'Training CLI bridge provenance mismatch: {run}')
                    if (run/'launcher_invocations.jsonl').exists():inputs.read(run/'launcher_invocations.jsonl')
                updates,partial=read_json_lines(update_path,inputs)
                if partial:issues.append(dict(run=inputs.name(run),status='last optimizer log record still being written'))
                metadata=dict(stage=stage,method=method,seed=seed)
                last_epochs=[]
                for update in updates:
                    for epoch in update['epochs']:
                        epochs.append(dict(metadata,step=update['step'],transition_count=update['transition_count'],
                            invocation_id=update.get('invocation_id'),checkpoint_start_steps=update.get('checkpoint_start_steps'),
                            preview_latent_min=update.get('preview_latent_min'),preview_latent_max=update.get('preview_latent_max'),
                            preview_latent_clipped_fraction=update.get('preview_latent_clipped_fraction'),**epoch))
                    if update['epochs']:last_epochs.append(dict(update['epochs'][-1],step=update['step']))
                ds,steps=[],[]
                for path in sorted(run.glob('train_trace_*.npz')):
                    try:trace=inputs.trace(path)
                    except (ValueError,zipfile.BadZipFile,EOFError):
                        issues.append(dict(run=inputs.name(run),source=inputs.name(path),status='trace write incomplete'));continue
                    begin=int(path.stem.split('_')[-2])
                    values=trace['lookahead_mm'][1:]
                    ds.extend(values.tolist());steps.extend(range(begin+1,begin+len(values)+1))
                    # The 100-step nonoverlapping block summary is a display
                    # reduction only; all original samples remain in raw NPZ.
                    for start in range(0,len(values),100):
                        piece=values[start:start+100]
                        if not len(piece):continue
                        block_rows.append(dict(metadata,step_first=begin+start+1,step_last=begin+start+len(piece),
                            block_samples=len(piece),distance_mean_mm=float(np.mean(piece)),
                            distance_p05_mm=float(np.percentile(piece,5)),distance_p95_mm=float(np.percentile(piece,95)),
                            source_trace=inputs.name(path)))
                if len(steps)!=len(set(steps)):
                    issues.append(dict(run=inputs.name(run),status='overlapping trace step indices; exploration includes preserved duplicate segments'))
                d=np.asarray(ds,float);d=d[np.isfinite(d)]
                live_steps=max((r['step'] for r in updates),default=0)
                entry=dict(metadata,latest_logged_optimizer_step=live_steps,planned_steps=protocol[f'{stage}_steps'],
                    update_records=len(updates),sampled_training_steps=len(d),trace_scope='saved completed trace chunks; active unsaved episode may be absent',
                    trace_step_overlap=len(steps)-len(set(steps)),preview_is_active=method.startswith('direct'))
                entry['separate_cli_bridge_sha256']=launcher['launcher_sha256'] if launcher else None
                for name,op in [('min',np.min),('max',np.max),('mean',np.mean),('sd',np.std)]:
                    entry['sampled_distance_'+name+'_mm']=float(op(d)) if len(d) else None
                entry['observed_bound_span_fraction']=float((np.max(d)-np.min(d))/3.8) if len(d) else None
                entry['sample_fraction_outside_initial_plusminus_0p3mm']=float(np.mean(np.abs(d-3.)>.3)) if len(d) else None
                entry['sample_fraction_in_outer_5percent_of_bounds']=float(np.mean((d<.99)|(d>4.41))) if len(d) else None
                mean_min=[e['preview_mu_min'] for e in last_epochs if 'preview_mu_min' in e]
                mean_max=[e['preview_mu_max'] for e in last_epochs if 'preview_mu_max' in e]
                entry['policy_mean_distance_min_mm']=float(.8+3.8*min(mean_min)) if mean_min else None
                entry['policy_mean_distance_max_mm']=float(.8+3.8*max(mean_max)) if mean_max else None
                sigma=[e['preview_std_mean'] for e in last_epochs if 'preview_std_mean' in e]
                entry['mean_conditional_distance_sigma_mm']=float(3.8*np.mean(sigma)) if sigma else None
                for metric in ('preview_mu_weight_gradient_norm','preview_mu_bias_gradient','motion_anchor_kl','anchor_loss','actor_loss'):
                    values=[e[metric] for e in last_epochs if metric in e]
                    entry[metric+'_mean']=float(np.mean(values)) if values else None
                    entry[metric+'_abs_max']=float(np.max(np.abs(values))) if values else None
                entry['interpretation']='Observed sampled support and gradients are descriptive; they do not establish sufficient exploration or a causal reason for failure'
                entry['policy_moment_scope']='States in each rollout buffer, at its last logged PPO epoch before that epoch optimizer step; not an independent final-checkpoint state grid'
                exploration.append(entry)
    return epochs,exploration,block_rows,issues


def save(fig,figures,name,products):
    saved.save_figure(fig,figures,name,products)
    return name


def performance_figures(aggregates,figures,products,prefix='preview_direct'):
    names=[]
    for stage,split,regime in sorted({(r['stage'],r['split'],r['regime']) for r in aggregates if r['n_evaluated']}):
        group=[r for r in aggregates if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        paths=sorted({r['path_id'] for r in group});methods=list(dict.fromkeys(r['method'] for r in group))
        lookup={(r['method'],r['path_id']):r for r in group}
        fig,axes=plt.subplots(2,len(paths),figsize=(7.2,max(5.2,2.2+len(methods)*.5)),constrained_layout=True,squeeze=False)
        for column,path_id in enumerate(paths):
            for index,method in enumerate(methods):
                r=lookup[(method,path_id)];color=COLORS.get(method,'#555555')
                if r['n_evaluated']:
                    axes[0,column].barh(index,r['success_rate'],color=color,height=.65)
                    axes[0,column].text(1.04,index,f"{r['successes']}/{r['n_evaluated']}"+('*' if not r['complete'] else ''),va='center',fontsize=6)
                    if r['success_time_s_mean'] is not None:
                        axes[1,column].errorbar(r['success_time_s_mean'],index,xerr=r['success_time_s_sd'],fmt='o',color=color,capsize=2,ms=4)
                    else:axes[1,column].scatter(0,index,marker='x',color=color,s=16)
                else:axes[0,column].text(.05,index,'pending',color='#888888',va='center',fontsize=6)
            for row in range(2):
                ax=axes[row,column];ax.set_ylim(len(methods)-.5,-.5)
                ax.set_yticks(range(len(methods)),[LABELS.get(m,m) for m in methods] if column==0 else [])
                ax.grid(axis='x',alpha=.15,ls=':')
            axes[0,column].set_xlim(0,1.3);axes[0,column].set_xticks([0,.5,1.]);axes[0,column].set_xlabel('Strict success fraction')
            axes[0,column].set_title(saved.short_path(path_id))
            axes[1,column].set_xlabel('Success time (s), mean ± SD')
        fig.suptitle(f'Direct-distance PPO: {stage}, {split}, {regime}\n* incomplete seeds; x = no successful run',fontsize=9)
        names.append(save(fig,figures,f'{prefix}_{stage}_{split}_{regime}_performance',products))
    return names


def learning_figures(epochs,figures,products,prefix='preview_direct'):
    names=[]
    for stage in ('pilot','formal'):
        available=[r for r in epochs if r['stage']==stage and r['method'].startswith('direct')]
        if not available:continue
        seed=min(r['seed'] for r in available)
        fig,axes=plt.subplots(2,2,figsize=(7.2,4.8),constrained_layout=True)
        for method in ('direct','direct_no_anchor'):
            subset=[r for r in available if r['method']==method and r['seed']==seed]
            # Last recorded epoch at each step; all epochs and invocation IDs
            # are retained in CSV, including attempts after resumed checkpoints.
            indexed={r['step']:r for r in subset};rows=[indexed[k] for k in sorted(indexed)]
            if not rows:continue
            x=[r['step'] for r in rows];color=COLORS[method]
            if all('preview_mu_min' in r for r in rows):
                low=[.8+3.8*r['preview_mu_min'] for r in rows];high=[.8+3.8*r['preview_mu_max'] for r in rows]
                axes[0,0].plot(x,low,color=color,label=LABELS[method]);axes[0,0].plot(x,high,color=color,ls='--')
                axes[0,0].fill_between(x,low,high,color=color,alpha=.12)
            axes[0,1].plot(x,[3.8*r.get('preview_std_mean',np.nan) for r in rows],color=color)
            axes[1,0].plot(x,[r['preview_mu_weight_gradient_norm'] for r in rows],color=color)
            axes[1,1].plot(x,[r['motion_anchor_kl'] for r in rows],color=color)
        axes[0,0].axhline(3.,color='#999999',ls=':',lw=.7);axes[0,0].legend(fontsize=6)
        for ax,title,ylabel in zip(axes.ravel(),['a  Policy mean support over states','b  Conditional exploration scale',
                'c  Preview-head gradient before clipping','d  Physical-action KL (diagnostic if beta=0)'],
                ['Mean distance range (mm)','Distance sigma (mm)','Gradient norm','KL divergence']):
            ax.set_title(title,loc='left',fontsize=8);ax.set_xlabel('Training step');ax.set_ylabel(ylabel);ax.grid(alpha=.15,ls=':')
        axes[1,1].set_yscale('symlog',linthresh=1e-5)
        sigma_values=[3.8*r.get('preview_std_mean',0.) for r in available if r['seed']==seed]
        axes[0,1].set_ylim(0.,max(.05,1.25*max(sigma_values,default=.038)))
        axes[0,1].ticklabel_format(axis='y',style='plain',useOffset=False)
        fig.suptitle(f'Direct-policy learning diagnostics: {stage}, lowest available seed {seed}',fontsize=9)
        names.append(save(fig,figures,f'{prefix}_{stage}_learning_diagnostics',products))
    return names


def matched_pair_figures(summaries,pairs,figures,products,prefix='preview_direct'):
    """Show successful paired effects only beside their retained pair counts."""
    names=[]
    roles={'matched_beta1_preview':('Beta = 1','#1f77b4'),'matched_beta0_preview':('Beta = 0','#d78332')}
    groups=sorted({(r['stage'],r['split'],r['regime']) for r in summaries
                   if r['comparison_role'] in roles and r['n_both_evaluated']})
    for stage,split,regime in groups:
        scope=[r for r in summaries if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        available=[role for role in roles if any(r['comparison_role']==role and r['n_both_evaluated'] for r in scope)]
        fig,axes=plt.subplots(1,len(available),figsize=(7.2,3.1),constrained_layout=True,squeeze=False)
        for ax,role in zip(axes[0],available):
            title,color=roles[role];group=[r for r in scope if r['comparison_role']==role]
            for index,row in enumerate(group):
                local=[p for p in pairs if (p['stage'],p['split'],p['regime'],p['comparison_role'],p['path_id'])
                       ==(stage,split,regime,role,row['path_id'])]
                if row['n_common_success']:
                    offsets=np.linspace(-.08,.08,len(local)) if len(local)>1 else [0.]
                    ax.scatter([p['baseline_minus_candidate_s']*1000 for p in local],index+np.asarray(offsets),
                               color=color,alpha=.45,s=13,zorder=2)
                    sd=row['baseline_minus_candidate_sd_s']
                    ax.errorbar(row['baseline_minus_candidate_mean_s']*1000,index,
                        xerr=sd*1000 if sd is not None else None,fmt='D',color=color,capsize=2,ms=4,zorder=3)
                elif row['n_both_evaluated']:ax.scatter(0,index,marker='x',color=color,s=20)
            ax.axvline(0,color='#888888',lw=.7,ls=':')
            ax.set_yticks(range(len(group)),[f"{saved.short_path(r['path_id'])} [{r['n_common_success']}/{r['n_both_evaluated']}]" for r in group])
            ax.set_ylim(len(group)-.5,-.5);ax.set_title(title);ax.set_xlabel('Fixed minus direct time (ms)')
            ax.grid(axis='x',alpha=.15,ls=':')
        fig.suptitle(f'{stage}, {split}, {regime}: matched physical-anchor settings\n[common successes / evaluated pairs]; diamonds = mean ± SD; x = no common success',fontsize=8)
        names.append(save(fig,figures,f'{prefix}_{stage}_{split}_{regime}_paired',products))
    return names


def mechanism_figures(rows,selection,inputs,figures,output,products,prefix='preview_direct'):
    names=[]
    for stage in ('pilot','formal'):
        scope=[r for r in rows if r['stage']==stage and r['split']=='validation' and r['regime']=='nominal']
        if not any(r['method']=='direct' for r in scope):continue
        seed=min(r['seed'] for r in scope if r['method']=='direct')
        baseline='selected_fixed' if stage=='formal' else selection['selected_fixed'] if selection else None
        baseline0='selected_fixed_no_anchor' if stage=='formal' else selection.get('selected_fixed_no_anchor') if selection else None
        for path_id in sorted({r['path_id'] for r in scope if r['method']=='direct' and r['seed']==seed}):
            selected=[r for r in scope if r['seed']==seed and r['path_id']==path_id and r['method'] in (baseline,baseline0,'direct','direct_no_anchor')]
            fig,axes=plt.subplots(3,2,figsize=(7.2,6.8),constrained_layout=True)
            direct_trace=None
            for r in selected:
                trace=inputs.trace(Path(r['_trace_path']));t=trace['timestamp_s'];method=r['method'];color=COLORS[method]
                label=LABELS[method]+(' (failed)' if not r['success'] else '')
                axes[0,0].plot(t,trace['lookahead_mm'],color=color,label=label)
                axes[0,1].plot(t,trace['v_mm_s'],color=color)
                axes[1,1].plot(t,np.nancumsum(trace['reward']),color=color)
                axes[2,0].plot(t,trace['kcm_intervention'],color=color)
                if method=='direct':direct_trace=trace
                fields=['timestamp_s','lookahead_mm','v_mm_s','reward','kcm_intervention']
                fields += [k for k in trace if k.startswith(('control_status__direct_','reward_components__r_'))]
                series=output/f'mechanism_{stage}_{method}_seed{seed}_{path_id}.csv'
                saved.write_csv(series,[{field:float(trace[field][i]) if np.isfinite(trace[field][i]) else None for field in fields} for i in range(len(t))])
                products.append(series)
            if direct_trace is not None:
                d=direct_trace;t=d['timestamp_s']
                for field,label,style in [('control_status__direct_observation_q','q','-'),('control_status__direct_observation_range_over_pi','range/pi','--')]:
                    axes[1,0].plot(t,d[field],label=label,ls=style)
                for key in ('r_progress','r_track','r_smooth','r_dir','r_time','r_terminal'):
                    field='reward_components__'+key
                    if field in d:axes[2,1].plot(t,np.nancumsum(d[field]),label=key.removeprefix('r_'))
            titles=['a  Directly selected distance','b  Feed','c  Geometry is observation only',
                    'd  Cumulative episode reward','e  KCM intervention','f  Direct-policy reward components']
            units=['Distance (mm)','Feed (mm/s)','Normalized feature','Reward','Intervention','Cumulative reward']
            for ax,title,unit in zip(axes.ravel(),titles,units):
                ax.set_title(title,loc='left',fontsize=8);ax.set_ylabel(unit);ax.set_xlabel('Time (s)');ax.grid(alpha=.15,ls=':')
            for ax in (axes[0,0],axes[1,0],axes[2,1]):ax.legend(fontsize=6,loc='best')
            fig.suptitle(f'{stage}, seed {seed}: {saved.short_path(path_id)}; all displayed failures retained',fontsize=9)
            names.append(save(fig,figures,f'{prefix}_{stage}_{path_id}_mechanism',products))
    return names


def tex_report(aggregates,paired,exploration,protocol,selection,figures,prefix='preview_direct',replacement_summaries=(),suite_summaries=()):
    study_title='Directly learned preview distance'
    if 'shared_guidance' in protocol:study_title+=' with revised shared heading gain'
    if 'shared_ordered_coverage' in protocol:study_title='Direct preview with a refined common coverage certificate'
    if 'preview_exploration' in protocol:study_title='Direct preview with an interior initial exploration scale'
    initial_sigma=protocol.get('preview_exploration',{}).get('initial_sigma_distance_mm',.038)
    lines=['% Generated from saved direct-distance PPO logs; no hybrid trajectories.',f'\\section{{{study_title}}}',f'\\label{{sec:{prefix}}}', '',
        r'The policy directly selects $D_t=0.8+3.8\,\operatorname{clip}(u_{L,t},0,1)$ mm. '
        'Turn cancellation and unwrapped tangent range are shared observation features only; no geometry distance rule, teacher or behaviour-cloning target is used. '
        'Matched fixed-distance and direct configurations use the same KL coefficient on the first two normalized physical-control action distributions. '
        'The preview head is excluded. The unanchored direct configuration uses beta zero. '
        'Critic dropout is disabled during gradient-based updates for every configuration to remove stochastic TD/GAE noise.', '',
        f"The pilot budget is {protocol['pilot_steps']:,} steps per configuration with seed {protocol['pilot_seed']}; "
        f"the formal budget is {protocol['formal_steps']:,} fresh-training steps for each method and each of {len(protocol['formal_seeds'])} independent seeds. "
        'The initial mean distance is 3 mm, motivated by the saved fixed-distance developer grid; this is initialization, not pretraining.', '']
    if 'shared_guidance' in protocol:
        lines += [f"This separate candidate uses shared heading gain {protocol['shared_guidance']['gain_per_s']:g} s$^{{-1}}$ "
            f"(blend coefficient {protocol['shared_guidance']['tangent_blend']:g} per 1 ms step) in every fixed and direct configuration. "
            'Its results remain separate from the preceding direct-distance pilots; any benefit of the shared gain is not attributed to preview learning.', '']
    if 'shared_ordered_coverage' in protocol:
        coverage=protocol['shared_ordered_coverage']
        lines += [f"The common ordered-witness spacing is {coverage['witness_spacing_mm']:g} mm "
            f"and the nominal matching radius is {coverage['nominal_matching_radius_mm']:g} mm "
            f"for unchanged one-sided physical tolerance {coverage['nominal_one_sided_tolerance_mm']:g} mm. "
            'The finer sufficient certificate is used in both online reward/termination and offline evaluation. '
            'This is a shared fixed/direct change with new training, not a relabelling of old models or an exact continuous Frechet certificate.', '']
    lines += ['Success and failure labels below are retained exactly from the archived evaluation protocol. '
        'Failure of its conservative ordered-coverage certificate does not by itself prove an actual path shortcut; '
        'certificate-resolution audits must be reported separately without relabelling these historical runs.', '']
    if selection:lines += [f"The uniform fixed distance selected on validation is {selection['fixed_distance_mm']:g} mm.",'']
    if selection and 'fixed_distance_no_anchor_mm' in selection:
        lines += [f"Within beta zero, the independently selected uniform fixed distance is {selection['fixed_distance_no_anchor_mm']:g} mm. "
            'Preview comparisons are matched by beta; the direct versus unanchored-direct comparison diagnoses the physical-action anchor, not the contribution of preview.', '']
    if 'preview_exploration' in protocol:
        lines += [f"Only the third head initial standard deviation changes from 0.01 to {protocol['preview_exploration']['initial_std_u']:g}, "
            f"corresponding to initial distance standard deviation {initial_sigma:g} mm. "
            'This avoids the observed float32 hard-clamp zero-gradient region. Fixed methods receive the same third-head initialization but exclude the inactive dimension from likelihood and entropy. '
            'The two physical standard deviations retain the preceding common initialization and its known initial zero-gradient limitation.', '']
    def value(mean,sd):
        return '--' if mean is None else f'${mean:.3f}$' if sd is None else f'${mean:.3f}\\pm{sd:.3f}$'
    for stage,split,regime in sorted({(r['stage'],r['split'],r['regime']) for r in aggregates if r['n_evaluated']}):
        subset=[r for r in aggregates if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        body=[[saved.short_path(r['path_id']),LABELS.get(r['method'],r['method']),f"{r['n_evaluated']}/{r['n_expected']}",
            f"{r['failures']}/{r['n_evaluated']}" if r['n_evaluated'] else 'pending',value(r['success_time_s_mean'],r['success_time_s_sd']),
            value(r['peak_error_mm_mean'],r['peak_error_mm_sd'])] for r in subset]
        lines += saved.common.table(f'Direct-distance study: {stage}, {split}, {saved.common.escaped(regime)}. '
            'Means and sample SD use independent training seeds within each path. Time retains successful runs only; errors retain failed runs. Missing evaluations remain pending.',
            f'tab:{prefix}_{stage}_{split}_{regime}',['Path','Method','Evaluated/planned','Failures','Success time/s','Peak error/mm'],body)
    formal=[r for r in aggregates if r['stage']=='formal']
    if not any(r['n_evaluated'] for r in formal):lines += ['Formal five-seed evaluation remains pending; pilot results do not establish a repeated efficiency advantage.','']
    for stage,split,regime in sorted({(r['stage'],r['split'],r['regime']) for r in paired if r['n_both_evaluated']}):
        group=[r for r in paired if (r['stage'],r['split'],r['regime'])==(stage,split,regime)]
        body=[[saved.short_path(r['path_id']),LABELS[r['baseline']],LABELS[r['candidate']],str(r['n_common_success']),
            value(r['baseline_minus_candidate_mean_s'],r['baseline_minus_candidate_sd_s'])] for r in group]
        lines+=saved.common.table(f'{stage.capitalize()} strict common-success paired time differences: baseline minus candidate. '
            'Both runs must succeed on the same path with the same training seed. Positive values favour the candidate; an empty pair set is unavailable, not zero. '
            'Fixed versus direct comparisons are matched by anchor coefficient.',
            f'tab:{prefix}_pairs_{stage}_{split}_{regime}',['Path','Baseline','Candidate','Paired seeds','Time difference/s'],body)
    suite_rows=[r for r in suite_summaries if r['stage']=='formal' and r['comparison_role'].startswith('matched_') and r['n_complete_suite_seeds']]
    if suite_rows:
        def interval(r):
            lo,hi=r['baseline_minus_candidate_total_95ci_low_s'],r['baseline_minus_candidate_total_95ci_high_s']
            return '--' if lo is None else f'$[{lo:.3f},{hi:.3f}]$'
        body=[[saved.common.escaped(r['split']+'/'+r['regime']),LABELS[r['candidate']],str(r['n_complete_suite_seeds']),
            value(r['baseline_minus_candidate_total_mean_s'],r['baseline_minus_candidate_total_sd_s']),interval(r)] for r in suite_rows]
        lines += saved.common.table('Whole-suite fixed-minus-direct time differences. Each independently trained seed first sums its predeclared path set; only seeds with both methods successful on every path enter the conditional comparison. '
            'Mean and sample SD, followed by a 95 percent Student t confidence interval, use seeds as replicates. These intervals concern training randomness on the tested path set, not arbitrary-path uncertainty.',
            f'tab:{prefix}_seed_suite',['Evaluation','Candidate','Complete seeds','Total difference / s','95 percent CI / s'],body)
    lines += ['Nonzero preview-head gradients and changes in the policy mean demonstrate participation in optimization, not an established performance benefit. '
              'Observed exploration support, conditional standard deviation and sampled distance histories are exported from training logs. '
              f'The 1 ms sampling interval and initial distance standard deviation of {initial_sigma:g} mm determine the initial local exploration scale; '
              'these descriptive records alone cannot prove that insufficient exploration caused a failure.', '']
    if replacement_summaries:
        body=[[LABELS[r['method']],str(r['seed']),f"{r['original_successes']}/{r['evaluated_paths']}",
            f"{r['replacement_successes']}/{r['evaluated_paths']}",str(r['common_success_paths']),
            value(r['replacement_minus_original_common_sum_s'],None)] for r in replacement_summaries]
        lines+=saved.common.table('Validation output-replacement diagnostic, without retraining. '
            'The original checkpoint is reused and its preview output is replaced by the validation-selected fixed distance from the first action; the initial observation is preserved. '
            'Time is replacement minus original, summed only over common-success paths. This diagnostic is separate from independently trained fixed baselines and has one trained seed per row.',
            f'tab:{prefix}_output_replacement',['Policy','Seed','Original S/E','Replacement S/E','Common paths','Time sum difference / s'],body)
    for name in figures:
        lines += [r'\begin{figure}[htbp]',r'\centering',f'\\includegraphics[width=0.98\\linewidth]{{figures/generated/{name}.pdf}}',
            r'\caption{Saved direct-distance study diagnostics. Statistical replicates are independently trained seeds within paths; time-series samples and optimizer epochs are descriptive only. Failed runs remain visible.}',
            f'\\label{{fig:{name}}}',r'\end{figure}','']
    return '\n'.join(lines)


def compact_paper(snapshot,aggregates,paired,trace_summaries,figures,prefix):
    """Short manuscript entry; held-out claims require saved frozen test rows."""
    protocol=snapshot['protocol'];selection=snapshot['selection'];rows=snapshot['rows']
    methods=protocol['formal_variants']
    primary=[r for r in aggregates if (r['stage'],r['split'],r['regime'])==('formal','test','nominal')]
    lines=['% Compact generated entry; complete data remain in the companion full report.',
        r'\subsection{Frozen simulation evaluation}',f'\\label{{sec:{prefix}_paper}}','',
        f"The registered study contains {len(protocol['pilot_variants'])} pilot configurations at {protocol['pilot_steps']:,} steps each, "
        f"followed by {len(methods)} methods with {len(protocol['formal_seeds'])} independent training seeds at {protocol['formal_steps']:,} steps per seed. "
        'Fixed distances are selected on validation within the corresponding physical-action anchor setting. '
        'Pilot, validation, regression and numerical-diagnostic records are retained in the full reproducibility report.', '']
    if selection:
        message=f"The selected uniform fixed distance is {selection['fixed_distance_mm']:g} mm at beta one"
        if 'fixed_distance_no_anchor_mm' in selection:message+=f" and {selection['fixed_distance_no_anchor_mm']:g} mm at beta zero"
        lines += [message+'. No per-test-path fixed-distance selection is performed.','']
    if not primary or not any(r['n_evaluated'] for r in primary):
        return '\n'.join(lines+['Frozen held-out test results are pending. Development validation results are not substituted for this evaluation.',''])
    if any(not r['complete'] for r in primary):
        lines += ['This is an incomplete test snapshot; missing registered seed/path evaluations remain pending.','']
    def number(mean,sd=None,digits=3):
        if mean is None:return '--'
        return f'${mean:.{digits}f}$' if sd is None else f'${mean:.{digits}f}\\pm{sd:.{digits}f}$'
    def name(method):return 'Direct' if method.startswith('direct') else 'Fixed'
    def beta(method):return '0' if method.endswith('_no_anchor') else '1'
    body=[[saved.short_path(r['path_id']),name(r['method']),beta(r['method']),f"{r['successes']}/{r['n_evaluated']}",
        number(r['success_time_s_mean'],r['success_time_s_sd']),number(r['peak_error_mm_mean'],r['peak_error_mm_sd']),
        number(r['rmse_error_mm_mean'],r['rmse_error_mm_sd'])] for r in primary]
    lines += saved.common.table('Frozen nominal test, with independent training seeds within each path. '
        'S/E denotes successful/evaluated runs. '+
        ('Registered runs not yet evaluated remain pending in the full report. ' if any(not r['complete'] for r in primary) else '')+
        'Time uses successful runs only; contour-error summaries include failures. Values are mean and sample SD.',
        f'tab:{prefix}_paper_test',['Path','Method',r'$\beta$','S/E','Time / s','Peak error / mm','RMSE / mm'],body)
    relevant=[r for r in paired if (r['stage'],r['split'],r['regime'])==('formal','test','nominal') and r['comparison_role'].startswith('matched_')]
    body=[[saved.short_path(r['path_id']),beta(r['candidate']),f"{r['n_common_success']}/{r['n_both_evaluated']}",
        number(r['baseline_minus_candidate_mean_s'],r['baseline_minus_candidate_sd_s']),
        number(r['relative_reduction_mean_percent'],r['relative_reduction_sd_percent'],2)] for r in relevant]
    lines += saved.common.table('Matched-anchor common-success time comparisons. '
        'Only the same seed and path with both methods successful forms a pair. Positive fixed-minus-direct differences favour direct preview; empty pairs are unavailable. '
        'These conditional effects must be read with the failure counts in the nominal test table.',f'tab:{prefix}_paper_pairs',
        ['Path',r'$\beta$','Common/evaluated pairs',r'$T_{\rm fixed}-T_{\rm direct}$ / s','Reduction / percent'],body)
    suites=[r for r in snapshot.get('suite_summaries',[]) if (r['stage'],r['split'],r['regime'])==('formal','test','nominal') and r['comparison_role'].startswith('matched_')]
    if suites:
        body=[]
        for r in suites:
            lo,hi=r['relative_reduction_total_95ci_low_percent'],r['relative_reduction_total_95ci_high_percent']
            interval='--' if lo is None else f'$[{lo:.2f},{hi:.2f}]$'
            body.append([beta(r['candidate']),f"{r['n_complete_suite_seeds']}/{len(protocol['formal_seeds'])}",
                number(r['baseline_minus_candidate_total_mean_s'],r['baseline_minus_candidate_total_sd_s']),
                number(r['relative_reduction_total_mean_percent'],r['relative_reduction_total_sd_percent'],2),interval])
        lines+=saved.common.table('Each independent training seed first totals the predeclared test path set. '
            'Only whole suites successful for both methods enter this conditional comparison. The 95 percent Student t interval and mean/sample SD use independent seed totals; path points and multiple paths from the same seed are not treated as independent training repeats.',
            f'tab:{prefix}_paper_suite',[r'$\beta$','Complete/planned seeds','Fixed minus direct total / s','Reduction / percent','95 percent CI / percent'],body)
    if not any(r['failures'] for r in primary):
        lines += [f"Across all methods, {sum(r['failures'] for r in primary)}/{sum(r['n_evaluated'] for r in primary)} nominal test runs failed.",'']
    for method in methods if any(r['failures'] for r in primary) else []:
        group=[r for r in primary if r['method']==method]
        reasons={}
        for row in group:
            for reason,count in row['failure_counts'].items():reasons[reason]=reasons.get(reason,0)+count
        failed=sum(r['failures'] for r in group);evaluated=sum(r['n_evaluated'] for r in group)
        detail='; '.join(f'{saved.common.escaped(k)}: {v}' for k,v in sorted(reasons.items()))
        lines += [f"{name(method)} at beta {beta(method)} has {failed}/{evaluated} failed nominal test runs"+
            (f" ({detail}; reason counts can overlap)." if detail else '.'),'']
    lines += ['A failed sufficient ordered-coverage certificate is retained as a failure under this protocol; it alone does not establish an actual shortcut. '
        'Constraint claims are limited to the checks implemented by the saved discrete evaluator. No machine tracking or surface-quality result is inferred.','']
    for r in snapshot.get('output_replacement_summaries',[]):
        delta=r['replacement_minus_original_common_sum_s']
        if delta is not None:
            lines += [f"A separate validation diagnostic for {name(r['method'])} at beta {beta(r['method'])}, seed {r['seed']}, "
                f"replaced its preview output by {r['replaced_distance_mm']:g} mm without retraining: "
                f"the original and replacement succeeded on {r['original_successes']}/{r['evaluated_paths']} and {r['replacement_successes']}/{r['evaluated_paths']} paths. "
                f"Across their {r['common_success_paths']} common-success paths, replacement minus original total time was {delta:+.3f} s. "
                'This is a same-checkpoint diagnostic, not an independently trained fixed baseline.','']
    body=[]
    serial={r['method']:r for r in snapshot.get('serial_actor_costs',[])}
    for method in methods:
        budgets=[r for r in snapshot['budgets'] if r['stage']=='formal' and r['method']==method and r['complete']]
        wall,wall_sd=saved.mean_sd([r['training_wall_s'] for r in budgets])
        evals=[r for r in rows if (r['stage'],r['split'],r['regime'],r['method'])==('formal','test','nominal',method)]
        traces=[r for r in trace_summaries if (r['stage'],r['split'],r['regime'],r['method'])==('formal','test','nominal',method)]
        def extent(values):
            finite=[v for v in values if v is not None and np.isfinite(v)]
            return '--' if not finite else f'${min(finite):.1f}$--${max(finite):.1f}$'
        timing=serial.get(method)
        actor=number(timing['median_us_across_seeds_mean'],timing['median_us_across_seeds_sd'],1) if timing else extent([r.get('inference_median_us') for r in evals])
        actor_p95=number(timing['p95_us_across_seeds_mean'],timing['p95_us_across_seeds_sd'],1) if timing else extent([r.get('inference_p95_us') for r in evals])
        body.append([name(method),beta(method),number(wall,wall_sd,1),actor,actor_p95,extent([r.get('mapping_us_median') for r in traces]),
            extent([r.get('geometry_features_us_median') for r in traces])])
    lines += saved.common.table('Computation for this candidate. Training wall time is mean and sample SD across completed independent seeds; parallel-worker wall times are not summed into path execution time. '+
        ('Actor median and p95 are from the separately verified serial CPU benchmark with shared training-path inputs; mean/sample SD are across independent models. ' if serial else
         'Actor values are minimum--maximum per-run summaries from evaluation traces; the separate serial benchmark remains pending. ')+
        'Mapping and feature columns show minimum--maximum per-run medians from evaluation traces, in microseconds. The actor benchmark excludes geometry, KCM, logging and controller latency; these entries are not an end-to-end real-time guarantee.',
        f'tab:{prefix}_paper_cost',['Method',r'$\beta$','Train wall / s','Actor median / us','Actor p95 / us','Mapping / us','Features / us'],body)
    auxiliary=[r for r in aggregates if r['stage']=='formal' and r['n_evaluated'] and
        (r['split']=='regression' or r['split']=='test' and r['regime']!='nominal')]
    if auxiliary:
        body=[]
        for split,regime,method in sorted({(r['split'],r['regime'],r['method']) for r in auxiliary}):
            group=[r for r in rows if (r['stage'],r['split'],r['regime'],r['method'])==('formal',split,regime,method)]
            body.append([saved.common.escaped(split+'/'+regime),name(method),beta(method),
                str(len({r['seed'] for r in group})),f"{sum(r['success'] for r in group)}/{len(group)}",
                number(max(r['contour_error_max_mm'] for r in group))])
        lines+=saved.common.table('Separate historical regression and sensitivity checks. S/E counts seed/path runs descriptively; repeated paths within a seed are not independent training repeats. '
            'Maximum contour error retains every run. The full report contains every path, failure reason and time. Historical regression is not unseen-path evidence.',
            f'tab:{prefix}_paper_auxiliary',['Condition','Method',r'$\beta$','Training seeds','S/E','Largest error / mm'],body)
    paired_figure=f'{prefix}_formal_test_nominal_paired'
    candidates=[n for n in figures if n.startswith(f'{prefix}_formal_') and n.endswith('_validation_mechanism')]
    mechanism=next((n for n in candidates if 'line_curve' in n),candidates[0] if candidates else None)
    for figure,caption in [(paired_figure,'Frozen nominal test time differences on same-seed, same-path common successes; positive values favour direct preview. Failure counts are reported separately.'),
                           (mechanism,'Representative validation trace for the lowest available formal seed, shown only to describe the distance, speed, geometry, reward and KCM response. The distance panel uses a local vertical scale in millimetres around 3 mm. It is not held-out performance evidence.')]:
        if figure and figure in figures:
            lines += [r'\begin{figure}[htbp]',r'\centering',f'\\includegraphics[width=0.98\\linewidth]{{figures/generated/{figure}.pdf}}',
                f'\\caption{{{caption}}}',f'\\label{{fig:{figure}_paper}}',r'\end{figure}','']
    return '\n'.join(lines)


def build_report(base=DEFAULT_RUNS,data=DATA,output=None,figures=None,tex=None,include_test=False):
    prefix=base.name
    if not prefix.startswith('preview_direct') or not all(c.isalnum() or c=='_' for c in prefix):
        raise ValueError('Direct study directory requires a distinct preview_direct* artifact prefix')
    output=output or base/'report';figures=figures or PAPER/'figures/generated';tex=tex or PAPER/f'generated/{prefix}_results.tex'
    snapshot=saved.discover(base,data,include_test)
    append_saved_regression(snapshot,base,data)
    replacement_pairs,replacement_summaries=output_replacement_diagnostics(snapshot,base,data)
    snapshot['output_replacement_summaries']=replacement_summaries
    serial_benchmark,serial_costs=serial_actor_benchmark(snapshot,base,data)
    snapshot['serial_actor_costs']=serial_costs
    rows,expected,protocol,inputs=snapshot['rows'],snapshot['expected'],snapshot['protocol'],snapshot['inputs']
    pause=inputs.json(base/'pause_record.json') if (base/'pause_record.json').exists() else None
    errata=inputs.json(base/'protocol_errata.json') if (base/'protocol_errata.json').exists() else None
    if errata and errata['frozen_protocol_sha256']!=inputs.sha(base/'protocol.json'):
        raise ValueError('Protocol errata is bound to a different frozen protocol')
    if not protocol['version'].startswith('preview-direct') or protocol.get('geometry_distance_rule_used') is not False:
        raise ValueError('This reporter only accepts the rule-free direct-distance study')
    aggregates=saved.aggregate(rows,expected)
    for row in aggregates:row['failure_rate']=row['failures']/row['n_evaluated'] if row['n_evaluated'] else None
    paired,pair_rows,suites=direct_pairs(rows,expected,snapshot['selection'])
    add_suite_confidence_intervals(suites);snapshot['suite_summaries']=suites
    traces=evaluation_trace_summaries(rows,inputs)
    epochs,exploration,blocks,issues=training_diagnostics(base,protocol,inputs)
    for budget in snapshot['budgets']:
        progress=next((r for r in exploration if all(r[k]==budget[k] for k in ('stage','method','seed'))),None)
        budget['latest_logged_optimizer_step']=progress['latest_logged_optimizer_step'] if progress else None
    output.mkdir(parents=True,exist_ok=True);products=[]
    tables={'raw_evaluations':[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows],
        'performance_by_path_method':aggregates,'paired_by_path':paired,'common_success_pairs':pair_rows,'complete_suite_pairs':suites,
        'complete_suite_seed_details':[{**{key:suite[key] for key in ('stage','split','regime','baseline','candidate','comparison_role')},
            **detail,'relative_reduction_total_percent':100*detail['baseline_minus_candidate_total_s']/detail['baseline_total_s']}
            for suite in suites for detail in suite['complete_suite_seed_details'] if detail['baseline_total_s']>0],
        'evaluation_trace_summaries':traces,'trace_across_training_seeds':saved.aggregate_trace_summaries(traces),
        'optimizer_epoch_diagnostics':epochs,'exploration_by_training_seed':exploration,'training_distance_blocks':blocks,
        'training_budget':snapshot['budgets'],'pending_evaluations':snapshot['pending'],'read_issues':issues}
    tables.update(output_replacement_pairs=replacement_pairs,output_replacement_summary=replacement_summaries)
    tables.update(serial_actor_inference_runs=serial_benchmark['rows'] if serial_benchmark else [],serial_actor_inference_by_method=serial_costs)
    for name,values in tables.items():
        for suffix in ('json','csv'):
            path=output/f'{name}.{suffix}';saved.write_json(path,values) if suffix=='json' else saved.write_csv(path,values);products.append(path)
    contract=dict(observation=dict(dimension=20,base=18,core=8,corner=4,multiscale=6,
        preview_scales=[.5,1.,2.],per_scale='wrapped tangent change / pi and clipped Euclidean target distance / nominal scaled arc distance',
        appended=['q(s_t), cancellation over a fixed 4.6 mm geometry window','unwrapped tangent range over that fixed window / pi'],
        sequencing='Policy observes state at t (base features built using D_(t-1)), selects full D_t; current control and base next observation use D_t; appended features query s_(t+1)',
        endpoint='Finite episode preview clips target arc at remaining endpoint; distance ratio retains nominal scaled denominator',
        geometry_rule_used=False),reward=dict(reference_distance_mm=4.6,
        reference_role='Fixed complete reward horizon, not multiplied by observation scales; local state features q/range do not assign D',
        formula='new ordered covered arc mm + dt * (raw reward - raw r_progress - raw r_stall) + terminal reward',
        trace_components='All r_* components are stored after time scaling; component reconstruction error is independently computed per run',
        certificate_spacing_mm=protocol.get('shared_ordered_coverage',{}).get('witness_spacing_mm',.25)),
        time=dict(execution='timestamp last minus first; 1 ms interpolation samples, not software wall time',
        actor='CPU forward including observation tensor creation and conversion to NumPy; mean and std heads both evaluated; environment outside this timer',
        direct_mapping='Selection/mapping block only, inside environment; separate from actor latency',
        geometry_features='Progress projection and q/range augmentation block; separate from actor latency',
        evaluation_wall='Episode loop plus trace save and metric computation up to result metadata; not hard real-time latency',
        training='training_summary/checkpoint wall time and registered steps; pending runs remain explicit'))
    saved.write_json(output/'runtime_log_contract.json',contract);products.append(output/'runtime_log_contract.json')
    saved.common.style()
    names=performance_figures(aggregates,figures,products,prefix)+matched_pair_figures(paired,pair_rows,figures,products,prefix)+learning_figures(epochs,figures,products,prefix)
    names+=mechanism_figures(rows,snapshot['selection'],inputs,figures,output,products,prefix)
    tex_contents=tex_report(aggregates,paired,exploration,protocol,snapshot['selection'],names,prefix,replacement_summaries,suites)
    if pause:tex_contents+='\nThis study was paused; unfinished planned runs remain pending. The archived pause record is included in the report provenance.\n'
    tex.parent.mkdir(parents=True,exist_ok=True);tex.write_text(tex_contents,encoding='utf-8');products.append(tex)
    compact=tex.with_name(prefix+'_paper.tex')
    compact.write_text(compact_paper(snapshot,aggregates,paired,traces,names,prefix),encoding='utf-8');products.append(compact)
    summary=dict(study=prefix,protocol_version=protocol['version'],scope='frozen test plus development validation and any saved historical regression' if include_test else 'development validation and any saved historical regression; no held-out test reads',
        evaluations=len(rows),successes=sum(r['success'] for r in rows),failures=sum(not r['success'] for r in rows),
        pending=len(snapshot['pending']),formal_evaluations=sum(r['stage']=='formal' for r in rows),
        interpretation='Direct policy selects complete distance; gradients/support are learning diagnostics, not proof of efficiency advantage',
        paired_sign='baseline minus candidate; positive favours the named candidate',figures=names,pause_record=pause,protocol_errata=errata)
    saved.write_json(output/'summary.json',summary);products.append(output/'summary.json')
    lines=['# Direct-distance PPO saved-log report','',f"Scope: {summary['scope']}.",
        f"Saved evaluations: {len(rows)}; successes: {summary['successes']}; failures: {summary['failures']}; pending: {summary['pending']}.",
        '', 'The policy selects the whole [0.8,4.6] mm distance interval. q/range are input features only; no hybrid-rule data enter this report.',
        'All failures remain. Time effects use strict same-seed/same-path common successes; empty pairs and n<2 SD are null.',
        'A failed conservative ordered-coverage certificate is not, by itself, proof of an actual path shortcut. Archived labels are unchanged; resolution audits are separate.',
        '', '| Stage | Method | Path | Failure/evaluated | Success time mean / s | SD / s |', '|---|---|---|---:|---:|---:|']
    for r in aggregates:
        if r['n_evaluated']:lines.append(f"| {r['stage']} | {r['method']} | {r['path_id']} | {r['failures']}/{r['n_evaluated']} | {r['success_time_s_mean']} | {r['success_time_s_sd']} |")
    lines+=['', '## Observed training exploration','', '| Method | Seed | Latest logged step | Sampled D min/max / mm | Fraction of available distance span | Conditional sigma_D / mm |', '|---|---:|---:|---|---:|---:|']
    for r in exploration:
        if r['preview_is_active']:lines.append(f"| {r['method']} | {r['seed']} | {r['latest_logged_optimizer_step']} | {r['sampled_distance_min_mm']} / {r['sampled_distance_max_mm']} | {r['observed_bound_span_fraction']} | {r['mean_conditional_distance_sigma_mm']} |")
    lines+=['', 'These support/gradient records can establish narrow observed sampling, but cannot identify insufficient exploration as the cause of a performance failure without another controlled comparison. No frozen setting is changed.',
        'Optimizer epochs and 1 ms trajectory samples are descriptive, not independent statistical repeats. Actor, geometry-feature, and distance-mapping overhead are reported separately.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');products.append(output/'report.md')
    contract=dict(backend='Python/matplotlib',archetype='quantitative comparison and synchronized traces',
        evidence='strict failure rates, conditional paired times, then optimization/exploration diagnostics',
        statistics='independent training seeds within paths; no epoch or trace-point iid assumption',
        design='existing manuscript palette and typography; no external precedent claimed for this offline task',
        exports='183 mm editable PDF/SVG plus PNG; source CSV and hash manifest')
    saved.write_json(output/'figure_contract.json',contract);products.append(output/'figure_contract.json')
    for script in (Path(__file__),Path(saved.__file__),Path(saved.common.__file__)):inputs.read(script)
    saved.write_json(output/'hash_manifest.json',dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        inputs=inputs.hashes,outputs={inputs.name(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in products},
        read_only_saved_logs=True,rule_free_direct_study=True,include_frozen_test=include_test))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=DEFAULT_RUNS);parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path);parser.add_argument('--figures',type=Path);parser.add_argument('--tex',type=Path)
    parser.add_argument('--include-frozen-test',action='store_true')
    args=parser.parse_args()
    print(json.dumps(build_report(args.runs,args.data,args.output,args.figures,args.tex,args.include_frozen_test),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
