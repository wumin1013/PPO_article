"""Read saved event-preview runs and generate auditable statistics and figures.

No policy/environment imports or evaluation. Failures stay in the source table;
successful time summaries and matched successful comparisons state their sets.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT/'论文项目'
STUDY = PAPER/'revision_20260909_event_preview/study'
DATA = ROOT/'PPO_project/data/preview_event_v1'
LABELS = {'legacy_direct':'Earlier direct', 'state_direct':'Corrected direct',
          'selected_fixed':'Trained fixed', 'selected_hold':'Held policy preview'}
COLORS = ['#818181','#628cac','#d58a43','#2467a4']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def statistics(values):
    values=list(values);a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float);n=len(a)
    missing=len(values)-n
    if not n:return dict(n=0,missing_n=missing,mean=None,std=None,ci95=None)
    sd=float(np.std(a,ddof=1)) if n>1 else None
    half=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n)) if n>1 else None
    mean=float(a.mean())
    return dict(n=n,missing_n=missing,mean=mean,std=sd,ci95=[mean-half,mean+half] if half is not None else None)


def collect(stage, split, regime, allow_partial=False):
    protocol=read(STUDY/'protocol.json');manifest=read(DATA/'manifest.json')
    methods=protocol['pilot_variants'] if stage=='pilot' else protocol['formal_variants']
    seeds=[protocol['pilot_seed']] if stage=='pilot' else protocol['formal_seeds']
    if regime!='nominal':seeds=[11]
    budget=protocol[f'{stage}_steps']
    if split=='regression':
        import yaml
        cfg=yaml.safe_load((PAPER/'generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        paths=[p['name'] for p in cfg['training']['path_curriculum']['paths']]
    else:paths=[p['id'] for p in manifest['paths'] if p['split']==split]
    if not paths:raise ValueError('No registered paths')
    inputs={};rows=[];pending=[];training=[]
    def record(path):inputs[str(Path(path).relative_to(ROOT).as_posix())]=sha(path)
    record(STUDY/'protocol.json');record(DATA/'manifest.json')
    frozen=read(STUDY/'freeze.json') if split=='test' else None
    if frozen:
        record(STUDY/'freeze.json');record(STUDY/'selection.json')
        if frozen['source_sha256']!=protocol['source_sha256'] or frozen['protocol_sha256']!=sha(STUDY/'protocol.json') or frozen['selection_sha256']!=sha(STUDY/'selection.json') or frozen['manifest_sha256']!=sha(DATA/'manifest.json'):
            raise ValueError('Test freeze binding changed')
        expected={f'formal/{m}_seed{s}' for m in protocol['formal_variants'] for s in protocol['formal_seeds']}
        if set(frozen['models'])!=expected or set(frozen['configs'])!=expected:raise ValueError('Incomplete formal freeze roster')
        for key in expected:
            if frozen['models'][key]!=sha(STUDY/key/'checkpoint.pt') or frozen['configs'][key]!=sha(STUDY/key/'config.yaml'):raise ValueError('Frozen model/config changed')
            record(STUDY/key/'checkpoint.pt');record(STUDY/key/'config.yaml')
    for method in methods:
        for seed in seeds:
            run=STUDY/stage/f'{method}_seed{seed}'
            summary_path=run/'training_summary.json'
            if not summary_path.exists():
                pending.append(dict(method=method,seed=seed,status='training_pending'));continue
            summary=read(summary_path);provenance=read(run/'provenance.json')
            model=sha(run/'checkpoint.pt');config=sha(run/'config.yaml')
            if summary['steps']!=budget or summary['checkpoint_sha256']!=model or provenance['config_sha256']!=config or provenance['source_sha256']!=protocol['source_sha256'] or provenance['manifest_sha256']!=sha(DATA/'manifest.json'):
                raise ValueError(f'Incomplete or changed model/config: {run}')
            with zipfile.ZipFile(run/'source.zip') as archive:
                for name,digest in provenance['source_files'].items():
                    if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ValueError('Source archive mismatch')
            if frozen:
                key=run.relative_to(STUDY).as_posix()
                if frozen['models'].get(key)!=model or frozen['configs'].get(key)!=config:raise ValueError('Frozen model changed')
            for name in ('training_summary.json','provenance.json','config.yaml','checkpoint.pt','source.zip'):record(run/name)
            training.append(summary)
            for path_id in paths:
                file=run/f'evaluation_{split}_{regime}'/f'{path_id}.json'
                if not file.exists():
                    pending.append(dict(method=method,seed=seed,path_id=path_id,status='evaluation_pending'));continue
                row=read(file)
                expected=dict(variant=method,seed=seed,path_id=path_id,split=split,regime=regime,training_steps=budget,
                    checkpoint_sha256=model,config_sha256=config,source_sha256=protocol['source_sha256'],manifest_sha256=sha(DATA/'manifest.json'))
                if any(row.get(k)!=v for k,v in expected.items()):raise ValueError(f'Misbound result: {file}')
                record(file);record(file.with_suffix('.npz'))
                row.update(result_file=file.relative_to(ROOT).as_posix(),trace_file=file.with_suffix('.npz').relative_to(ROOT).as_posix())
                rows.append(row)
    if pending and not allow_partial:raise ValueError(f'{len(pending)} registered jobs pending; use --allow-partial for progress only')
    return dict(stage=stage,split=split,regime=regime,methods=methods,seeds=seeds,paths=paths,rows=rows,pending=pending,
                training=training,input_sha256=inputs,expected_trajectories=len(methods)*len(seeds)*len(paths))


def summarize(rows):
    success=[r for r in rows if r['success']]
    return dict(n=len(rows),successes=len(success),failures=len(rows)-len(success),
        failure_reasons=dict(Counter(reason for r in rows for reason in r['failure_reasons'])),
        successful_seeds=[r['seed'] for r in success],
        successful_time_s=statistics([r.get('elapsed_time_s') for r in success]),
        all_time_s=statistics([r.get('elapsed_time_s') for r in rows]),
        all_peak_error_mm=statistics([r.get('contour_error_max_mm') for r in rows]),
        all_rmse_mm=statistics([r.get('contour_error_rmse_mm') for r in rows]),
        invalid_traces=sum(r.get('valid_trace') is False for r in rows),
        missing_numeric_policy='Unknown numeric fields remain null/blank; missing_n records exclusions from each numeric summary, never from the failure denominator',
        inference_median_us_by_trajectory=statistics([r.get('inference_median_us') for r in rows]),
        inference_note='Per-trajectory timings collected during concurrent evaluation; not an isolated deployment benchmark')


def paired(rows, methods, paths, seeds, baseline='selected_fixed'):
    lookup={(r['variant'],r['seed'],r['path_id']):r for r in rows};result=[]
    for method in methods:
        if method==baseline:continue
        for path_id in [*paths,'all_paths']:
            used=[];excluded=[]
            subset=paths if path_id=='all_paths' else [path_id]
            for seed in seeds:
                pairs=[(lookup.get((method,seed,p)),lookup.get((baseline,seed,p))) for p in subset]
                if any(a is None or b is None for a,b in pairs):
                    excluded.append(dict(seed=seed,reason='missing_result'));continue
                if any(not a['success'] or not b['success'] for a,b in pairs):
                    excluded.append(dict(seed=seed,reason='not_jointly_successful'));continue
                if any(a.get('elapsed_time_s') is None or b.get('elapsed_time_s') is None for a,b in pairs):
                    excluded.append(dict(seed=seed,reason='unknown_elapsed_time'));continue
                candidate=sum(a['elapsed_time_s'] for a,b in pairs);reference=sum(b['elapsed_time_s'] for a,b in pairs)
                used.append(dict(seed=seed,candidate_time_s=candidate,fixed_time_s=reference,improvement_pct=100*(reference-candidate)/reference))
            result.append(dict(candidate=method,baseline=baseline,path_id=path_id,pairs=used,excluded=excluded,
                improvement_pct=statistics([p['improvement_pct'] for p in used]),unit='Independent matched training seed; fixed path set'))
    return result


def export_figure(fig, prefix, outputs):
    fig.align_labels()
    for extension in ('pdf','svg','png'):
        path=PAPER/'figures/generated'/f'{prefix}.{extension}'
        fig.savefig(path,dpi=220,bbox_inches='tight');outputs.append(path)
    plt.close(fig)


def figures(snapshot, pairs, prefix, outputs):
    methods,paths,rows=snapshot['methods'],snapshot['paths'],snapshot['rows']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    if snapshot['stage']=='pilot':
        fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
        for j,path in enumerate(paths):
            ax=axes[j]
            for i,method in enumerate(methods):
                rr=[r for r in rows if r['variant']==method and r['path_id']==path]
                for r in rr:
                    if r.get('elapsed_time_s') is not None:ax.scatter(i,r['elapsed_time_s'],color='#2467a4' if r['success'] else '#bd493f',marker='o' if r['success'] else 'x')
            ax.set_title(path.replace('_',' '));ax.set_ylabel('Elapsed time (s)');ax.set_xticks(range(len(methods)),methods,rotation=60,ha='right')
        fig.suptitle('Validation screening: one seed, 32,768 steps each\nBlue circles: strict success; red crosses: failed termination')
        export_figure(fig,prefix+'_performance',outputs);return
    fig,axes=plt.subplots(2,len(paths),figsize=(10,5.8),squeeze=False,layout='constrained')
    for j,path in enumerate(paths):
        for i,method in enumerate(methods):
            rr=sorted([r for r in rows if r['variant']==method and r['path_id']==path],key=lambda r:r['seed'])
            for k,r in enumerate(rr):
                x=i+(k-(len(rr)-1)/2)*.07
                for a,field in ((axes[0,j],'elapsed_time_s'),(axes[1,j],'contour_error_max_mm')):
                    if r.get(field) is not None:a.scatter(x,r[field],edgecolors=COLORS[i],facecolors=COLORS[i] if r['success'] else 'none',s=32)
            axes[0,j].text(i,1.01,f"{sum(r['success'] for r in rr)}/{len(rr)}",transform=axes[0,j].get_xaxis_transform(),ha='center',fontsize=8)
        axes[0,j].set_title(path.replace('_event_v1_test','').replace('_',' '),pad=20)
        for ax in axes[:,j]:ax.set_xticks(range(len(methods)),[LABELS.get(m,m).replace(' ','\n') for m in methods],fontsize=8)
    axes[0,0].set_ylabel('Elapsed time, all runs (s)');axes[1,0].set_ylabel('Peak contour error (mm)')
    fig.suptitle(f"{snapshot['split']} / {snapshot['regime']}: counts show strict successes; open markers are failures")
    export_figure(fig,prefix+'_performance',outputs)
    fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
    candidates=[m for m in methods if m!='selected_fixed'];categories=[*paths,'all_paths']
    for i,method in enumerate(candidates):
        for j,path in enumerate(categories):
            row=next(p for p in pairs if p['candidate']==method and p['path_id']==path);st=row['improvement_pct'];x=j+(i-1)*.22
            for k,p in enumerate(row['pairs']):ax.scatter(x+(k-2)*.018,p['improvement_pct'],s=16,color=COLORS[methods.index(method)],alpha=.55)
            if st['n']:
                yerr=None if st['ci95'] is None else [[st['mean']-st['ci95'][0]],[st['ci95'][1]-st['mean']]]
                ax.errorbar(x,st['mean'],yerr=yerr,fmt='D',color=COLORS[methods.index(method)],capsize=3,label=LABELS[method] if j==0 else None)
    ax.axhline(0,color='#666666',lw=.8);ax.set_xticks(range(len(categories)),[p.replace('_event_v1_test','').replace('_',' ') for p in categories])
    ax.set_ylabel('Time improvement over trained fixed (%)');ax.legend(fontsize=8)
    ax.set_title('Matched successful seeds: points and mean with 95% t interval')
    export_figure(fig,prefix+'_paired',outputs)


def tex_table(snapshot, summaries, pairs):
    def fmt(s):return '--' if not s['n'] else f"{s['mean']:.3f}"+(f" $\\pm$ {s['std']:.3f}" if s['std'] is not None else '')
    lines=['% Automatically generated; all source hashes are in the accompanying report manifest.',r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{3pt}',
        r'\caption{Event-preview '+snapshot['split']+' results ('+snapshot['regime'].replace('_',r'\_')+r'). Time is mean $\pm$ sample SD over the listed strict successes; error includes failures.}',
        r'\label{tab:event-'+snapshot['stage']+'-'+snapshot['split']+'-'+snapshot['regime']+'}',r'\begin{tabular}{llrrrr}\toprule',
        r'Path & Method & Success & Time (s) & Peak (mm) & RMSE (mm)\\\midrule']
    for s in summaries:
        path=s['path_id'].replace('_event_v1_test','').replace('_validation','').replace('_',' ')
        method=LABELS.get(s['method'],s['method']).replace('_',r'\_')
        lines.append(f"{path} & {method} & {s['successes']}/{s['n']} & {fmt(s['successful_time_s'])} & {fmt(s['all_peak_error_mm'])} & {fmt(s['all_rmse_mm'])} "+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    return '\n'.join(lines)+'\n'


def tex_paired_table(snapshot, pairs):
    """Display the actual jointly successful seed set; never infer missing CIs."""
    label='event-paired-'+snapshot['stage']+'-'+snapshot['split']+'-'+snapshot['regime']
    lines=[r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
        r'\caption{Paired total-time improvement over trained fixed preview on '+snapshot['split']+
        ' ('+snapshot['regime'].replace('_',r'\_')+r'). Positive values favor the candidate. Each pair sums all three paths within one independent training seed; only jointly successful suites enter. A dash means no estimable interval, not zero uncertainty.}',
        r'\label{tab:'+label+'}',r'\begin{tabular}{lrlrr}\toprule',
        r'Candidate & Pairs & Included seeds & Mean (\%) & 95\% $t$ interval (\%)\\\midrule']
    for item in pairs:
        if item['path_id']!='all_paths':continue
        stats=item['improvement_pct']
        mean='--' if stats['mean'] is None else f"{stats['mean']:.3f}"
        ci='--' if stats['ci95'] is None else f"[{stats['ci95'][0]:.3f}, {stats['ci95'][1]:.3f}]"
        seeds=', '.join(str(p['seed']) for p in item['pairs']) or '--'
        lines.append(f"{LABELS.get(item['candidate'],item['candidate'])} & {stats['n']} & {seeds} & {mean} & {ci} "+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['pilot','formal'],default='formal')
    parser.add_argument('--split',choices=['validation','test','regression'],default='validation')
    parser.add_argument('--regime',choices=['nominal','tight','low_jerk'],default='nominal');parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--figures',action='store_true',help='Explicitly generate reference draft figures; default leaves all manuscript figures unchanged')
    args=parser.parse_args();s=collect(args.stage,args.split,args.regime,args.allow_partial)
    prefix=f'preview_event_{args.stage}_{args.split}_{args.regime}'
    out=STUDY/'reports'/prefix;out.mkdir(parents=True,exist_ok=True);outputs=[]
    summaries=[dict(method=m,path_id=p,**summarize([r for r in s['rows'] if r['variant']==m and r['path_id']==p])) for p in s['paths'] for m in s['methods']]
    pairs=paired(s['rows'],s['methods'],s['paths'],s['seeds']) if args.stage=='formal' else []
    result=dict(**{k:v for k,v in s.items() if k!='input_sha256'},summaries=summaries,paired=pairs,
        statistical_scope='Independent training seeds conditional on the registered finite path set. No population-wide unseen-path guarantee.',
        training_total_steps=sum(r['steps'] for r in s['training']),training_process_wall_s=sum(r['training_wall_s'] for r in s['training']))
    write(out/'report.json',result);outputs.append(out/'report.json')
    fields=['variant','seed','path_id','split','regime','success','path_completed','ordered_progress','elapsed_time_s','contour_error_max_mm','contour_error_mean_mm','contour_error_rmse_mm','inference_median_us','inference_p95_us','terminal_feed_mm_s','preview_min_mm','preview_max_mm','preview_event_count','failure_reasons','checkpoint_sha256','config_sha256','source_sha256','result_file','trace_file']
    with (out/'source_data.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in s['rows']:writer.writerow({k:json.dumps(r.get(k)) if isinstance(r.get(k),list) else r.get(k) for k in fields})
    outputs.append(out/'source_data.csv')
    if not s['pending']:
        if args.figures:figures(s,pairs,prefix,outputs)
        if args.stage=='formal':
            path=PAPER/'generated'/f'{prefix}.tex';path.write_text(tex_table(s,summaries,pairs),encoding='utf-8');outputs.append(path)
            path=PAPER/'generated'/f'{prefix}_paired.tex';path.write_text(tex_paired_table(s,pairs),encoding='utf-8');outputs.append(path)
    write(out/'report_manifest.json',dict(command='python '+' '.join(sys.argv),script_sha256=sha(__file__),
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_sha256=s['input_sha256'],
        output_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in outputs},pending_count=len(s['pending']),
        figures_requested=args.figures,figure_role='Reference drafts only; final artwork is left to the author'))
    print(json.dumps(dict(report=str(out),rows=len(s['rows']),pending=len(s['pending']),paired_suites=[p for p in pairs if p['path_id']=='all_paths']),ensure_ascii=False))


if __name__=='__main__':main()
