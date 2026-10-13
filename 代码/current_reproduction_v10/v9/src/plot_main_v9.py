"""Generate v9 full-command plots and tables; style derived from generate_frozen_main_v6.py."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from exact_distance_index_v5 import SegmentDistanceIndex

LABEL={'JNN':'J-NNC','FIXED':'Fixed preview','S2':'Two-step (S2-Q)',
       'S2R':'Two-step (S2-R)','SCMO':'SCMO','COPP':'COPP','NNC':'NNC'}
COLOR={'JNN':'#28688a','FIXED':'#5b9098','S2':'#ba7944','S2R':'#577862',
       'SCMO':'#806595','COPP':'#53565d','NNC':'#b05664'}
STYLE={'JNN':'-','FIXED':':','S2':'--','S2R':'-.','SCMO':(0,(5,1.3,1,1.3)),'COPP':(0,(1,1)),'NNC':'--'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_text(value, digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


def generate(data,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data/'manifest.json').read_text('utf-8'))
    for name,digest in manifest['files'].items():assert sha(data/name)==digest,name
    assert sha(data/manifest['main_model'])==manifest['model_sha256']
    if (out/'figure8.pdf').exists():assert sha(out/'figure8.pdf')==manifest['figure8_sha256']
    metrics={};plots={};references={}
    for item in manifest['runs']:
        trace=data/item['trace'];assert sha(trace)==item['trace_sha256']
        with np.load(trace) as z:p=z['positions'].copy();reference=z['reference'].copy()
        dt=item['dt_s'];m=dict(item['metrics']);n=len(p)-1
        v=np.diff(p,axis=0)/dt;a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
        j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
        values=np.column_stack([np.linalg.norm(x,axis=1) for x in (v,a,j)])
        length=float(values[:,0].sum()*dt);deficit=float((100.-values[:,0]).sum()*dt)
        assert np.allclose(values.max(0),m['vector_peaks'],rtol=1e-8,atol=2e-4),item['id']
        assert abs(length-m['executed_length_mm'])<1e-6
        assert abs(n*dt-m['elapsed_prefix_s'])<1e-8
        m.update(id=item['id'],case=item['case'],method=item['method'],dt_s=dt,
                 executed_length_mm=length,deficit_mm=deficit,mean_speed_mm_s=length/(n*dt),
                 trace_sha256=item['trace_sha256'])
        metrics[item['id']]=m
        if dt==.001:
            plots[(item['case'],item['method'])]=dict(positions=p,values=values,
                time=np.arange(1,n+1)*dt,distance=np.cumsum(values[:,0])*dt)
            references[item['case']]=reference
    methods=[m for m in LABEL if ('original',m) in plots and ('dense',m) in plots]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
        'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'path.simplify':False,'svg.hashsalt':'jnn-v9'})
    outputs=[]
    def save(fig,name):
        for ext in ('pdf','png'):
            file=out/f'{name}.{ext}';fig.savefig(file,dpi=400,bbox_inches='tight',facecolor='white');outputs.append(file)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
    for ax,case,title in zip(axes,('original','dense'),('a  Long mixed path','b  Dense-corner path')):
        r=references[case];ax.plot(*r.T,color='#4c4c4c',lw=.8)
        ax.scatter(*r[0],s=20,color=COLOR['JNN'],zorder=3,label='Start / end')
        ax.set(xlabel='X (mm)',ylabel='Y (mm)');ax.set_title(title,loc='left',fontweight='bold');ax.set_aspect('equal',adjustable='datalim')
    axes[0].legend(loc='best');fig.tight_layout();save(fig,'programs')
    # Full-command spatial motion and error, without smoothing or sample selection.
    for case in ('original','dense'):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),gridspec_kw={'width_ratios':[1,1.3]})
        r=references[case];p=plots[(case,'JNN')]['positions'];m=metrics[f'{case}_JNN_dt1000us']
        axes[0].plot(*r.T,c='#555555',ls='--',lw=.75,label='Reference')
        axes[0].plot(*p.T,c=COLOR['JNN'],lw=.85,label='J-NNC',rasterized=True)
        axes[0].set_aspect('equal',adjustable='datalim');axes[0].set(xlabel='X (mm)',ylabel='Y (mm)')
        axes[0].set_title('a  Complete command trajectory',loc='left',fontweight='bold');axes[0].legend(loc='best')
        distance=SegmentDistanceIndex(r).distances(p[1:])
        assert abs(distance.max()-m['max_polyline_deviation_mm'])<1e-7
        axes[1].plot(plots[(case,'JNN')]['distance'],distance,c=COLOR['JNN'],lw=.7,rasterized=True)
        axes[1].axhline(m['tolerance_mm'],color='#555555',ls='--',lw=.8,label='Polygonal allowance')
        axes[1].set(xlabel='Executed distance (mm)',ylabel='Command deviation (mm)',ylim=(0,1.35))
        axes[1].set_title('b  Full-path geometric deviation',loc='left',fontweight='bold');axes[1].legend(loc='upper right')
        fig.tight_layout();save(fig,'trajectory_'+case)
        fig=plt.figure(figsize=(7.2,4.9))
        grid=fig.add_gridspec(3,2,width_ratios=(3.8,1.4),wspace=.48,hspace=.5)
        axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[2,0])]
        peak_axes=[fig.add_subplot(grid[1,1]),fig.add_subplot(grid[2,1])]
        for method in methods:
            series=plots[(case,method)]
            completed=metrics[f'{case}_{method}_dt1000us']['success']
            label=LABEL[method]+('' if completed else ' (prefix)')
            axes[0].plot(series['distance'],series['values'][:,0]/100.,
                color=COLOR[method],ls=STYLE[method],lw=.75,label=label,rasterized=True)
        for k in (1,2):
            series=plots[(case,'JNN')]
            axes[k].plot(series['distance'],series['values'][:,k]/[100.,2000.,62500.][k],
                color=COLOR['JNN'],lw=.7,rasterized=True)
            peak_ax=peak_axes[k-1]
            peaks=[]
            for i,method in enumerate(methods):
                ratio=metrics[f'{case}_{method}_dt1000us']['vector_limit_ratios'][k]
                peaks.append(ratio)
                peak_ax.scatter(ratio,i,s=15,color=COLOR[method],zorder=3)
            peak_ax.set_yticks(range(len(methods)),[LABEL[x].replace('Two-step (','').replace(')','')+
                ('' if metrics[f'{case}_{x}_dt1000us']['success'] else '*') for x in methods],fontsize=6)
            peak_ax.invert_yaxis();peak_ax.axvline(1,color='#555555',ls='--',lw=.7)
            peak_ax.set_title('Recorded peaks',fontsize=8,loc='left')
            if max(peaks)>10:peak_ax.set_xscale('log')
            else:peak_ax.set_xlim(0,max(1.07,max(peaks)*1.04))
            peak_ax.tick_params(axis='x',labelsize=6)
            peak_ax.grid(axis='x',color='#eeeeee',lw=.4)
        for ax,label in zip(axes,('a  Speed / limit','b  Acceleration / limit','c  Jerk / limit')):
            ax.axhline(1.,c='#444444',ls='--',lw=.65);ax.set_ylim(-.04,1.07)
            ax.set_ylabel(label);ax.grid(axis='y',color='#e9e9e9',lw=.4)
        axes[0].set_ylim(-.04,max(1.07,max(plots[(case,m)]['values'][:,0].max()/100. for m in methods)*1.04))
        axes[-1].set_xlabel('Executed distance (mm)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='upper center',frameon=False)
        fig.subplots_adjust(top=.83,bottom=.11,left=.09,right=.98);save(fig,'kinematics_'+case)
    # All numerical manuscript values are emitted from the same complete records.
    macros=[]
    for case,stem in [('original','Long'),('dense','Dense')]:
        j=metrics[f'{case}_JNN_dt1000us']
        values={'Time':f"{j['complete_time_s']:.3f}",'Length':f"{j['executed_length_mm']:.2f}",
            'Deficit':f"{j['deficit_mm']:.2f}",'MeanSpeed':f"{j['mean_speed_mm_s']:.2f}",
            'Deviation':f"{j['max_polyline_deviation_mm']:.4f}",
            'Gates':str(j['geometry']['gate_count']),
            'HalfTime':f"{metrics[f'{case}_JNN_dt500us']['complete_time_s']:.4f}"}
        for method in methods:
            b=metrics[f'{case}_{method}_dt1000us'];suffix=method.replace('2','Two')
            values[suffix+'Time']=time_text(b['complete_time_s'])
            values[suffix+'Saving']=('--' if b['complete_time_s'] is None else
                f"{100*(b['complete_time_s']-j['complete_time_s'])/b['complete_time_s']:.2f}")
        for name,value in values.items():macros.append('\\newcommand{\\'+stem+name+'}{'+value+'}')
    (out/'numbers.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8')
    table=[r'\begin{table}[htbp]',r'\centering',
        r'\caption{Complete simulated execution time at 1 ms. Numerical entries satisfy the common geometric and planar vector constraints and include startup, ordered traversal, braking and final rest. A dash denotes an execution that does not satisfy the complete-task criteria.}',
        r'\label{tab:complete_times}',r'\small',r'\begin{tabular}{lrr}',r'\toprule',
        r'Method & Long path (s) & Dense path (s) \\',r'\midrule']
    for method in methods:
        a,b=[metrics[f'{case}_{method}_dt1000us']['complete_time_s'] for case in ('original','dense')]
        table.append(f'{LABEL[method]} & {time_text(a)} & {time_text(b)} '+r'\\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'complete_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    table=[r'\begin{table}[htbp]',r'\centering',
        r'\caption{Time-resolution check with frozen policies and planning parameters. A dash denotes an incomplete execution or an unavailable paired time difference. Each numerical difference compares two complete executions of one program.}',
        r'\label{tab:resolution}',r'\small',r'\begin{tabular}{llrrr}',r'\toprule',
        r'Program & Method & 1 ms (s) & 0.5 ms (s) & Change (ms) \\',r'\midrule']
    for case,label in [('original','Long'),('dense','Dense')]:
        for method in methods:
            a,b=[metrics[f'{case}_{method}_dt{us}us']['complete_time_s'] for us in (1000,500)]
            difference='--' if a is None or b is None else f'{1000*(b-a):+.1f}'
            table.append(f'{label} & {LABEL[method]} & {time_text(a)} & {time_text(b,4)} & {difference} '+r'\\')
        if case=='original':table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (out/'resolution_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    outputs += [out/name for name in ('numbers.tex','complete_table.tex','resolution_table.tex','metrics.json')]
    receipt=dict(input_manifest_sha256=sha(data/'manifest.json'),model_sha256=manifest['model_sha256'],
        recorded_runs=len(metrics),complete_runs=sum(x['success'] for x in metrics.values()),
        methods=methods,outputs={p.name:sha(p) for p in outputs},
        figure8_sha256=manifest['figure8_sha256'],generator_sha256=sha(__file__),
        plotting='Complete recorded commands; rasterized line artists only for PDF size; no smoothing, temporal cropping or sample removal')
    (out/'generation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(runs=len(metrics),methods=methods,figures=5),ensure_ascii=False))


