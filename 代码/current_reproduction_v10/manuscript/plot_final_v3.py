"""Publication figures from the audited complete final grid; no outcome selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'shared_manuscript'
COLORS={'adaptive':'#28688a','fixed':'#ba7944','S2R':'#577862','S2':'#999999','SCMO':'#806595'}
LABELS={'adaptive':'J-NNC','fixed':'Fixed preview','S2R':'S2-R','S2':'S2-Q','SCMO':'SCMO'}
NAMES={'final_dense_fold':'Dense folds','final_chamfer_chain':'Chamfer chain',
       'final_rounded_triangle':'Rounded triangle','final_quintic_double_bend':'Double bend',
       'final_inward_channels':'Inward channels'}
METHODS=['adaptive','fixed','S2R','SCMO']

def read(p):return json.loads(Path(p).read_text('utf-8'))

def setup():
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                         'legend.frameon':False,'xtick.labelsize':7,'ytick.labelsize':7})

def save(fig,stem):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(stem+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)

def trajectory(row):
    with np.load(ROOT/row['trace_file'],allow_pickle=False) as z:
        p=z['positions'];ref=z['reference']
    v=np.diff(p,axis=0)/row['dt_s'];a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/row['dt_s']
    j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/row['dt_s']
    speed=np.linalg.norm(v,axis=1);length=np.cumsum(speed)*row['dt_s']
    return dict(p=p,ref=ref,v=v,a=a,j=j,speed=speed,length=length,
                deficit=np.cumsum(100-speed)*row['dt_s'],time=np.arange(1,len(p))*row['dt_s'])

def selected(data,path):
    rows=[next(r for r in data['rows'] if r['path']==path and r['method']==m and r['dt_s']==.001
               and r['seed']==(101 if m in ['adaptive','fixed'] else None)) for m in METHODS]
    return rows

def representative(data,path,stem):
    rows=selected(data,path);traces={r['method']:trajectory(r) for r in rows if r['trajectory_present']}
    ref=next(iter(traces.values()))['ref']
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.35),layout='constrained')
    ax,zoom,speedax,deficitax=axes.flat
    for panel in [ax,zoom]:panel.plot(*ref.T,color='#333333',lw=1.,ls='--',label='Reference')
    tube=LineString(ref).buffer(rows[0]['tolerance_mm'],quad_segs=40)
    for poly in getattr(tube,'geoms',[tube]):
        xy=np.asarray(poly.exterior.coords);zoom.fill(*xy.T,color='#edf0f2',zorder=0)
        for ring in poly.interiors:
            hole=np.asarray(ring.coords);zoom.fill(*hole.T,color='white',zorder=.1)
    # Select the first maximal-turn reference vertex using reference geometry alone.
    delta=np.diff(ref,axis=0);directions=delta/np.linalg.norm(delta,axis=1)[:,None]
    turns=np.arccos(np.clip(np.sum(directions[:-1]*directions[1:],axis=1),-1,1))
    center=ref[int(np.argmax(turns))+1];halfwidth=6.
    for row in rows:
        m=row['method']
        if m not in traces:continue
        tr=traces[m];color=COLORS[m]
        status='' if row['success'] else ' (incomplete)'
        label=LABELS[m]+status
        for panel in [ax,zoom]:panel.plot(*tr['p'].T,c=color,lw=1.05,label=label)
        if not row['success']:
            for panel in [ax,zoom]:panel.scatter(*tr['p'][-1],marker='x',c=color,s=20,zorder=5)
        speedax.plot(tr['length'],tr['speed'],c=color,lw=.85,label=label)
        deficitax.plot(tr['length'],tr['deficit'],c=color,lw=1.1,label=label)
    ax.set_aspect('equal');zoom.set_aspect('equal');ax.set_xlabel('$X$ (mm)');ax.set_ylabel('$Y$ (mm)')
    zoom.set_xlim(center[0]-halfwidth,center[0]+halfwidth);zoom.set_ylim(center[1]-halfwidth,center[1]+halfwidth)
    zoom.set_xlabel('$X$ (mm)');zoom.set_ylabel('$Y$ (mm)')
    speedax.set_xlabel('Executed distance (mm)');speedax.set_ylabel('Speed (mm/s)');speedax.set_ylim(bottom=0)
    speedax.axhline(100,c='#555',ls=':',lw=.7)
    deficitax.set_xlabel('Executed distance (mm)');deficitax.set_ylabel('Accumulated speed deficit (mm)');deficitax.set_ylim(bottom=0)
    for panel,label in zip(axes.flat,['a','b','c','d']):panel.text(-.13,1.04,label,transform=panel.transAxes,fontweight='bold',fontsize=10)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='outside upper center',fontsize=8)
    save(fig,stem)

def constraints(data,paths):
    fig,axes=plt.subplots(3,2,figsize=(7.2,5.35),sharex='col',layout='constrained')
    for col,path in enumerate(paths):
        for row in selected(data,path):
            if not row['trajectory_present']:continue
            tr=trajectory(row);color=COLORS[row['method']]
            for i,(name,limit) in enumerate([('v',100),('a',2000),('j',62500)]):
                axes[i,col].plot(tr['time'],np.linalg.norm(tr[name],axis=1)/limit,color=color,lw=.65,
                                 label=LABELS[row['method']]+('' if row['success'] else ' (incomplete)'))
        axes[0,col].set_title(NAMES[path],fontsize=9)
    for i in range(3):
        for col in range(2):
            ax=axes[i,col];ax.axhline(1,c='#444444',ls='--',lw=.7);ax.set_ylim(bottom=0)
            ax.text(-.14,1.04,chr(97+2*i+col),transform=ax.transAxes,fontweight='bold',fontsize=10)
            if col==0:ax.set_ylabel(['Velocity / limit','Acceleration / limit','Jerk / limit'][i])
            if i==2:ax.set_xlabel('Command time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='outside upper center',fontsize=8)
    save(fig,'constraints_v3')

def training(data):
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7),layout='constrained')
    for method in ['adaptive','fixed']:
        group=[]
        for seed in [101,211,307,401,503]:
            folder=ROOT/'results/unified_v3/paired'/f'seed_{seed}'/method
            curve=read(folder/'training_curve.json');steps=[r['step'] for r in curve]
            axes[1].plot(steps,[r['clip_fraction'] for r in curve],c=COLORS[method],lw=.65,alpha=.45)
            vals=[]
            for step in [0,4096,8192]:
                records=read(folder/f'validation_{step:05d}.json')['records']
                vals.append(sum(r['complete_time_s'] for r in records) if all(r['success'] for r in records) else np.nan)
            axes[0].plot([0,4096,8192],vals,c=COLORS[method],lw=.6,alpha=.45,marker='.',ms=3)
            group.append(vals)
        array=np.array(group)
        if not np.isfinite(array).all():
            axes[0].text(.02,.95,'Missing time: incomplete validation',transform=axes[0].transAxes,va='top',fontsize=7)
        axes[0].plot([0,4096,8192],np.nanmean(array,axis=0),c=COLORS[method],lw=1.6,marker='o',ms=3,label=LABELS[method])
    axes[0].set_ylabel('Two-program validation time (s)');axes[1].set_ylabel('PPO clip fraction')
    for i,ax in enumerate(axes):
        ax.set_xlabel('Fresh PPO decisions');ax.set_xticks([0,4096,8192]);ax.text(-.13,1.03,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=10)
    axes[0].legend(fontsize=8);axes[1].set_ylim(bottom=0)
    save(fig,'training_v3')

def main():
    setup();data=read(ROOT/'manuscript/results_v3/analysis.json');assert data['valid_for_manuscript']
    reg=read(ROOT/'prospective/final_v3/registration.json')
    paths=reg['representative_figure_paths'];assert len(paths)==2
    representative(data,paths[0],'dense_experiment_v3')
    representative(data,paths[1],'mixed_experiment_v3')
    constraints(data,paths);training(data)
    (ROOT/'audit/final_figure_contract_v3.json').write_text(json.dumps({
        'backend':'Python/matplotlib only','figures':{
            'dense_experiment_v3':'Complete dense-fold trajectory, geometrically selected corner, unsmoothed speed and integrated deficit.',
            'mixed_experiment_v3':'Complete line/arc program, geometrically selected corner, unsmoothed speed and integrated deficit.',
            'constraints_v3':'Every retained command in predetermined seed-101 representative executions; no smoothing or peak clipping.',
            'training_v3':'All five independent seeds and predetermined validation checkpoints; empirical behavior, no convergence theorem.'},
        'selection':'Program identities and seed 101 fixed before evaluation; local view uses first maximal geometric direction change.',
        'source':'manuscript/results_v3/analysis.json and every referenced trace',
        'visual_precedent':'Provided motion figures supply common physical axes and trajectory/speed comparison grammar. Li2023 Fig4 and Fig17 captions were inspected as domain examples, but no verified CAS ranking is asserted and their layouts/data are not copied.',
        'export':'Editable-text PDF/SVG with PNG preview; Arial, consistent family colors; 183 mm width',
        'statistics':'Training thin lines are individual seeds; thick validation lines are means. No stochastic band uses command samples as independent replicates.'
    },indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
