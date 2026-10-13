"""Read-only scientific diagnostic: registered local corners, real full traces."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
geometry=json.loads((ROOT/'dense_closed_case/geometry.json').read_text(encoding='utf-8'))
reference=np.loadtxt(ROOT/'dense_closed_case/reference.csv',delimiter=',',skiprows=1)
datasets=[('S2',ROOT/'dense_s2_initial','trace.npz','all84_metrics.json','#555555','--'),
    ('Three actions, 32 updates',ROOT/'ppo14_action3_allow_seed922301_credit30','eval_0032.npz','all84_update32.json','#0072B2','-'),
    ('Two actions, 32 updates',ROOT/'ppo14_action2_allow_seed922301_credit30_d1p5','eval_0032.npz','all84_update32.json','#D55E00','-.')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(10,3.8),layout='constrained')
manifest=[]
for ax,corner_id in zip(axes,geometry['mechanism_columns']):
    corner=next(c for c in geometry['sharp_corners'] if c['id']==corner_id)
    k=corner['reference_vertex_index'];vertex=reference[k]
    incoming=reference[k]-reference[k-1];incoming/=np.linalg.norm(incoming)
    normal=np.sign(corner['signed_heading_change_deg'])*np.array([-incoming[1],incoming[0]])
    frame=np.stack([incoming,normal],axis=1)
    path=(reference[k-1:k+2]-vertex)@frame
    local_points=[]
    for label,folder,trace,metrics,color,style in datasets:
        rows=np.load(folder/trace)['rows']
        metric=json.loads((folder/metrics).read_text(encoding='utf-8'))
        record=next(c for c in metric['corners'] if c['id']==corner_id)
        ids=np.arange(record['first_index'],record['last_index']+1)
        local=(rows[ids,:2]-vertex)@frame;local_points.append(local)
        ax.plot(local[:,0],local[:,1],style,color=color,lw=1.5,label=label,zorder=3)
        manifest.append(dict(corner=corner_id,label=label,trace=str(folder/trace),trace_sha256=metric['trace_sha256'],indices=[int(ids[0]),int(ids[-1])]))
    points=np.vstack(local_points);low=points.min(0)-1.;high=points.max(0)+1.
    xx,yy=np.meshgrid(np.linspace(low[0],high[0],200),np.linspace(low[1],high[1],200));query=np.stack([xx,yy],axis=-1)
    distances=[]
    for p0,p1 in zip(path[:-1],path[1:]):
        edge=p1-p0;u=np.clip(((query-p0)*edge).sum(-1)/(edge@edge),0.,1.)
        distances.append(np.linalg.norm(query-(p0+u[...,None]*edge),axis=-1))
    distance=np.minimum(*distances)
    ax.contourf(xx,yy,distance,levels=[0.,1.25],colors=['#EAEAEA'],zorder=0)
    ax.plot(path[:,0],path[:,1],':',color='black',lw=1,label='Reference',zorder=2)
    ax.plot(0,0,'o',ms=2.5,color='black')
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),aspect='equal',xlabel='Incoming direction (mm)',ylabel='Turn-side displacement (mm)',title=f"{corner['interior_angle_deg']:g}° included angle")
    ax.grid(alpha=.15,lw=.5)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('Development diagnostic: actual trajectories at equal training budgets',fontsize=11)
output=ROOT/'local_development_update32'
fig.savefig(output.with_suffix('.png'),dpi=180)
fig.savefig(output.with_suffix('.pdf'))
output.with_suffix('.json').write_text(json.dumps(dict(scope='Development diagnostic only; not a final manuscript figure or accepted-model comparison',selection='Registered three mechanism columns, chosen before training comparisons',sources=manifest),indent=2)+'\n',encoding='utf-8')
print(output.with_suffix('.png'))
