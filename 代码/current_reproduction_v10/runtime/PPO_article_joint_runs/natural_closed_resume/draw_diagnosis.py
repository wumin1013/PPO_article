"""Exploratory audit visualization, not a final manuscript figure."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from natural_case_metrics import ordered_projection
from study_io import write_json,sha

plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
audit=json.loads((OUT/'saved_evidence_audit.json').read_text(encoding='utf-8'))
queries=json.loads((OUT/'nonlocal_effects.json').read_text(encoding='utf-8'))
write_json(OUT/'diagnostic_figure_contract.json',dict(status='exploratory internal diagnostic; not a submission-ready figure',
    claim='Most excess time is outside corner windows; far-late feed requests respond to global scalar inputs, not accumulated memory.',
    evidence=['Raw full-lap progress-aligned duration difference','Raw request and execution traces','Matched-state global-input queries','Complete-period cruise duration'],
    backend='Python matplotlib',archetype='quantitative grid',replications='one recorded model; no error bars',
    sources={str(OUT/n):sha(OUT/n) for n in ('saved_evidence_audit.json','nonlocal_effects.json')},
    visual_precedent='Exploratory diagnostic only; final manuscript figure precedent/layout review is pending.'))
fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
case=BASE/'paper_evidence_review/natural_closed_case'
binding=json.loads((case/'paper_result_binding.json').read_text(encoding='utf-8'))
data={}
for name in ('Joint','S2'):
    z=np.load(BASE/binding['primary_methods'][name]['trace']);rows=z['rows']
    s=ordered_projection(z['reference'],rows[:,:2]);t=np.arange(1,len(rows)+1)*.001
    data[name]=(s,t,z)
grid=np.linspace(0,44419,3000)
delta=np.interp(grid,data['Joint'][0],data['Joint'][1])-np.interp(grid,data['S2'][0],data['S2'][1])
axes[0,0].plot(grid/1000,delta,color='#235d88');axes[0,0].axhline(0,color='.6',lw=.6)
axes[0,0].set(xlabel='Reference progress (m)',ylabel='Joint minus S2 time (s)',title='a  Excess time accumulates late')
sj,tj,zj=data['Joint'];speed=np.linalg.norm(zj['rows'][:,2:4],axis=1)
axes[0,1].plot(tj[::20],speed[::20],color='#235d88',lw=.7,label='Executed speed')
axes[0,1].plot(np.arange(len(zj['actions']))*.02,zj['actions'][:,1],color='#bc6c25',lw=.6,alpha=.7,label='Feed request')
axes[0,1].axhline(99,color='.3',lw=.7,ls='--');axes[0,1].set(xlabel='Complete-lap time (s)',ylabel='Speed (mm/s)',ylim=(93,100.3),title='b  Cruise detail (low corner speeds clipped)');axes[0,1].legend(loc='lower left',fontsize=8)
samples=[r for r in queries['samples'] if r['distance_to_corner_mm']>30 and r['time_s']>1]
times=[r['time_s'] for r in samples]
for label,color,field in [('Recorded request','#235d88',None),('Global inputs set to zero','#bc6c25','progress_gate_time_zero'),('Memory readout zero','#667760','zero_memory')]:
    y=[r['baseline'][1] if field is None else r['queries'][field][1] for r in samples]
    axes[1,0].plot(times,y,marker='.' if field=='zero_memory' else None,ms=3,ls='none' if field=='zero_memory' else '-',lw=.8,color=color,label=label)
axes[1,0].set(xlabel='Saved-state time (s)',ylabel='Feed request (mm/s)',title='c  Same-state diagnostic, no new motion');axes[1,0].legend(fontsize=8)
q=np.arange(4);width=.34
for shift,name,color in [(-width/2,'Joint','#235d88'),(width/2,'S2','#aaaaaa')]:
    values=[r['cruise99_s'] for r in audit['methods'][name]['quarters']]
    axes[1,1].bar(q+shift,values,width,color=color,label=name)
axes[1,1].set(xticks=q,xticklabels=['Q1','Q2','Q3','Q4'],ylabel='Time at >=99 mm/s (s)',title='d  Cruise loss is concentrated in Q3-Q4');axes[1,1].legend(fontsize=8)
for ext in ('png','pdf','svg'):fig.savefig(OUT/('saved_trace_diagnosis.'+ext),dpi=180)
plt.close(fig)
