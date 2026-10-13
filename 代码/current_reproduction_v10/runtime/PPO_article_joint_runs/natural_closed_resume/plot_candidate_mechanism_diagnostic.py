"""Exploratory raw-trajectory checks, not accepted manuscript figures."""
from pathlib import Path
import sys,json,shutil,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
R=OUT/'all34_dominance_research';CASE=BASE/'paper_evidence_review/natural_closed_case'
ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('fixed');args=ap.parse_args()
J=Path(args.candidate);F=Path(args.fixed)
assert json.loads((J/'full_result.json').read_text(encoding='utf-8'))['success']
assert json.loads((F/'full_result.json').read_text(encoding='utf-8'))['success']
STAGE=J/'mechanism_diagnostic';STAGE.mkdir(exist_ok=False)
shutil.copy2(Path(__file__),STAGE/'source.py')
geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
datasets={
    'Joint':(J/'full_trace.npz',J/'full_metrics.json'),
    'Fixed':(F/'full_trace.npz',F/'full_metrics.json'),
    'S2':(CASE/'s2_complete/trace.npz',CASE/'s2_paper_metrics.json')}
data={name:(np.load(p),json.loads(m.read_text(encoding='utf-8'))) for name,(p,m) in datasets.items()}
palette={'Joint':'#25659C','Fixed':'#4E8374','S2':'#C47C32'};styles={'Joint':'-','Fixed':'--','S2':':'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
    'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
ids=[next(i for i,c in enumerate(geometry['sharp_corners']) if c['id']==cid) for cid in geometry['mechanism_columns']];pieces={};quant=[]
for i in ids:
    c=geometry['sharp_corners'][i];cid=c['id'];ref=data['Joint'][0]['reference'];k=c['reference_vertex_index'];vertex=ref[k]
    incoming=ref[k]-ref[k-1];incoming/=np.linalg.norm(incoming)
    sign=np.sign(c['signed_heading_change_deg']);normal=sign*np.array([-incoming[1],incoming[0]])
    outgoing=(ref[k+1]-ref[k]);outgoing/=np.linalg.norm(outgoing);outgoing_local=np.array([outgoing@incoming,outgoing@normal])
    for name,(z,m) in data.items():
        corner=m['corners'][i];tc=corner['crossing_time_s'];rows=z['rows'];t=(np.arange(len(rows))+1)*.001-tc
        sel=(t>=-.35)&(t<=.50);rr=rows[sel];tt=t[sel];speed=np.linalg.norm(rr[:,2:4],axis=1)
        xy=(rr[:,:2]-vertex)@np.column_stack([incoming,normal])
        heading=np.rad2deg(np.unwrap(np.arctan2(rr[:,2:4]@normal,rr[:,2:4]@incoming)))
        curvature=sign*(rr[:,2]*rr[:,5]-rr[:,3]*rr[:,4])/np.maximum(speed**3,1e-12)
        pieces[(cid,name)]=dict(t=tt,xy=xy,heading=heading,curvature=curvature,speed=speed,outgoing=outgoing_local)
        if 'actions' in z.files:
            action_t=np.arange(len(z['actions']))*.020-tc;active=(action_t>=-.37)&(action_t<=.52)
            actions=z['actions'][active].copy();actions[:,0]*=sign;at=action_t[active]
            pieces[(cid,name)].update(action_t=at,actions=actions)
            with (STAGE/(cid+'_'+name+'_requests.csv')).open('w',encoding='utf-8',newline='') as f:
                w=csv.writer(f);w.writerow(['relative_time_s','steering_inward_positive_rad','feed_mm_s','preview_mm']);w.writerows(np.column_stack([at,actions]))
        with (STAGE/(cid+'_'+name+'_actual.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['relative_time_s','incoming_x_mm','inside_y_mm','actual_heading_deg','signed_curvature_mm_inv','speed_mm_s']);w.writerows(np.column_stack([tt,xy,heading,curvature,speed]))
        quant.append(dict(corner=cid,method=name,first_turn_inward=corner['first_turn_30mm_audit_inside'],
            first_turn_distance_mm=corner['first_turn_30mm_distance_before_mm'],minimum_speed=corner['minimum_speed_mm_s'],
            speed_deficit_mm=corner['speed_deficit_mm'],legacy_exit_heading_metric_not_used=True,
            reverse_heading_deg=corner['reverse_heading_deg']))
fig,axes=plt.subplots(1,3,figsize=(8.0,2.9),layout='constrained')
for col,i in enumerate(ids):
    c=geometry['sharp_corners'][i];cid=c['id'];ax=axes[col];out=pieces[(cid,'Joint')]['outgoing']
    xx,yy=np.meshgrid(np.linspace(-9,7,400),np.linspace(-2,10,300));pts=np.stack([xx,yy],axis=-1)
    left_distance=np.hypot(np.maximum(xx,0.),yy);along=np.maximum(pts@out,0.);right_distance=np.linalg.norm(pts-along[...,None]*out,axis=-1)
    ax.contourf(xx,yy,np.minimum(left_distance,right_distance),levels=[0,1.25],colors=['#F0F2F4'],zorder=0)
    ax.plot([-12,0,12*out[0]],[0,0,12*out[1]],color='#70777E',lw=.9,ls='-.',label='Reference')
    for name in ['S2','Fixed','Joint']:
        xy=pieces[(cid,name)]['xy'];ax.plot(xy[:,0],xy[:,1],styles[name],color=palette[name],lw=1.4 if name=='Joint' else 1.1,label=name)
    ax.scatter([0],[0],s=10,color='#555555',zorder=5)
    ax.set(xlim=(-9,7),ylim=(-2,10),xlabel='Incoming coordinate (mm)',ylabel='Inside coordinate (mm)' if col==0 else '',title=f'{c["interior_angle_deg"]:.0f}°  /  {cid.replace("corner_","C")}')
    ax.set_aspect('equal');ax.grid(alpha=.15,lw=.5)
axes[1].legend(loc='upper right',fontsize=7)
fig.savefig(STAGE/'local_shapes_diagnostic.png',dpi=220)
plt.close(fig)
fig,axes=plt.subplots(6,3,figsize=(7.2,9.1),sharex=True,layout='constrained')
labels=['Inward steering\n(deg)','Feed request\n(mm/s)','Preview (mm)','Actual heading\n(deg)','Inward curvature\n(1/mm)','Actual speed\n(mm/s)']
for col,i in enumerate(ids):
    c=geometry['sharp_corners'][i];cid=c['id']
    for name in ['S2','Fixed','Joint']:
        p=pieces[(cid,name)];style=dict(color=palette[name],ls=styles[name],lw=1.25 if name=='Joint' else 1.)
        if 'actions' in p:
            for row in range(3):axes[row,col].step(p['action_t'],np.rad2deg(p['actions'][:,row]) if row==0 else p['actions'][:,row],where='post',label=name,**style)
        for row,key in [(3,'heading'),(4,'curvature'),(5,'speed')]:axes[row,col].plot(p['t'],p[key],label=name,**style)
    axes[0,col].set_title(f'{c["interior_angle_deg"]:.0f}°  /  {cid.replace("corner_","C")}')
    axes[5,col].set_xlabel('Time from corner gate (s)')
    for row in range(6):
        ax=axes[row,col];ax.axvline(0,color='#A0A6AC',lw=.6,zorder=0);ax.set_xlim(-.35,.50);ax.grid(axis='y',alpha=.16,lw=.5)
        if col==0:ax.set_ylabel(labels[row])
    for row in [1,5]:axes[row,col].set_ylim(20,105)
    axes[2,col].set_ylim(.5,12.5)
axes[0,2].legend(loc='upper right',fontsize=7)
axes[3,2].legend(loc='lower right',fontsize=7)
fig.savefig(STAGE/'three_action_chain_diagnostic.png',dpi=200)
plt.close(fig)
write_json(STAGE/'binding.json',dict(kind='exploratory_mechanism_visual_check',manuscript_figure=False,goal_complete=False,
    source_sha256={str(p):sha(p) for paths in datasets.values() for p in paths},script_sha256=sha(Path(__file__)),
    corners=ids,claim_to_check='Actual inward transitions and observed requested/executed coordination; no guarantee of exact tangency.',
    alignment='Original witnessed corner-bisector crossing time for every trace; rigid local frame with turn handedness normalized for both requested steering and actual heading/curvature. No smoothing, filtering or resampling.',
    artifact_scope='Working diagnostics only. Publication visual-precedent and full figure QA remain pending; these do not replace any manuscript figure.',
    metrics=quant))
print('DIAGNOSTIC_PLOTS',STAGE,flush=True)
