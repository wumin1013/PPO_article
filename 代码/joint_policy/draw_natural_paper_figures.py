"""Nine evidence-bound figures; raw data are never smoothed for appearance."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from study_io import BASE,sha,write_json
from natural_case_metrics import ordered_projection

CASE=BASE/'paper_evidence_review/natural_closed_case'
OUT=BASE/'manuscript_joint_v1/natural_case_figures'
COLORS={'Joint':'#246A9A','S2':'#D17C2F','Fixed':'#43816B'}
STYLES={'Joint':'-','S2':'--','Fixed':':'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
    'axes.titlesize':9,'axes.spines.right':False,'axes.spines.top':False,
    'legend.frameon':False,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
    'lines.linewidth':1.1,'savefig.facecolor':'white'})


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight')
    plt.close(fig)


def label(ax,letter):ax.text(-.10,1.035,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)


def envelope(x,y,width=200):
    ids=[]
    for i in range(0,len(y),width):
        b=y[i:i+width];ids.extend([i,i+int(np.argmin(b)),i+int(np.argmax(b)),i+len(b)-1])
    ids=np.unique(ids)
    return x[ids],y[ids]


def box(ax,x,y,w,h,text,color='#EDF3F7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.015',
                             facecolor=color,edgecolor='#668299',linewidth=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)


def arrow(ax,a,b,text=None):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1,color='#435769'))
    if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.025,text,ha='center',fontsize=7)


def main():
    OUT.mkdir(exist_ok=True)
    binding=read(CASE/'paper_result_binding.json');geometry=read(CASE/'geometry.json')
    refs=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    data={}
    for name,item in binding['primary_methods'].items():
        path=BASE/item['trace'];assert sha(path)==item['trace_sha256']
        z=np.load(path);rows=z['rows']
        data[name]=dict(rows=rows,t=(np.arange(len(rows))+1)*.001,
            speed=np.linalg.norm(rows[:,2:4],axis=1),actions=z['actions'] if 'actions' in z else None,
            progress=ordered_projection(refs,rows[:,:2]),metrics=read(BASE/item['metrics']),result=read(BASE/item['result']))
    # The user-prescribed relationships and panel sequence govern all layouts.
    contract=dict(backend='Python/matplotlib',source_binding_sha256=sha(CASE/'paper_result_binding.json'),
        design_basis='User-specified nine-figure contract and existing project figure grammar; no new external visual precedent claimed.',
        raw_xy_smoothed=False,curve_display='Original points; extrema-preserving envelope only for full-duration signals.',
        color_roles=COLORS,N_path=1,N_model_per_method=1,N_independent_full_training_replications=0,
        figures={
            'Figure_1':'Two separate planning chains; distinguish learned coordination from explicit execution rules.',
            'Figure_2':'Spatial hero plus all34 angle coverage and local tolerance; static geometry explicitly labeled.',
            'Figure_3':'Learning, decision and execution lanes with actual timing and returned measurements.',
            'Figure_4':'Full XY, fixed local XY and complete V/A/J, including stopping.',
            'Figure_5':'The three pre-registered corner windows, equal aspect and common scales.',
            'Figure_6':'Three corner columns by six physical/action rows, seconds aligned to gate crossing.',
            'Figure_7':'Full-route speed, all34 deficit and99mm/s spatial coverage; no independent-angle inference.',
            'Figure_8':'GlobalD and conditional training opportunity plus full-case support selection outcomes.',
            'Figure_9':'Singleton full-time and cruise comparison; no invented seed error bars.'})
    write_json(OUT/'figure_contract.json',contract)
    fig,ax=plt.subplots(figsize=(7,3.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.01,.93,'Conventional two-stage planning',fontweight='bold',color=COLORS['S2'])
    for x,txt in zip((.03,.37,.71),('Smooth reference\nwithin the corridor','Schedule feed\nunder vector limits','Execute and verify\nthe complete lap')):box(ax,x,.68,.25,.16,txt,'#FBF2E8')
    arrow(ax,(.29,.76),(.355,.76));arrow(ax,(.63,.76),(.695,.76))
    ax.text(.01,.52,'Learned joint coordination',fontweight='bold',color=COLORS['Joint'])
    for x,txt in zip((.03,.37,.71),('Observe local geometry,\nmotion and memory','Choose steering,\nfeed and preview','Guide, constrain,\nbrake and execute')):box(ax,x,.25,.25,.16,txt)
    arrow(ax,(.29,.33),(.355,.33));arrow(ax,(.63,.33),(.695,.33))
    ax.plot([.83,.83,.15,.15],[.23,.10,.10,.23],color='#435769',lw=1)
    arrow(ax,(.15,.10),(.15,.24));ax.text(.5,.065,'Return actual XY, velocity and acceleration',ha='center',fontsize=8)
    save(fig,'Figure_1')
    fig=plt.figure(figsize=(7,4.8));gs=fig.add_gridspec(2,3,width_ratios=[1.1,1.1,1],hspace=.42,wspace=.36)
    ax=fig.add_subplot(gs[:,:2]);ax.plot(refs[:,0]/1000,refs[:,1]/1000,color='#50616D',lw=.9)
    for p in geometry['primitives']:
        ids=slice(p['start_reference_index'],p['end_reference_index']+1)
        col='#5089A8' if p['kind']=='arc' else ('#9BA8AC' if p['length_mm']<=700 else '#495866')
        ax.plot(refs[ids,0]/1000,refs[ids,1]/1000,color=col,lw=1.8 if p['kind']=='arc' else 1)
    for n,cid in zip(('B:135°','C:90°','D:45°'),geometry['mechanism_columns']):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid);xy=np.asarray(c['point_mm'])/1000
        ax.scatter(*xy,s=17,color=COLORS['Joint']);ax.annotate(n,xy,xytext=(5,7),textcoords='offset points',fontsize=7)
    ax.scatter(*refs[0]/1000,marker='s',s=25,color='#222222');ax.text(1.5,7.35,'Start / stop',fontsize=7)
    ax.set(aspect='equal',xlabel='X (m)',ylabel='Y (m)');label(ax,'a')
    ax=fig.add_subplot(gs[0,2]);angles=np.array([c['interior_angle_deg'] for c in geometry['sharp_corners']])
    ax.scatter(np.arange(34),angles,s=13,color='#495866');ax.axhline(90,color='#D17C2F',lw=.7,ls='--')
    ax.set(xlabel='Corner index',ylabel='Included angle β (°)',ylim=(35,185));label(ax,'b')
    ax=fig.add_subplot(gs[1,2]);ax.add_patch(Polygon([[-4,-1.25],[-4,1.25],[-1.25,1.25],[-1.25,4],[1.25,4],[1.25,-1.25]],facecolor='#DDE8EF',edgecolor='none'))
    ax.plot([-4,0,0],[0,0,4],color='#5B6470',ls='--');ax.plot([-2,2],[2,-2],color='#8898A2',lw=.7)
    ax.text(-3.8,-2.3,'ε = 1.25 mm',fontsize=8);ax.text(-3.8,-3.3,'Static geometry',fontsize=7)
    ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal',xlabel='Local x (mm)',ylabel='Local y (mm)');label(ax,'c')
    save(fig,'Figure_2')
    fig,ax=plt.subplots(figsize=(7,4.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.01,.95,'Learning and selection provenance',fontweight='bold')
    for x,txt in zip((.02,.36,.70),('Historical PPO\ninteraction updates','Fit offline and search\nclosed-loop parameters','Freeze checkpoint and\nobservation-support setting')):box(ax,x,.72,.27,.14,txt,'#F0F0ED')
    arrow(ax,(.30,.79),(.35,.79));arrow(ax,(.64,.79),(.69,.79))
    ax.text(.01,.57,'20 ms decisions',fontweight='bold',color=COLORS['Joint'])
    box(ax,.03,.33,.25,.16,'Encode observations\nand causal memory');box(ax,.38,.33,.25,.16,'Evaluate one policy\nand hold three actions')
    arrow(ax,(.29,.41),(.36,.41));ax.text(.325,.515,'72-D state',ha='center',fontsize=7)
    box(ax,.73,.33,.24,.16,'Update guide and\nvector constraints\nevery 1 ms')
    arrow(ax,(.65,.41),(.71,.41));ax.text(.68,.515,'α, F, D',ha='center',fontsize=7)
    arrow(ax,(.83,.695),(.52,.505));ax.text(.78,.605,'Frozen weights',ha='center',fontsize=7)
    ax.plot([.85,.85,.15,.15],[.31,.16,.16,.31],color='#435769',lw=1);arrow(ax,(.15,.16),(.15,.32))
    ax.text(.5,.105,'Return simulated physical state; audit all commands, gates and stopping',ha='center',fontsize=8)
    ax.text(.5,.02,'Separate sensitivity run: 0.5 ms execution with the same 20 ms decisions',ha='center',fontsize=8)
    save(fig,'Figure_3')
    fig=plt.figure(figsize=(7,7.1));gs=fig.add_gridspec(5,2,height_ratios=[1.9,1,1,1,1],hspace=.6,wspace=.35)
    ax=fig.add_subplot(gs[0,:]);ax.plot(refs[:,0]/1000,refs[:,1]/1000,color='#A4A8AB',lw=.8)
    for name,d in data.items():ax.plot(d['rows'][::12,0]/1000,d['rows'][::12,1]/1000,STYLES[name],color=COLORS[name],label=name,lw=.9)
    ax.set(aspect='equal',xlabel='X (m)',ylabel='Y (m)');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.22));label(ax,'a')
    c=next(c for c in geometry['sharp_corners'] if c['id']=='corner_02');center=np.array(c['point_mm'])
    ax=fig.add_subplot(gs[1,0]);ax.plot([-20,0,0],[0,0,20],color='#A4A8AB',lw=.9)
    for name,d in data.items():
        xy=d['rows'][:,:2]-center;keep=np.linalg.norm(xy,axis=1)<30
        ax.plot(xy[keep,0],xy[keep,1],STYLES[name],color=COLORS[name])
    ax.set(xlim=(-20,20),ylim=(-20,20),aspect='equal',xlabel='Local x (mm)',ylabel='Local y (mm)');label(ax,'b')
    ax=fig.add_subplot(gs[1,1]);
    for name,d in data.items():ax.plot(d['t'][-300:]-d['t'][-1],d['speed'][-300:],STYLES[name],color=COLORS[name])
    ax.set(xlabel='Time before end (s)',ylabel='Speed (mm/s)');label(ax,'c')
    for j,(cols,title,limit) in enumerate(((slice(2,4),'V (mm/s)',100),(slice(4,6),'A (mm/s²)',2000),(slice(6,8),'J (mm/s³)',62500))):
        ax=fig.add_subplot(gs[j+2,:])
        for name,d in data.items():
            yy=np.linalg.norm(d['rows'][:,cols],axis=1);xx,yy=envelope(d['t'],yy,180)
            ax.plot(xx,yy,STYLES[name],color=COLORS[name],lw=.6,alpha=.8)
        ax.axhline(limit,color='#333333',ls='--',lw=.7);ax.set(ylabel=title,xlim=(0,490));label(ax,chr(ord('d')+j))
    ax.set_xlabel('Complete elapsed time (s)');save(fig,'Figure_4')
    fig,axes=plt.subplots(1,3,figsize=(7,2.8))
    for col,(ax,cid) in enumerate(zip(axes,geometry['mechanism_columns'])):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid);k=c['reference_vertex_index'];u=refs[k]-refs[k-1];u/=np.linalg.norm(u)
        rot=np.stack([u,[-u[1],u[0]]],axis=1);delta=np.deg2rad(c['signed_heading_change_deg']);out=np.array([np.cos(delta),np.sin(delta)])
        ref=np.array([[-30,0],[0,0],30*out]);ax.plot(ref[:,0],ref[:,1],color='#79838A',ls='--',lw=.8)
        # Uniform-width corridor bands convey the declared Euclidean tolerance.
        for a,b in zip(ref[:-1],ref[1:]):
            e=b-a;n=np.array([-e[1],e[0]])/np.linalg.norm(e)*1.25
            ax.add_patch(Polygon([a+n,b+n,b-n,a-n],color='#E5EBEF',zorder=0))
        for name,d in data.items():
            xy=(d['rows'][:,:2]-refs[k])@rot;mask=np.linalg.norm(xy,axis=1)<30
            ax.plot(xy[mask,0],xy[mask,1],STYLES[name],color=COLORS[name],label=name)
        ax.set(xlim=(-20,20),ylim=(-20,20),aspect='equal',xlabel='Incoming-axis x (mm)',title=f"{c['category'].replace('_',' ').title()}  β={c['interior_angle_deg']:.0f}°")
        ax.text(.04,.05,f"δ=+{c['signed_heading_change_deg']:.0f}°",transform=ax.transAxes,fontsize=7);label(ax,chr(97+col))
        cm=next(x for x in data['Joint']['metrics']['corners'] if x['id']==cid)
        ax.text(.04,.25,f"First turn: {cm['first_turn_30mm_distance_before_mm']:.1f} mm upstream",transform=ax.transAxes,fontsize=6.4,va='top')
    axes[0].set_ylabel('Left-normal y (mm)');axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.60),ncol=3)
    save(fig,'Figure_5')
    fig,axes=plt.subplots(6,3,figsize=(7,8.4),sharex='col');fig.subplots_adjust(hspace=.18,wspace=.30)
    for col,cid in enumerate(geometry['mechanism_columns']):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid);k=c['reference_vertex_index'];u=refs[k]-refs[k-1];u/=np.linalg.norm(u)
        ax=axes[0,col];ax.set_title(f"β={c['interior_angle_deg']:.0f}°, δ=+{c['signed_heading_change_deg']:.0f}°")
        for name,d in data.items():
            cm=next(x for x in d['metrics']['corners'] if x['id']==cid);tc=cm['crossing_time_s']
            first,last=cm['first_index'],cm['last_index'];mask=slice(first,last+1)
            t=d['t'][mask]-tc;v=d['rows'][mask,2:4];a=d['rows'][mask,4:6];sv=d['speed'][mask]
            h=np.unwrap(np.arctan2(v[:,1],v[:,0])-np.arctan2(u[1],u[0]))*180/np.pi
            h=(h+180)%360-180
            curvature=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(sv,5)**3
            for row,yy in ((3,h),(4,curvature),(5,sv)):
                axes[row,col].plot(t,yy,STYLES[name],color=COLORS[name],label=name)
            if name=='Joint':
                action=d['actions'];ta=np.arange(len(action))*.02-tc;keep=(ta>=-.17)&(ta<=.30)
                for row,yy in ((0,np.rad2deg(action[:,0])),(1,action[:,1]),(2,action[:,2])):
                    axes[row,col].step(ta[keep],yy[keep],where='post',color=COLORS[name])
        for row in range(6):axes[row,col].axvline(0,color='#A5A5A5',lw=.5);axes[row,col].set_xlim(-.17,.22)
        axes[3,col].axhline(c['signed_heading_change_deg'],color='#AAAAAA',ls=':',lw=.7)
        for threshold in (95,99):axes[5,col].axhline(threshold,color='#AAAAAA',ls=':',lw=.5)
        axes[0,col].set_ylim(-23,23);axes[1,col].set_ylim(0,105);axes[2,col].set_ylim(0,13);axes[3,col].set_ylim(-8,145);axes[4,col].set_ylim(-.12,1.8);axes[5,col].set_ylim(0,105)
        axes[5,col].set_xlabel('Time from gate crossing (s)')
    for row,name in enumerate(('Steering (°)','Requested feed\n(mm/s)','Preview D (mm)','Heading (°)','Curvature\n(mm⁻¹)','Actual speed\n(mm/s)')):axes[row,0].set_ylabel(name)
    axes[5,1].legend(loc='upper center',bbox_to_anchor=(.5,-.60),ncol=3)
    save(fig,'Figure_6')
    fig,axes=plt.subplots(3,1,figsize=(7,6.3),gridspec_kw={'height_ratios':[1.2,1.1,1]},constrained_layout=True)
    for name,d in data.items():
        x,y=envelope(d['progress']/1000,d['speed'],180);axes[0].plot(x,y,STYLES[name],color=COLORS[name],label=name,lw=.6)
        angles=[x['speed_deficit_mm'] for x in d['metrics']['corners']]
        axes[1].plot(np.arange(34),angles,STYLES[name],color=COLORS[name],marker='o',ms=2.5)
    axes[0].set(xlabel='Reference progress (m)',ylabel='Speed (mm/s)',ylim=(0,104));axes[0].axhline(99,color='#777777',lw=.7,ls=':');axes[0].legend(ncol=3)
    axes[1].set(xlabel='All registered corners (path order)',ylabel='Speed deficit (mm)',xticks=range(0,34,3))
    for j,(name,d) in enumerate(data.items()):
        indices=np.flatnonzero(d['speed']>=99)[::8];axes[2].plot(d['progress'][indices]/1000,np.full(len(indices),j),linestyle='',marker='|',ms=10,color=COLORS[name])
    axes[2].set(yticks=range(len(data)),yticklabels=list(data),xlabel='Reference progress (m)',ylim=(-.6,2.6),xlim=(0,44.421))
    for ax,l in zip(axes,'abc'):label(ax,l)
    save(fig,'Figure_7')
    fig,axes=plt.subplots(1,3,figsize=(7,3.1),constrained_layout=True)
    train=read(CASE/'fixed_conditional_training/training.json')['trials']
    grid=train[:7];axes[0].plot([r['spec']['fixed_d'] for r in grid],[3-r['key'][0] for r in grid],'-o',color=COLORS['Fixed'],ms=4)
    axes[0].set(xlabel='One global D (mm)',ylabel='Complete training snippets / 3',ylim=(-.1,3.3),yticks=(0,1,2,3));label(axes[0],'a')
    for r in train:
        if r['key'][0]==0:axes[1].scatter(r['trial'],r['key'][2],color=COLORS['Fixed'],s=17)
        else:axes[1].scatter(r['trial'],4.15,marker='x',color='#A55A50',s=20)
    axes[1].axhline(4.15,color='#CCCCCC',lw=.5);axes[1].text(.02,.91,'× = incomplete set',transform=axes[1].transAxes,fontsize=7)
    axes[1].set(xlabel='Conditional candidate',ylabel='Three-snippet time (s)');label(axes[1],'b')
    fixed=read(CASE/'fixed_support_calibration/results.json')
    vals=[read(CASE/'fixed_historical/result.json')['complete_time_s']]+[r['complete_time_s'] for r in fixed]
    labs=['Raw','Clip .141','Clip .20','Nonlocal'][:len(vals)]
    axes[2].scatter(np.arange(len(vals)),vals,color=COLORS['Fixed'],s=23)
    axes[2].axhline(data['Joint']['t'][-1],color=COLORS['Joint'],ls='-',lw=1)
    axes[2].set(xticks=range(len(labs)),xticklabels=labs,ylabel='Full-lap time (s)');axes[2].tick_params(axis='x',rotation=55);label(axes[2],'c')
    save(fig,'Figure_8')
    fig,axes=plt.subplots(1,3,figsize=(7,3),constrained_layout=True)
    names=list(data)
    for j,name in enumerate(names):
        d=data[name];m=d['metrics'];c=COLORS[name]
        axes[0].scatter(j,d['t'][-1],color=c,s=40);axes[0].text(j,d['t'][-1]+1.0,f"{d['t'][-1]:.3f}",ha='center',fontsize=7)
        axes[1].bar(j,100*m['cruise']['99']['distance_fraction'],color=c,width=.5)
        axes[2].bar(j,m['cruise']['99']['longest_time_s'],color=c,width=.5)
    axes[0].axhline(442.725521,color='#999999',ls=':',lw=.8)
    axes[0].set(ylabel='Complete time (s)',ylim=(440,max(d['t'][-1] for d in data.values())+6))
    axes[1].set(ylabel='≥99 mm/s distance coverage (%)',ylim=(0,105));axes[2].set(ylabel='Longest ≥99 mm/s interval (s)')
    for ax,l in zip(axes,'abc'):ax.set_xticks(range(len(names)),names);label(ax,l)
    fig.suptitle('One registered path; one model per method; no independent-seed error bars',fontsize=8,y=1.06)
    save(fig,'Figure_9')
    files={p.name:sha(p) for p in OUT.glob('Figure_*.*')}
    write_json(OUT/'receipt.json',dict(status='generated_nine_main_figures',files_sha256=files,
        source_binding_sha256=sha(CASE/'paper_result_binding.json'),figure_contract_sha256=sha(OUT/'figure_contract.json'),visual_qa_complete=False))


if __name__=='__main__':main()
