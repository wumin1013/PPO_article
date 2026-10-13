"""Reorganize manuscript figures from bound recorded data; no simulations."""
import sys
import json
import hashlib
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OLD = Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1/paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision')
OUT = OLD / 'figure_revision_20260924'
FIG = OUT / 'figures'
sys.path.insert(0, str(ROOT.parent))
import draw_all34_publication_figures as previous
common = previous.common

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

METHODS = ['S2', 'Two-action', 'J-NNC']
KEYS = dict(zip(METHODS, ['S2', 'Fixed', 'Joint']))
COLORS = {'S2':'#D27C26', 'Two-action':'#82748C', 'J-NNC':'#17689A', 'NNC':'#B94C40'}
LIMITS = np.array([100., 2000., 62500.])
YLABELS = [r'$\|v\|/V_{\max}$', r'$\|a\|/A_{\max}$', r'$\|j\|/J_{\max}$']

def line(ax, x, y, name, **kwargs):
    ax.plot(x, y, color=COLORS[name], ls=common.STYLES[KEYS[name]], label=name, **kwargs)

def norms(d):
    return np.stack([np.linalg.norm(d['rows'][:,2+2*i:4+2*i], axis=1) for i in range(3)], axis=1)

def figure_nnc(binding, data, nnc_binding):
    fig, axes = plt.subplots(2,2,figsize=(7.25,4.9), gridspec_kw=dict(hspace=.4,wspace=.25))
    rows=[]
    peaks={}
    for col, case in enumerate(['original','dense']):
        item=nnc_binding['records'][case]
        assert sha(item['trace'])==item['trace_sha256']
        z=np.load(item['trace'])
        dt=.001
        v=np.diff(z['positions'],axis=0)/dt
        a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
        j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
        actual=np.stack([np.linalg.norm(a,axis=1)/2000,np.linalg.norm(j,axis=1)/62500],axis=1)
        assert np.allclose(actual[:,0]*2000,z['acceleration'])
        assert np.allclose(actual[:,1]*62500,z['jerk'])
        n=len(actual); t=np.arange(1,n+1)*dt
        main=norms(data[case]['J-NNC'])[:n,1:]/LIMITS[1:]
        peaks[case]={'acceleration_multiple':float(actual[:,0].max()),'jerk_multiple':float(actual[:,1].max()),
                     'prefix_duration_s':float(t[-1]),'post20ms_multiple':actual[20:].max(axis=0).tolist()}
        for k in range(n): rows.append([col,t[k],*actual[k],*main[k]])
        for row in range(2):
            ax=axes[row,col]
            shown=t<=.080000001
            ax.plot(t[shown]*1000,actual[shown,row],color=COLORS['NNC'],lw=.9,label='NNC')
            ax.plot(t[shown]*1000,main[shown,row],color=COLORS['J-NNC'],lw=1.0,label='J-NNC')
            ax.axvspan(0,20,color='#EDF1F4',zorder=-4)
            ax.axhline(1,color='#3F4549',ls='--',lw=.8,label='Common limit')
            ax.set_yscale('symlog',linthresh=1,linscale=.7)
            ax.set(xlim=(0,80),ylim=(-.08,45 if row==0 else 2000),xticks=[0,20,40,60,80],
                   yticks=[0,1,10,30] if row==0 else [0,1,10,100,1000])
            ax.set_yticklabels(['0','1','10','30'] if row==0 else ['0','1','10','100','1000'])
            if col==0: ax.set_ylabel(YLABELS[row+1])
            if row==1: ax.set_xlabel('Elapsed time (ms)')
            common.panel(ax,chr(97+row*2+col),('Long route' if case=='original' else 'Dense case')+(' — acceleration' if row==0 else ' — jerk'))
            ax.text(.98,.94,f'NNC peak: {actual[shown,row].max():.1f} × limit',transform=ax.transAxes,
                    ha='right',va='top',fontsize=8,bbox=dict(fc='white',ec='none',alpha=.9,pad=1.5))
            ax.grid(axis='y',color='#E3E7EA',lw=.4)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.52,1.015))
    fig.subplots_adjust(left=.1,right=.98,top=.87,bottom=.11)
    common.save(fig,'Figure_5',FIG)
    np.savetxt(FIG/'Figure_5_source_data.csv',rows,delimiter=',',comments='',header='case_0_long_1_dense,time_s,NNC_A_use,NNC_J_use,JNNC_A_use,JNNC_J_use')
    return peaks

def figure_full(data):
    fig=plt.figure(figsize=(7.25,6.8))
    grid=fig.add_gridspec(4,3,height_ratios=[1,1,1,1.1],hspace=.65,wspace=.18)
    plotted=[]
    for col,name in enumerate(METHODS):
        d=data[name]; y=norms(d)/LIMITS
        for row in range(3):
            ax=fig.add_subplot(grid[row,col])
            x,z=common.extrema_envelope(d['t'],y[:,row])
            ax.plot(x,z,color=COLORS[name],lw=.7)
            plotted.extend([col,row,float(xx),float(yy)] for xx,yy in zip(x,z))
            ax.axhline(1,color='#4B5054',ls='--',lw=.7)
            ax.set(xlim=(0,447),ylim=(-.03,1.08),xticks=[0,200,400],yticks=[0,.5,1])
            if col==0: ax.set_ylabel(YLABELS[row])
            else: ax.tick_params(labelleft=False)
            if row==0:common.panel(ax,chr(97+col),name)
            if row==2:ax.set_xlabel('Elapsed time (s)')
    ax=fig.add_subplot(grid[3,:])
    stop=[]
    for name in METHODS:
        d=data[name]; y=norms(d)[:,0]/100
        line(ax,d['t'][-171:]-d['t'][-1],y[-171:],name,lw=1.2)
        stop.extend([METHODS.index(name),float(x),float(z)] for x,z in zip(d['t'][-171:]-d['t'][-1],y[-171:]))
    ax.axvspan(-.02,0,color='#EDF1F4',zorder=-3)
    ax.text(-.01,.7,'≥20 ms\nrest',ha='center',fontsize=7.5)
    ax.set(xlabel='Time relative to complete stop (s)',ylabel=YLABELS[0],xlim=(-.17,.003),ylim=(-.03,1.07),yticks=[0,.5,1])
    common.panel(ax,'d','Final braking and physical rest')
    ax.legend(loc='lower left',ncol=3)
    fig.subplots_adjust(left=.105,right=.985,top=.94,bottom=.085)
    common.save(fig,'Figure_6',FIG)
    np.savetxt(FIG/'Figure_6_display_data.csv',plotted,delimiter=',',comments='',header='method_S2_Two_JNNC,quantity_V_A_J,time_s,utilization')
    np.savetxt(FIG/'Figure_6_stop_data.csv',stop,delimiter=',',comments='',header='method_S2_Two_JNNC,time_relative_stop_s,speed_use')

def figure_local(ref,geometry,data):
    fig=plt.figure(figsize=(7.25,8.1))
    grid=fig.add_gridspec(4,3,height_ratios=[1.45,1,1,1],hspace=.55,wspace=.25)
    values=[]; paths=[]; summary={}
    for col,cid in enumerate(['corner_01','corner_02','corner_06']):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid)
        vertex,rot,u=common.local_geometry(ref,c)
        ax=fig.add_subplot(grid[0,col])
        common.corridor(ax,u,(-6,4),(-1.5,8.5))
        for name in METHODS:
            xy=common.local_curve(data[name],vertex,rot,c,window=20)
            line(ax,*xy.T,name,lw=1.25)
            paths.extend([col,METHODS.index(name),float(x),float(y)] for x,y in xy)
        ax.set(xlabel='Incoming-axis x (mm)',xticks=[-5,0,4],yticks=[0,4,8])
        if col==0:ax.set_ylabel('Inward-normal y (mm)')
        common.panel(ax,chr(97+col),f'{c["interior_angle_deg"]:.0f}° / C{cid[-2:]}')
        summary[cid]={}
        for name in METHODS:
            d=data[name]; s=d['progress']-c['reference_s_mm']; mask=np.abs(s)<=10
            y=norms(d)/LIMITS
            summary[cid][name]={'deficit_mm':float(.001*np.maximum(100-y[mask,0]*100,0).sum()),
                                'min_speed_mm_s':float(y[mask,0].min()*100)}
            show=np.abs(s)<=20
            values.extend([col,METHODS.index(name),float(x),*map(float,z)] for x,z in zip(s[show],y[show]))
        local_axes=[fig.add_subplot(grid[row,col]) for row in range(1,4)]
        for name in METHODS:
            d=data[name]; s=d['progress']-c['reference_s_mm']; mask=np.abs(s)<=20; y=norms(d)/LIMITS
            for row,ax in enumerate(local_axes):line(ax,s[mask],y[mask,row],name,lw=1.0)
        for row,ax in enumerate(local_axes):
            ax.axhline(1,color='#4B5054',ls='--',lw=.65)
            ax.axvline(0,color='#CDD3D8',lw=.65,zorder=-3)
            ax.set(xlim=(-20,20),ylim=(-.035,1.085),xticks=[-20,-10,0,10,20],yticks=[0,.5,1])
            if col==0:ax.set_ylabel(YLABELS[row])
            else:ax.tick_params(labelleft=False)
            if row<2:ax.tick_params(labelbottom=False)
            else:ax.set_xlabel('Progress from corner (mm)')
            if row==0:common.panel(ax,chr(100+col),'Speed use')
    handles=[Line2D([],[],color=COLORS[n],ls=common.STYLES[KEYS[n]],label=n) for n in METHODS]
    handles.append(Line2D([],[],color='#7C858C',ls='--',label='Reference / limit'))
    fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.53,1.005))
    fig.subplots_adjust(left=.1,right=.98,top=.93,bottom=.065)
    common.save(fig,'Figure_7',FIG)
    np.savetxt(FIG/'Figure_7_source_data.csv',values,delimiter=',',comments='',header='corner_C01_C02_C06,method_S2_Two_JNNC,progress_relative_mm,V_use,A_use,J_use')
    np.savetxt(FIG/'Figure_7_xy_data.csv',paths,delimiter=',',comments='',header='corner_C01_C02_C06,method_S2_Two_JNNC,local_x_mm,local_y_mm')
    return summary

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    binding=read(OLD/'paper_result_binding.json'); nnc=read(OLD/'nnc_evidence_binding.json')
    contract=dict(backend='Python/matplotlib',new_simulation=False,policy_changed=False,
        user_design_priority='Use the approved original18page path-aligned utilization style. User-directed precedent supersedes external style search; no external published precedent claimed.',
        figure5=dict(conclusion='Recorded direct NNC commands violate common vector A/J bounds on both cases.',evidence='Plot identical first80ms windows from retained1s prefixes; independent XY differences; common1ms sampling. Shading marks initial20ms; displayed peak annotations refer only to plotted80ms.',geometry='Two cases by two constraint ratios; same row scales; symlog with linear segment below1.'),
        figure6=dict(conclusion='Full long-route J-NNC execution remains within V/A/J limits and completes braking.',evidence='Complete records; all extrema retained for display.',geometry='Three method columns with normalized V/A/J rows and a full-width terminal panel.'),
        figure7=dict(conclusion='Representative inward trajectories reduce local speed loss while satisfying vector constraints.',evidence='Already prescribed C01/C02/C06, not newly selected by ranking; whole-window metrics and all34 results remain.',geometry='135/90/45degree columns; local trajectory above path-aligned normalized V/A/J; no curve smoothing.'),
        exports='PDF, editable SVG, PNG; source CSV, original trace SHA256 bindings',statistics='Deterministic selected1ms records; no error bars or inferred repetitions',protected_original_figures='1--3 unchanged')
    write(OUT/'figure_contract.json',contract)
    data={};refs={};geometries={}
    used={}
    for case in ['original','dense']:
        refs[case]=np.loadtxt(binding['cases'][case]['reference'],delimiter=',',skiprows=1)
        geometries[case]=read(binding['cases'][case]['geometry'])
        data[case]={}
        for name in METHODS:
            item=binding['methods'][name][case][0]
            assert item['dt_s']==.001 and sha(item['trace'])==item['trace_sha256']
            used[item['trace']]=item['trace_sha256']
            data[case][name]=previous.load_trace(item['trace'],refs[case])
    for i in range(1,5):shutil.copy2(OLD/'latex_source_en/figures_lowdim_revision'/f'Figure_{i}.pdf',FIG/f'Figure_{i}.pdf')
    for new,old in [(8,7),(9,8)]:
        shutil.copy2(OLD/'latex_source_en/figures_lowdim_revision'/f'Figure_{old}.pdf',FIG/f'Figure_{new}.pdf')
    peaks=figure_nnc(binding,data,nnc)
    figure_full(data['original'])
    summary=figure_local(refs['original'],geometries['original'],data['original'])
    write(OUT/'new_figure_metrics.json',dict(nnc=peaks,corners=summary))
    write(OUT/'figure_data_binding.json',dict(source_binding=str(OLD/'paper_result_binding.json'),source_binding_sha256=sha(OLD/'paper_result_binding.json'),
        nnc_binding=str(OLD/'nnc_evidence_binding.json'),nnc_binding_sha256=sha(OLD/'nnc_evidence_binding.json'),
        source_sha256=used,script_sha256=sha(__file__),outputs={p.name:sha(p) for p in FIG.iterdir()},
        figure_mapping={'4':'old4a-c','5':'new NNC prefix constraint failure','6':'old4d-g normalized','7':'old5 local trajectory plus vector utilization','8':'old7','9':'old8'},
        retired_main_figures=['old6 action-mechanism grid','old9 efficiency plot'],protected_originals_unchanged=True))
    print(json.dumps(dict(output=str(OUT),metrics=read(OUT/'new_figure_metrics.json')),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
