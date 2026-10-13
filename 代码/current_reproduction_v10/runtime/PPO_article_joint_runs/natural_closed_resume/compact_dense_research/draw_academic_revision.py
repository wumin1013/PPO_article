"""Editorial revision of recorded-data figures. No model or simulation changes."""
import sys,json,hashlib,shutil,inspect,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from revise_original_vectors_academic import ROOT,OLD,PREV,OUT,FIG,WORK,sha,write
sys.path.insert(0,str(ROOT.parent))
import draw_all34_publication_figures as prev
import revise_nnc_utilization_figures as util
common=prev.common
ORDER=['J-NNC','Two-action','S2'];KEY={'J-NNC':'Joint','Two-action':'Fixed','S2':'S2'}
LABEL={'J-NNC':'J-NNC','Two-action':'Two-action','S2':'Two-step planner'}
COL={'J-NNC':'#17689A','Two-action':'#BE7A32','S2':'#727C85','NNC':'#B85248'}
STYLE={'J-NNC':'-','Two-action':(0,(1.6,1.5)),'S2':(0,(5,2))}
LIMIT=np.array([100.,2000.,62500.])
for n in ORDER:
    common.COLORS[KEY[n]]=COL[n];common.NAMES[KEY[n]]=LABEL[n];common.STYLES[KEY[n]]=STYLE[n]
common.ORDER[:]=[KEY[n] for n in ORDER]
plt.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,
    'axes.titleweight':'normal','xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42})
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def norms(d):return np.stack([np.linalg.norm(d['rows'][:,2+2*i:4+2*i],axis=1) for i in range(3)],axis=1)
def line(ax,x,y,n,lw=None,**kw):
    return ax.plot(x,y,color=COL[n],ls=STYLE[n],lw=lw or (1.5 if n=='J-NNC' else 1.1),
        label=LABEL[n],zorder=5 if n=='J-NNC' else 3,**kw)
def legend(fig,reference=False,anchor=(.52,1.01),names=ORDER):
    h=[Line2D([],[],color=COL[n],ls=STYLE[n],lw=1.5 if n=='J-NNC' else 1.1,label=LABEL[n]) for n in names]
    if reference:h.append(Line2D([],[],color='#8E989F',ls='--',lw=.7,label='Reference'))
    fig.legend(handles=h,loc='upper center',bbox_to_anchor=anchor,ncol=len(h),handlelength=2.4,columnspacing=1.6)
def csv(name,rows,header):np.savetxt(FIG/name,rows,delimiter=',',comments='',header=header)
def setup():
    b=read(OLD/'paper_result_binding.json');ds={};rs={};gs={};sources={}
    cache=WORK/'plot_cache';cache.mkdir(exist_ok=True)
    for case in ['original','dense']:
        cr=b['cases'][case];assert sha(cr['reference'])==cr['reference_sha256'] and sha(cr['geometry'])==cr['geometry_sha256']
        rs[case]=np.loadtxt(cr['reference'],delimiter=',',skiprows=1);gs[case]=read(cr['geometry']);ds[case]={}
        for n in ORDER:
            r=b['methods'][n][case][0];assert r['dt_s']==.001 and sha(r['trace'])==r['trace_sha256']
            sources[r['trace']]=r['trace_sha256'];p=cache/f'{case}_{KEY[n]}_{r["trace_sha256"][:12]}.npz'
            if p.exists():
                z=np.load(p);d={k:z[k] for k in z.files}
            else:
                d=prev.load_trace(r['trace'],rs[case]);d={k:d[k] for k in ['rows','t','progress']}
                d['speed']=norms(d)[:,0];np.savez_compressed(p,**d)
            d['binding']=r
            if 'metrics' in r:
                assert sha(r['metrics'])==r['metrics_sha256'];m=read(r['metrics']);d['metrics']=m.get('metrics',m)
            ds[case][n]=d
    return b,rs,gs,ds,sources
def figure4(ref,g,d):
    # Reuse the approved original-case annotation geometry, with leaner labels.
    src=inspect.getsource(common.figure4_case)
    src=src.replace("figsize=(7.25, 8.4)","figsize=(7.25, 7.6)")
    src=src.replace("    ax.annotate('', xy=(4210, 2230), xytext=(4200, 2230), arrowprops=dict(arrowstyle='<->', lw=.7, shrinkA=0, shrinkB=0))\n",'')
    src=src.replace("    ax.text(4205, 2236, '10 mm', fontsize=7, ha='center')\n",'')
    src=src.replace("'Closed reference path: 44 420.514 mm; 34 corners'","'Long closed route: 44.421 m; 34 corners'")
    ns=dict(common.__dict__);exec(compile(src,'approved_case_layout','exec'),ns)
    ns['figure4_case'](ref,g,FIG,{KEY[n]:d[n] for n in ORDER})
def figure5(b,d):
    # Existing independent reconstruction; reorder the legend with J-NNC first.
    util.FIG=FIG;util.COLORS.update(COL)
    source=inspect.getsource(util.figure_nnc)
    source=source.replace("fig.legend(handles,labels,loc=", "fig.legend([handles[i] for i in [1,0,2]],[labels[i] for i in [1,0,2]],loc=")
    ns=dict(util.__dict__);exec(compile(source,'nnc_same_window','exec'),ns)
    return ns['figure_nnc'](b,d,read(OLD/'nnc_evidence_binding.json'))
def corner_stats(d,c,halfwidth=10):
    s=d['progress']-c['reference_s_mm'];y=norms(d);m=abs(s)<=halfwidth
    return dict(min_speed_mm_s=float(y[m,0].min()),deficit_mm=float(.001*np.maximum(100-y[m,0],0).sum()))
def figure6(g,d):
    c=next(c for c in g['sharp_corners'] if c['id']=='corner_02')
    fig=plt.figure(figsize=(7.25,5.8));grid=fig.add_gridspec(3,2,width_ratios=[1.9,1.0],height_ratios=[1.28,1,1],hspace=.58,wspace=.48)
    axes=[fig.add_subplot(grid[i,0]) for i in range(3)];values=[];stats={}
    for n in ORDER:
        dd=d[n];s=dd['progress']-c['reference_s_mm'];m=abs(s)<=20;y=norms(dd)
        stats[n]=corner_stats(dd,c)
        for i,ax in enumerate(axes):line(ax,s[m],y[m,i]/(1 if i==0 else LIMIT[i]),n)
        values.extend([ORDER.index(n),float(x),*map(float,z)] for x,z in zip(s[m],y[m]))
    for i,ax in enumerate(axes):
        ax.axvline(0,color='#BBC4CB',lw=.65);ax.grid(axis='y',lw=.4,color='#E5E9EC')
        ax.axhline(100 if i==0 else 1,color='#404A50',ls=':',lw=.75)
        ax.set(xlim=(-20,20),xticks=[-20,-10,0,10,20],ylim=(0,107) if i==0 else (-.035,1.09))
        ax.set_ylabel(['Speed (mm/s)',r'$\|a\|/A_{\max}$',r'$\|j\|/J_{\max}$'][i])
        common.panel(ax,chr(97+i),['90° / C02: retained feedrate','Acceleration use','Jerk use'][i])
        if i<2:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Reference progress from corner (mm)')
    ax=fig.add_subplot(grid[:,1]);ax.spines[['left','bottom']].set_visible(False)
    yy=[0,1,2];vv=[stats[n]['deficit_mm'] for n in ORDER]
    ax.barh(yy,vv,height=.35,color=[COL[n] for n in ORDER],alpha=.9)
    for k,n in enumerate(ORDER):
        ax.text(0,k-.34,LABEL[n],color=COL[n],fontsize=8.7,weight='bold' if n=='J-NNC' else 'normal')
        ax.text(vv[k]+.25,k,f'{vv[k]:.3f}',va='center',fontsize=8.2,color=COL[n])
        ax.text(0,k+.35,f'Min. speed: {stats[n]["min_speed_mm_s"]:.2f} mm/s',va='center',fontsize=7.7,color=COL[n])
    ax.set(xlim=(0,12.9),ylim=(3.8,-.68),yticks=[],xticks=[0,5,10],xlabel='Speed deficit (mm)')
    ax.grid(axis='x',color='#E5E9EC',lw=.4,zorder=-1);ax.set_axisbelow(True)
    common.panel(ax,'d','Integrated corner loss')
    gain_step=100*(1-vv[0]/vv[2]);gain_two=100*(1-vv[0]/vv[1])
    ax.text(0,2.85,f'J-NNC loss reduction\n{gain_step:.1f}% vs two-step\n{gain_two:.1f}% vs two-action',fontsize=8.7,color=COL['J-NNC'],linespacing=1.65)
    ax.text(0,3.59,'Same ±10 mm window',fontsize=7.7,color='#687781')
    legend(fig,anchor=(.53,1.005));fig.subplots_adjust(top=.905,left=.105,right=.98,bottom=.09)
    common.save(fig,'Figure_6',FIG);csv('Figure_6_source_data.csv',values,'method_JNNC_Two_S2,relative_progress_mm,speed_mm_s,acceleration_mm_s2,jerk_mm_s3')
    write(FIG/'Figure_6_corner_summary.json',dict(corner='C02',stats=stats,loss_reduction_vs_two_step_pct=gain_step,loss_reduction_vs_two_action_pct=gain_two))
    return stats
def figure7(ref,g,d):
    fig,axes=plt.subplots(2,3,figsize=(7.25,5.3),gridspec_kw=dict(hspace=.55,wspace=.3))
    rows=[]
    for col,cid in enumerate(['corner_01','corner_02','corner_06']):
        c=next(c for c in g['sharp_corners'] if c['id']==cid);vertex,rot,u=common.local_geometry(ref,c)
        for row,(xx,yy) in enumerate([((-7,5),(-2,10)),((-4.5,1.5),(-.6,5.4))]):
            ax=axes[row,col];common.corridor(ax,u,xx,yy)
            for n in ORDER:
                xy=common.local_curve(d[n],vertex,rot,c,window=20);line(ax,*xy.T,n)
                if row==0:rows.extend([col,ORDER.index(n),*map(float,z)] for z in xy)
            if row==0:
                ax.add_patch(Rectangle((-4.5,-.6),6,6,fill=False,ec='#9BA7AF',lw=.65,ls=':'))
                ax.set(xticks=[-5,0,5],yticks=[0,5,10]);common.panel(ax,chr(97+col),f'{c["interior_angle_deg"]:.0f}° / C{cid[-2:]}')
            else:
                ax.set(xticks=[-4,-2,0],yticks=[0,2,4]);common.panel(ax,chr(100+col),'Transition detail')
            if col==0:ax.set_ylabel('Inward-normal y (mm)')
            ax.set_xlabel('Incoming-axis x (mm)')
    legend(fig,True,anchor=(.51,1.005));fig.subplots_adjust(top=.88,bottom=.10,left=.09,right=.985)
    common.save(fig,'Figure_7',FIG);csv('Figure_7_xy_data.csv',rows,'corner_C01_C02_C06,method_JNNC_Two_S2,local_x_mm,local_y_mm')
def figure8(g,d):
    fig=plt.figure(figsize=(7.25,6.2));grid=fig.add_gridspec(3,1,height_ratios=[1.45,1.35,.63],hspace=.73)
    ax=fig.add_subplot(grid[0]);x=np.arange(34);rows=[]
    for cid in ['corner_01','corner_02','corner_06']:i=int(cid[-2:]);ax.axvspan(i-.25,i+.25,color='#EFF3F5',zorder=-2)
    for n in ORDER:
        y=np.array([c['speed_deficit_mm'] for c in d[n]['metrics']['corners']]);assert len(y)==34
        line(ax,x,y,n,lw=1 if n=='J-NNC' else .75,marker={'J-NNC':'o','Two-action':'s','S2':'^'}[n],ms=3.2,
            mfc=COL[n] if n=='J-NNC' else 'white',mew=.7)
        rows.extend([ORDER.index(n),int(i),float(z)] for i,z in zip(x,y))
    ax.set(xlim=(-.6,33.6),ylim=(-.35,15.5),xticks=list(range(0,34,2))+[33],ylabel='Speed deficit (mm)',xlabel='Corner in path order (C00–C33)')
    ax.set_xticks(x,minor=True);ax.grid(axis='y',color='#E5E9EC',lw=.4);common.panel(ax,'a','All 34 corners: lower integrated speed loss')
    ax=fig.add_subplot(grid[1]);cruise={};trace=[]
    for n in ORDER:
        dd=d[n];low=dd['speed']<99;yy=np.cumsum(low)*.001;xx,z=common.extrema_envelope(dd['t'],yy,100)
        line(ax,xx,z,n);starts,durs=common.cruise_intervals(dd['speed']);total=float((~low).sum()*.001)
        cruise[n]=dict(cruise_s=total,cruise_percent=100*total/dd['t'][-1],longest_cruise_s=float(durs.max()))
        trace.extend([ORDER.index(n),float(x),float(v)] for x,v in zip(xx,z))
    ax.set(xlim=(0,447),ylim=(0,4.7),xticks=[0,100,200,300,400],xlabel='Complete elapsed time (s)',ylabel='Time below 99 mm/s (s)')
    ax.grid(color='#E5E9EC',lw=.4);common.panel(ax,'b','Cumulative time outside near-limit cruise')
    ax=fig.add_subplot(grid[2]);ax.axis('off')
    cells=[[LABEL[n],f'{cruise[n]["cruise_s"]:.3f}',f'{cruise[n]["cruise_percent"]:.3f}',f'{cruise[n]["longest_cruise_s"]:.3f}'] for n in ORDER]
    tb=ax.table(cellText=cells,colLabels=['Method','≥99 mm/s time (s)','Time share (%)','Longest interval (s)'],cellLoc='center',colWidths=[.26,.26,.22,.26],bbox=[0,0,1,1]);tb.auto_set_font_size(False);tb.set_fontsize(8)
    for (r,c),cell in tb.get_celld().items():
        cell.set_edgecolor('white')
        if r==0:cell.set_facecolor('#EFF3F5')
        elif r==1:cell.set_facecolor('#EFF5F9');cell.get_text().set_color(COL['J-NNC']);cell.get_text().set_weight('bold')
        elif c==0:cell.get_text().set_color(COL[ORDER[r-1]])
    legend(fig,anchor=(.54,1.005));fig.subplots_adjust(top=.90,left=.105,right=.985,bottom=.035)
    common.save(fig,'Figure_8',FIG);csv('Figure_8_all34_data.csv',rows,'method_JNNC_Two_S2,corner_index,deficit_mm');csv('Figure_8_cruise_data.csv',trace,'method_JNNC_Two_S2,time_s,below99_time_s');write(FIG/'Figure_8_cruise_summary.json',cruise)
def figure9(ref,g,d):
    fig=plt.figure(figsize=(7.25,5.8));grid=fig.add_gridspec(2,2,height_ratios=[1,1.13],hspace=.50,wspace=.34)
    ax=fig.add_subplot(grid[0,0]);ax.plot(*ref.T,color='#6F808C',lw=.8)
    ax.set(aspect='equal',xlabel='X (mm)',ylabel='Y (mm)');common.panel(ax,'a','Dense closed case: 84 corners')
    ids=['corner_00','corner_27'];xyrows=[];lossrows=[]
    for cid in ids:
        c=next(c for c in g['sharp_corners'] if c['id']==cid);ax.plot(*c['point_mm'],'o',ms=4,color=COL['J-NNC'])
        ax.annotate('C'+cid[-2:],c['point_mm'],xytext=(6,6),textcoords='offset points',fontsize=7.5)
    ax=fig.add_subplot(grid[0,1])
    for n in ['J-NNC','Two-action']:
        dd=d[n];deficit=np.cumsum(np.maximum(100-norms(dd)[:,0],0))*.001
        line(ax,dd['progress'],deficit,n);ax.text(1460,deficit[-1],f'{deficit[-1]:.1f}',color=COL[n],fontsize=8,va='center')
        lossrows.extend([ORDER.index(n),float(s),float(v)] for s,v in zip(dd['progress'],deficit))
    ax.set(xlim=(0,1660),ylim=(0,230),xticks=[0,400,800,1200,1600],xlabel='Reference progress (mm)',ylabel='Cumulative speed deficit (mm)')
    ax.grid(color='#E5E9EC',lw=.4);common.panel(ax,'b','Loss across all 84 transitions')
    for col,cid in enumerate(ids):
        ax=fig.add_subplot(grid[1,col]);c=next(c for c in g['sharp_corners'] if c['id']==cid);vertex,rot,u=common.local_geometry(ref,c)
        common.corridor(ax,u,(-5,3),(-1.25,6.75))
        for n in ['J-NNC','Two-action']:
            xy=common.local_curve(d[n],vertex,rot,c,window=8);line(ax,*xy.T,n);xyrows.extend([col,ORDER.index(n),*map(float,z)] for z in xy)
        ax.set(xlabel='Incoming-axis x (mm)',ylabel='Inward-normal y (mm)',xticks=[-4,-2,0,2],yticks=[0,2,4,6]);common.panel(ax,chr(99+col),f'{c["interior_angle_deg"]:.0f}° / C{cid[-2:]}: local transition')
    legend(fig,True,anchor=(.53,1.005),names=['J-NNC','Two-action']);fig.subplots_adjust(top=.88,left=.1,right=.985,bottom=.09)
    common.save(fig,'Figure_9',FIG);csv('Figure_9_loss_data.csv',lossrows,'method_JNNC_Two_S2,reference_progress_mm,cumulative_deficit_mm');csv('Figure_9_xy_data.csv',xyrows,'corner_C00_C27,method_JNNC_Two_S2,local_x_mm,local_y_mm')
def full_support(d):
    util.FIG=FIG;util.METHODS[:]=ORDER;util.COLORS.update(COL)
    src=inspect.getsource(util.figure_full).replace("name)\n            if row==2", "LABEL[name])\n            if row==2")
    src=src.replace("common.save(fig,'Figure_6',FIG)","common.save(fig,'Figure_C1',FIG)")
    src=src.replace("'Figure_6_display_data.csv'","'Figure_C1_display_data.csv'").replace("'Figure_6_stop_data.csv'","'Figure_C1_stop_data.csv'")
    src=src.replace("'method_S2_Two_JNNC", "'method_JNNC_Two_S2")
    ns=dict(util.__dict__);ns.update(LABEL=LABEL,line=lambda ax,x,y,n,**kw:line(ax,x,y,n,**kw))
    exec(compile(src,'complete_physical_record','exec'),ns);ns['figure_full'](d)
def main():
    a=argparse.ArgumentParser();a.add_argument('--figures',default='4,5,6,7,8,9,C1');opt=a.parse_args()
    b,r,g,d,sources=setup();metrics={}
    for name in opt.figures.split(','):
        print('DRAW',name,flush=True)
        if name=='4':figure4(r['original'],g['original'],d['original'])
        elif name=='5':metrics['nnc']=figure5(b,d)
        elif name=='6':metrics['C02']=figure6(g['original'],d['original'])
        elif name=='7':figure7(r['original'],g['original'],d['original'])
        elif name=='8':figure8(g['original'],d['original'])
        elif name=='9':figure9(r['dense'],g['dense'],d['dense'])
        elif name=='C1':full_support(d['original'])
    write(OUT/'figure_contract.json',dict(backend='Python/matplotlib',layout_basis='User-approved manuscript and requested comparison logic; no external figure precedent claimed.',
        conclusions={'4':'Natural long-route scale, geometry and complete34angle coverage','5':'Recorded NNC requests violate common planar limits','6':'Prespecified C02 comparison: reduced depth and integrated duration of speed loss under the same V/A/J limits','7':'Three prespecified trajectory windows show advance turning and recovery with honest local magnification','8':'All34corner losses and full-route cruise corroborate local comparison','9':'Dense full-case loss and local paths support developed three/two-action comparison','C1':'Complete vector execution and terminal rest under common bounds'},
        source_integrity='Same bound selected1ms model and baselines; no smoothing, simulation, model selection or rescaling to manufacture separation.',
        nomenclature='Display Two-step planner; legacyS2 keys retained only for binding.',order=ORDER,
        statistics='Recorded selected trajectories, not independent repetitions; no inferred error bars.',export='PDF and editable SVG for4--9,C1; retained-vector PDFs plus editable replacement overlays for1/3.'))
    write(OUT/'figure_data_binding.json',dict(source_binding=str(OLD/'paper_result_binding.json'),source_binding_sha256=sha(OLD/'paper_result_binding.json'),
        source_sha256=sources,script_sha256=sha(__file__),metrics=metrics,outputs={p.name:sha(p) for p in FIG.iterdir()},new_simulations=0,new_training=0))
    print('ACADEMIC_FIGURES_READY',str(FIG))
if __name__=='__main__':main()
