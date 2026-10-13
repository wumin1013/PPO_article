"""Publication figures from a passed, explicitly bound complete-task comparison.

Only reads recorded trajectories. Protected Figures 1--3 are copied unchanged.
Shared geometry helpers retain the approved original-revision plotting style.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import all34_figure_helpers as common
from all34_figure_helpers import (read, sha, write_json, save, panel, legend_handles,
    local_geometry, local_curve, corridor, extrema_envelope, COLORS, STYLES, NAMES, ORDER)


def load_data(binding):
    ref = np.loadtxt(binding['reference'], delimiter=',', skiprows=1)
    assert sha(Path(binding['reference'])) == binding['reference_sha256']
    data = {}
    for name, source in binding['primary_methods'].items():
        for kind in ['trace', 'result', 'metrics']:
            assert sha(Path(source[kind])) == source[kind+'_sha256']
        data[name] = load_trace(source['trace'], ref)
        data[name].update(result=read(source['result']), metrics=read(source['metrics']))
    for name, source in binding['interventions'].items():
        for kind in ['trace', 'result']:
            assert sha(Path(source[kind])) == source[kind+'_sha256']
        data[name] = load_trace(source['trace'], ref)
        data[name].update(result=read(source['result']), failed_corner=source.get('failed_corner'))
    return ref, data


def load_trace(path, ref):
    z = np.load(path)
    rows = z['rows']
    # Independent XY differences supply the displayed vector dynamics.
    v = np.diff(np.vstack([ref[0], rows[:,:2]]), axis=0)/.001
    a = np.diff(np.vstack([np.zeros(2), v]), axis=0)/.001
    j = np.diff(np.vstack([np.zeros(2), a]), axis=0)/.001
    physical = rows.copy()
    physical[:,2:8] = np.hstack([v,a,j])
    return dict(rows=physical, t=(np.arange(len(rows))+1)*.001,
        speed=np.linalg.norm(rows[:,2:4],axis=1),
        actions=z['actions'] if 'actions' in z else None,
        decision_t=z['decision_times_s'] if 'decision_times_s' in z else
            (np.arange(len(z['actions']))*.02 if 'actions' in z else None),
        progress=common.ordered_projection(ref,rows[:,:2]), source=str(path))


def figure5(ref,data,geometry,out):
    fig,axes=plt.subplots(2,3,figsize=(7.25,5.65),gridspec_kw=dict(hspace=.4,wspace=.24))
    for col,cid in enumerate(geometry['mechanism_columns']):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid)
        vertex,rot,u=local_geometry(ref,c)
        for row,(xx,yy) in enumerate([((-10,10),(-2,12)),((-4,2),(-1,5))]):
            ax=axes[row,col]
            corridor(ax,u,xx,yy)
            for name in ORDER:
                xy=local_curve(data[name],vertex,rot,c)
                ax.plot(*xy.T,color=COLORS[name],ls=STYLES[name],lw=1.35,zorder=4 if name=='Joint' else 3)
            if row==0:
                ax.add_patch(Rectangle((-4,-1),6,6,fill=False,ec='#505F69',lw=.7,ls=':'))
                ax.set(xticks=[-10,0,10],yticks=[0,5,10])
                panel(ax,chr(97+col),f'{c["interior_angle_deg"]:.0f}° / C{cid[-2:]}')
            else:
                ax.set(xticks=[-4,-2,0,2],yticks=[0,2,4])
                panel(ax,chr(100+col),'Transition detail')
            if col==0:ax.set_ylabel('Inward-normal y (mm)')
            ax.set_xlabel('Incoming-axis x (mm)')
    fig.legend(handles=legend_handles(ORDER)+[Line2D([],[],color='#7C858C',ls='--',lw=.8,label='Reference')],
        loc='upper center',bbox_to_anchor=(.53,1.01),ncol=4)
    fig.text(.5,.018,'Shading: 1.25 mm Euclidean corridor. Equal aspect and common scales within each row.',ha='center',fontsize=7.5)
    fig.subplots_adjust(top=.9,bottom=.12,left=.095,right=.985)
    save(fig,'Figure_5',out)


def figure6(ref,data,geometry,out):
    fig,axes=plt.subplots(6,3,figsize=(7.25,8.1),sharex=True)
    values=[[] for _ in range(6)]
    by_column={(row,col):[] for row in [3,4] for col in range(3)}
    for col,cid in enumerate(geometry['mechanism_columns']):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid)
        _,rot,_=local_geometry(ref,c)
        turn_sign=np.sign(c['signed_heading_change_deg'])
        axes[0,col].set_title(f'({chr(97+col)})  {c["interior_angle_deg"]:.0f}° / C{cid[-2:]}',pad=9)
        for name in ORDER:
            d=data[name]
            cm=next(m for m in d['metrics']['corners'] if m['id']==cid)
            tc=cm['crossing_time_s']
            keep=abs(d['t']-tc)<=.35
            v=d['rows'][keep,2:4];a=d['rows'][keep,4:6];speed=np.linalg.norm(v,axis=1)
            heading=np.rad2deg(np.unwrap(np.arctan2(v@rot[:,1],v@rot[:,0])))
            curvature=turn_sign*(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(speed,1e-9)**3
            for row,y in [(3,heading),(4,curvature),(5,speed)]:
                axes[row,col].plot(d['t'][keep]-tc,y,color=COLORS[name],ls=STYLES[name],lw=1.05)
                values[row].append(y)
                if row in [3,4]:by_column[row,col].append(y)
            if name!='S2':
                aa=d['actions'];ta=d['decision_t']-tc;keep=abs(ta)<=.39
                for row,y in [(0,turn_sign*np.rad2deg(aa[:,0])),(1,aa[:,1]),(2,aa[:,2])]:
                    axes[row,col].step(ta[keep],y[keep],where='post',color=COLORS[name],ls=STYLES[name],lw=1.05)
                    values[row].append(y[keep])
        axes[3,col].axhline(abs(c['signed_heading_change_deg']),color='#7F8C95',lw=.65,ls=':')
        axes[5,col].axhline(99,color='#7F8C95',lw=.65,ls=':')
    steer=max(15.,float(np.max(np.abs(np.concatenate(values[0]))))*1.12)
    curvature=np.concatenate(values[4])
    limits=[(-steer,steer),(0,105),(0,12.5),(-10,150),
        (min(-.025,float(curvature.min())*1.12),float(curvature.max())*1.12),(0,105)]
    labels=['Steering (°)','Requested feed\n(mm/s)','Preview (mm)','Actual heading\n(°)','Curvature\n(mm⁻¹)','Actual speed\n(mm/s)']
    displayed_limits={}
    for row,label in enumerate(labels):
        axes[row,0].set_ylabel(label)
        for col in range(3):
            ylim=limits[row]
            if row in [3,4]:
                yy=np.concatenate(by_column[row,col])
                ylim=(min(-5.,float(yy.min())*1.12),float(yy.max())*1.1) if row==3 else (min(-.015,float(yy.min())*1.12),float(yy.max())*1.12)
            displayed_limits[f'row{row+1}_column{col+1}']=list(ylim)
            ax=axes[row,col];ax.set(xlim=(-.35,.35),ylim=ylim,xticks=[-.3,0,.3])
            ax.axvline(0,color='#B0BAC1',lw=.55);ax.grid(axis='y',color='#E7EBEE',lw=.5)
            if col and row not in [3,4]:ax.tick_params(labelleft=False)
    for ax in axes[-1]:ax.set_xlabel('Time from gate crossing (s)')
    fig.legend(handles=legend_handles(ORDER),loc='upper center',bbox_to_anchor=(.55,1),ncol=3)
    fig.subplots_adjust(left=.13,right=.985,top=.938,bottom=.075,hspace=.23,wspace=.17)
    save(fig,'Figure_6',out)
    write_json(out/'Figure_6_sign_convention.json',dict(steering_heading_curvature='Inward-positive at each prescribed corner',
        action_hold_s=.020,physical_sample_s=.001,no_signal_smoothing=True,columns=geometry['mechanism_columns'],
        displayed_y_limits=displayed_limits,scale_note='Heading and curvature use labelled column-specific ranges; other rows use common ranges. All methods within each panel share an axis.'))


def figure7(data,geometry,out):
    fig=plt.figure(figsize=(7.25,6.15))
    gs=fig.add_gridspec(3,1,height_ratios=[2.05,1.8,.82],hspace=.7)
    ax=fig.add_subplot(gs[0]);x=np.arange(34)
    for cid in geometry['mechanism_columns']:
        index=int(cid[-2:]);ax.axvspan(index-.25,index+.25,color='#EDF1F4',zorder=-3)
    for name in ORDER:
        yy=np.array([c['speed_deficit_mm'] for c in data[name]['metrics']['corners']])
        ax.plot(x,yy,color=COLORS[name],ls=STYLES[name],lw=.85,
            marker=dict(Joint='o',S2='^',Fixed='s')[name],ms=3.5,mew=.7,
            mfc=COLORS[name] if name=='Joint' else 'white',label=NAMES[name])
    maximum=max(c['speed_deficit_mm'] for n in ORDER for c in data[n]['metrics']['corners'])
    ax.set(xlim=(-.65,33.65),ylim=(-.4,maximum*1.15),xlabel='Corner ID in path order (C00–C33)',
        ylabel='Speed deficit (mm)',xticks=list(range(0,34,2))+[33])
    ax.set_xticks(x,minor=True);ax.grid(axis='y',color='#E2E7EB',lw=.5)
    ax.legend(loc='upper center',ncol=3)
    panel(ax,'a','All 34 registered ±10 mm corner windows')
    ax=fig.add_subplot(gs[1]);statistics={}
    for name in ORDER:
        d=data[name];low=d['speed']<99.;cumulative=np.cumsum(low)*.001
        xx,yy=extrema_envelope(d['t'],cumulative,width=100)
        ax.plot(xx,yy,color=COLORS[name],ls=STYLES[name],lw=1.25)
        starts,length=common.cruise_intervals(d['speed'])
        total=float((~low).sum()*.001);longest=float(length.max())
        statistics[name]=dict(cruise_s=total,cruise_percent=100*total/d['t'][-1],
            longest_cruise_s=longest,below_threshold_s=float(cumulative[-1]))
    ax.set(xlim=(0,447),ylim=(0,max(v['below_threshold_s'] for v in statistics.values())*1.1),
        xlabel='Complete elapsed time (s)',ylabel='Cumulative time\nbelow 99 mm/s (s)',xticks=[0,100,200,300,400])
    ax.grid(color='#E2E7EB',lw=.5)
    panel(ax,'b','Accumulated time outside near-limit cruise')
    ax=fig.add_subplot(gs[2]);ax.axis('off')
    cells=[[NAMES[n],f'{statistics[n]["cruise_s"]:.3f}',f'{statistics[n]["cruise_percent"]:.3f}',
        f'{statistics[n]["longest_cruise_s"]:.3f}'] for n in ORDER]
    table=ax.table(cellText=cells,colLabels=['Method','≥99 mm/s time (s)','Complete-time share (%)','Longest interval (s)'],
        cellLoc='center',colLoc='center',colWidths=[.17,.27,.3,.26],bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(8)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('white');cell.set_linewidth(.4)
        if row==0:cell.set_facecolor('#EDF1F4')
        elif col==0:cell.get_text().set_color(COLORS[ORDER[row-1]])
    fig.subplots_adjust(left=.12,right=.98,top=.935,bottom=.035)
    save(fig,'Figure_7',out)
    write_json(out/'Figure_7_cruise_statistics.json',dict(threshold_mm_s=99,dt_s=.001,methods=statistics,
        trace_reduction='Cumulative monotone curve, endpoints and block extrema retained; no fill bands.'))


def figure8(ref,data,geometry,binding,out):
    fig=plt.figure(figsize=(7.25,6.5))
    gs=fig.add_gridspec(2,2,height_ratios=[1.2,1.75],hspace=.58,wspace=.38)
    ax=fig.add_subplot(gs[0,:]);ax.axis('off')
    rows=[]
    for name in ['Joint','Fixed','Zero','Max']:
        d=data[name];r=d['result']
        if r['success']:
            rows.append([NAMES[name],'Complete',f'{r["complete_time_s"]:.4f} s','—'])
        else:
            rows.append([NAMES[name],'Corridor violation','—',f'{r["elapsed_prefix_or_complete_s"]:.3f} s / C{d["failed_corner"][-2:]}'])
    table=ax.table(cellText=rows,colLabels=['Configuration','Outcome','Complete time','Failed prefix / corner'],
        cellLoc='center',colLoc='center',colWidths=[.22,.27,.2,.31],bbox=[0,.01,1,.93])
    table.auto_set_font_size(False);table.set_fontsize(8)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row==0:cell.set_facecolor('#EDF1F4')
        elif col==0:cell.get_text().set_color(COLORS[['Joint','Fixed','Zero','Max'][row-1]])
    panel(ax,'a','Optimized fixed preview and two output interventions')
    for col,name in enumerate(['Zero','Max']):
        cid=data[name]['failed_corner']
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid)
        vertex,rot,u=local_geometry(ref,c);ax=fig.add_subplot(gs[1,col])
        corridor(ax,u,(-5,3),(-1.5,6.5))
        for method in ['Fixed',name,'Joint']:
            xy=local_curve(data[method],vertex,rot,c)
            ax.plot(*xy.T,color=COLORS[method],ls=STYLES[method],lw=1.35,label=NAMES[method])
        last=(data[name]['rows'][-1,:2]-vertex)@rot
        ax.plot(*last,'x',ms=7,mew=1.2,color=COLORS[name])
        ax.annotate('First corridor violation',xy=last,xytext=(-4.75,5.7),fontsize=7.5,color=COLORS[name],
            arrowprops=dict(arrowstyle='->',lw=.65,color=COLORS[name]),bbox=dict(fc='white',ec='none',pad=1.5))
        ax.set(xlabel='Incoming-axis x (mm)',ylabel='Inward-normal y (mm)',xticks=[-4,-2,0,2],yticks=[0,2,4,6])
        ax.legend(loc='lower right',fontsize=7.3)
        panel(ax,chr(98+col),f'{"Steering" if name=="Zero" else "Feedrate"} intervention: C{cid[-2:]} / {c["interior_angle_deg"]:.1f}°')
    fig.text(.5,.017,'Fixed is separately optimized. Output interventions are not retrained ablations; failed prefixes have no complete time.',ha='center',fontsize=7.3)
    fig.subplots_adjust(left=.095,right=.98,top=.94,bottom=.11)
    save(fig,'Figure_8',out)


def figure9(data,binding,out):
    fig=plt.figure(figsize=(7.25,6.45))
    gs=fig.add_gridspec(3,1,height_ratios=[1.1,1.1,1.2],hspace=.8)
    pair=read(binding['paired_assessment']);decomp=read(binding['efficiency_decomposition'])
    ax=fig.add_subplot(gs[0]);gains={}
    for i,name in enumerate(['S2','Fixed']):
        r=pair['time_comparisons'][name];values=[r['nominal_gain_s']*1000,r['half_ms_gain_s']*1000]
        gains[name]=values
        for k,(value,marker) in enumerate(zip(values,['o','s'])):
            y=i+(-.13 if k==0 else .13)
            ax.plot(value,y,marker,color=COLORS[name],mfc=COLORS[name] if k==0 else 'white',ms=6,mew=1)
            ax.annotate(f'{value:.1f}',xy=(value,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
    margins={n:pair['time_comparisons'][n]['required_gain_s']*1000 for n in ['S2','Fixed']}
    for i,name in enumerate(['S2','Fixed']):
        ax.vlines(margins[name],i-.4,i+.4,color='#77848D',ls=':',lw=.8)
    ax.set(xlim=(0,max(max(v) for v in gains.values())*1.14),ylim=(1.55,-.5),yticks=[0,1],yticklabels=['vs S2','vs Fixed'],xlabel='J-NNC complete-time advantage (ms)')
    ax.grid(axis='x',color='#E3E8EC',lw=.5)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#495761',label='1 ms'),
        Line2D([],[],marker='s',mfc='white',ls='',color='#495761',label='0.5 ms'),
        Line2D([],[],color='#77848D',ls=':',label='Required time margin')],loc='upper center',bbox_to_anchor=(.55,1.15),ncol=3,fontsize=7.5)
    panel(ax,'a','Complete time at two execution resolutions')
    ax=fig.add_subplot(gs[1]);components=[]
    for i,name in enumerate(['S2','Fixed']):
        r=decomp['comparisons'][name]
        for k,(key,color,marker) in enumerate([('path_length_contribution_s','#8B929A','s'),('speed_deficit_contribution_s','#17689A','o')]):
            value=1000*r[key];y=i+(-.14 if k==0 else .14)
            ax.hlines(y,0,value,color=color,lw=1);ax.plot(value,y,marker,color=color,ms=5)
            ax.annotate(f'{value:+.2f}',xy=(value,y),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
            components.append(value)
    ax.axvline(0,color='#C4CDD3',lw=.6)
    ax.set(xlim=(min(-60,min(components)*1.6),max(components)*1.15),ylim=(1.5,-.5),yticks=[0,1],yticklabels=['vs S2','vs Fixed'],xlabel='Contribution to complete-time advantage (ms)')
    ax.legend(handles=[Line2D([],[],marker='s',ls='',color='#8B929A',label='Executed distance'),
        Line2D([],[],marker='o',ls='',color='#17689A',label='Integrated speed deficit')],loc='upper center',bbox_to_anchor=(.54,1.15),ncol=2,fontsize=7.5)
    panel(ax,'b','Measured 1 ms efficiency decomposition')
    ax=fig.add_subplot(gs[2]);search=Path(binding['final_joint_search'])
    if binding.get('retained_complete_validation_index'):
        complete=read(binding['retained_complete_validation_index'])['records']
        complete_values=[]
        for i,row in enumerate(complete):
            values=[row['nominal_complete_time_s'],row['half_ms_complete_time_s']]
            complete_values.extend(values)
            for k,(v,marker) in enumerate(zip(values,['o','s'])):
                ax.plot(i+(-.10 if k==0 else .10),v,marker,color=COLORS['Joint'],
                    mfc=COLORS['Joint'] if k==0 else 'white',ms=5,mew=1.)
                ax.annotate(f'{v:.4f}',(i+(-.10 if k==0 else .10),v),xytext=(0,-13 if k==0 else 8),
                    textcoords='offset points',ha='center',fontsize=7.4)
        lo,hi=min(complete_values),max(complete_values);padding=max(.012,(hi-lo)*.45)
        ax.set(xlim=(-.4,len(complete)-.6),ylim=(lo-padding,hi+padding),xticks=range(len(complete)),
            xticklabels=[str(r['conditional_search_seed']) for r in complete],
            xlabel='Conditional search seed; shared inherited model',ylabel='Complete time (s)')
        ax.ticklabel_format(axis='y',useOffset=False,style='plain')
        ax.grid(axis='y',color='#E3E8EC',lw=.5)
        panel(ax,'c','Complete validation of retained parameter candidates')
        fig.subplots_adjust(left=.14,right=.965,top=.94,bottom=.085)
        save(fig,'Figure_9',out)
        write_json(out/'Figure_9_search_scope.json',dict(complete_candidates=complete,
            global_time_gains_ms=gains,required_time_margins_ms=margins,independent_complete_pipeline_ppo_repeats=0,
            independent_model_error_bars=False,scope='Each point is an actual complete run of a retained parameter candidate; inherited PPO model and search contexts are shared.'))
        return
    results=read(search/'results.json');retained=read(search/'retained.json')
    base=next(r for r in results if r['label']=='identity')['transit_time_s']
    outcomes=[]
    for i,r in enumerate(retained):
        result=next(x for x in results if x['label']==r['selected_label'])
        gain=(base-result['transit_time_s'])*1000
        ax.plot(i,gain,'o',color=COLORS['Joint'],ms=5)
        ax.annotate(f'{gain:.2f} ms',(i,gain),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        outcomes.append(dict(seed=r['seed'],selected_label=r['selected_label'],local_gain_ms=gain,
            comparison_context_count=len(result['records']),positive_gap_sum_mm=result['positive_gap_sum_mm']))
    ymax=max(r['local_gain_ms'] for r in outcomes)
    ax.set(xlim=(-.4,2.4),ylim=(min(0,min(r['local_gain_ms'] for r in outcomes)*1.2),max(5,ymax*1.4)),
        xticks=range(len(retained)),xticklabels=[str(r['seed']) for r in retained],
        xlabel='Conditional search seed; same inherited model and saved contexts',ylabel='Local transit-time\nimprovement (ms)')
    ax.grid(axis='y',color='#E3E8EC',lw=.5)
    panel(ax,'c','Conditional refinement outcomes; independent PPO repeats = 0')
    fig.subplots_adjust(left=.14,right=.965,top=.94,bottom=.085)
    save(fig,'Figure_9',out)
    write_json(out/'Figure_9_search_scope.json',dict(local_outcomes=outcomes,
        global_time_gains_ms=gains,required_time_margins_ms=margins,independent_complete_pipeline_ppo_repeats=0,
        independent_model_error_bars=False,scope='Panel c uses local-context transit times, not complete-path replication.'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('binding');ap.add_argument('output')
    ap.add_argument('--only',type=int,nargs='*',default=[4,5,6,7,8,9]);args=ap.parse_args()
    binding=read(args.binding)
    assert binding['research_acceptance']['passed'] and binding['method_frozen']
    assert read(binding['paired_assessment'])['quantitative_core_passed']
    for path,digest in binding['supporting_evidence_sha256'].items():assert sha(Path(path))==digest
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    ref,data=load_data(binding);geometry=read(binding['geometry'])
    assert sha(Path(binding['geometry']))==binding['geometry_sha256']
    for i in range(1,4):shutil.copy2(common.ORIGINAL/f'Figure_{i}.pdf',out/f'Figure_{i}.pdf')
    for i in args.only:
        if i==4:common.figure4_case(ref,geometry,out,data);common.figure4_execution(data,out)
        elif i==5:figure5(ref,data,geometry,out)
        elif i==6:figure6(ref,data,geometry,out)
        elif i==7:figure7(data,geometry,out)
        elif i==8:figure8(ref,data,geometry,binding,out)
        elif i==9:figure9(data,binding,out)
        print('PUBLICATION_FIGURE',i,flush=True)
    write_json(out/'figure_contract.json',dict(binding=str(Path(args.binding)),binding_sha256=sha(Path(args.binding)),
        original_figures_1_3_unchanged=True,main_figure_count=9,figure4_continued=True,
        all_34_corners=True,prescribed_columns=geometry['mechanism_columns'],
        no_signal_smoothing=True,vector_motion_reconstructed_from_raw_xy=True,
        local_trajectory_windows=True,grid_spacing_mm=10,global_grid_rasterized=False,
        figure7='All-corner deficits, cumulative time below99mm/s, quantitative cruise summary; no dense color band.',
        repeats='No independent PPO repetitions or invented error bars; conditional local searches explicitly labelled.',
        source_sha256={str(Path(__file__)):sha(Path(__file__)),str(Path(common.__file__)):sha(Path(common.__file__))},
        output_sha256={p.name:sha(p) for p in out.glob('Figure_*.*')},visual_qa_complete=False))


if __name__=='__main__':main()
