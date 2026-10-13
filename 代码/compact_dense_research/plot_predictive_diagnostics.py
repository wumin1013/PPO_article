"""Internal raw-data diagnostics in the user's prescribed local/6x3 layouts."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import LineString

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'predictive_dynamics_registered_window_figures'
MAIN=ROOT/'historical_dense_predictive_action_diagnostic'
S2=ROOT/'dense_s2_initial'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    geometry=json.loads((ROOT/'dense_closed_case/geometry.json').read_text(encoding='utf-8'))
    selected=[next(c for c in geometry['sharp_corners'] if abs(c['interior_angle_deg']-angle)<1e-5) for angle in [135,90,45]]
    sources=[MAIN/'trace.npz',MAIN/'all84_metrics.json',MAIN/'action_search.json',S2/'trace.npz',S2/'all84_metrics.json',ROOT/'dense_closed_case/geometry.json']
    contract=dict(scope='Internal diagnostic figures, not final manuscript figures or a method-freeze claim.',
        core_conclusion='Relate recorded local path differences to three requested actions and actual heading,curvature,speed; no causal or every-corner superiority claim.',
        selection='First registered corner of each exact included angle135,90,45degrees; chosen without ranking outcome.',
        selected_corners=[c['id'] for c in selected],archetype='quantitative grid',backend='Python matplotlib',
        geometry='User-prescribed three columns and six mechanism rows; separate local equal-aspect trajectory comparison.',
        visual_precedent='No external publication precedent asserted. Internal diagnostic scope and user-prescribed geometry; formal manuscript-precedent review remains deferred until adopted-method figures.',
        sources={str(p):sha(p) for p in sources},statistics='Single deterministic complete trajectory per method; no error bars or independent-repeat claims.',
        image_integrity='Original1ms recorded physical samples; curves clipped to the pre-registered spatial windows before time alignment. No trajectory smoothing,downsampling or fabricated data.',
        exports='Editable SVG and vector PDF plus240dpi PNG; width180mm.')
    (OUT/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(Path(__file__),OUT/'source.py')
    plt.rcParams.update({'font.family':'Arial','font.size':7.5,'axes.labelsize':7.5,
        'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'lines.linewidth':1.2,'legend.frameon':False,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    blue='#275A84';orange='#B36930';gray='#747B82';violet='#9A86AD'
    traces={'Predictive correction':np.load(MAIN/'trace.npz'),'S2':np.load(S2/'trace.npz')}
    stats={label:json.loads((folder/'all84_metrics.json').read_text(encoding='utf-8')) for label,folder in [('Predictive correction',MAIN),('S2',S2)]}
    search=json.loads((MAIN/'action_search.json').read_text(encoding='utf-8'))
    nominal=np.array([x['candidates'][0]['action'] for x in search])
    reference=traces['Predictive correction']['reference']
    fig,axes=plt.subplots(1,3,figsize=(180/25.4,72/25.4),layout='constrained')
    mech,m_axes=plt.subplots(6,3,figsize=(180/25.4,225/25.4),sharex='col',layout='constrained')
    csv_records=[]
    for col,c in enumerate(selected):
        k=c['reference_vertex_index'];vertex=reference[k]
        incoming=vertex-reference[k-1];incoming/=np.linalg.norm(incoming)
        sign=np.sign(c['signed_heading_change_deg']);normal=sign*np.array([-incoming[1],incoming[0]])
        frame=np.stack([incoming,normal],axis=1)
        local_ref=(reference-vertex)@frame
        ax=axes[col]
        corridor=LineString(local_ref).buffer(1.25,resolution=32)
        polygons=list(corridor.geoms) if corridor.geom_type=='MultiPolygon' else [corridor]
        for polygon in polygons:
            px,py=polygon.exterior.xy;ax.fill(px,py,color='#EEF0F2',zorder=0)
            for ring in polygon.interiors:
                px,py=ring.xy;ax.fill(px,py,color='white',zorder=0)
        ax.plot(local_ref[:,0],local_ref[:,1],color=gray,ls='--',lw=.75,zorder=1)
        bounds=[];time_bounds=[]
        for label,color in [('Predictive correction',blue),('S2',orange)]:
            data=traces[label];rows=data['rows']
            metric=next(x for x in stats[label]['corners'] if x['id']==c['id'])
            idx=np.arange(metric['first_index'],metric['last_index']+1)
            xy=(rows[idx,:2]-vertex)@frame
            bounds.append(xy)
            ax.plot(xy[:,0],xy[:,1],color=color,label=label,zorder=3 if label=='Predictive correction' else 2)
            time=(np.arange(len(rows))+1)*.001-metric['crossing_time_s']
            use=np.zeros(len(rows),dtype=bool);use[idx]=True
            rr=rows[use];tt=time[use];speed=np.linalg.norm(rr[:,2:4],axis=1)
            time_bounds.extend([float(tt.min()),float(tt.max())])
            heading=np.unwrap(np.arctan2(rr[:,2:4]@normal,rr[:,2:4]@incoming))*180/np.pi
            curvature=sign*(rr[:,2]*rr[:,5]-rr[:,3]*rr[:,4])/np.maximum(speed**3,1e-12)
            curvature[speed<=5]=np.nan
            for row_idx,y in enumerate([heading,curvature,speed],start=3):
                m_axes[row_idx,col].plot(tt,y,color=color,lw=1.1)
            csv_records.extend([[c['id'],label,float(t),float(h),float(curv),float(v)] for t,h,curv,v in zip(tt,heading,curvature,speed)])
            if label=='Predictive correction':
                decision_times=np.arange(len(data['actions']))*.02-metric['crossing_time_s']
                first=max(0,int(np.searchsorted(decision_times,tt.min(),side='right')-1))
                last=min(len(decision_times),int(np.searchsorted(decision_times,tt.max(),side='right')+1))
                action_ids=np.arange(first,last)
                for row_idx in range(3):
                    multiplier=sign*180/np.pi if row_idx==0 else 1.
                    m_axes[row_idx,col].step(decision_times[action_ids],nominal[action_ids,row_idx]*multiplier,where='post',color=violet,ls='--',lw=.9)
                    m_axes[row_idx,col].step(decision_times[action_ids],data['actions'][action_ids,row_idx]*multiplier,where='post',color=blue,lw=1.15)
                np.savetxt(OUT/f"{c['id']}_requested_actions.csv",np.column_stack([decision_times[action_ids],nominal[action_ids],data['actions'][action_ids]]),delimiter=',',
                    header='relative_time_s,nominal_steering_rad,nominal_feed_mm_s,nominal_preview_mm,final_steering_rad,final_feed_mm_s,final_preview_mm',comments='')
        both=np.vstack(bounds)
        lo=np.array([-11.,-2.]);hi=np.array([8.,9.])
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='Incoming coordinate (mm)',
            title=f"{chr(97+col)}  {round(c['interior_angle_deg'])}° · {c['id'].replace('corner_','C')}")
        ax.grid(color='#E4E7E9',lw=.45)
        if col==0:ax.set_ylabel('Inward coordinate (mm)')
        m_axes[0,col].set_title(f"{round(c['interior_angle_deg'])}° · {c['id'].replace('corner_','C')}")
        for row_idx in range(6):
            a=m_axes[row_idx,col];a.axvline(0,color='#B9BEC4',ls=':',lw=.7)
            a.set_xlim(min(time_bounds)-.015,max(time_bounds)+.015)
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            a.grid(axis='y',color='#ECEEF0',lw=.45)
        m_axes[5,col].set_xlabel('Time from corner gate (s)')
        m_axes[5,col].set_ylim(0,105)
        m_axes[5,col].axhline(99,color=gray,ls=':',lw=.6)
    labels=['Steering (°)','Feed (mm/s)','Preview (mm)','Actual heading (°)','Curvature (1/mm)','Speed (mm/s)']
    for row,label in enumerate(labels):m_axes[row,0].set_ylabel(label)
    handles=[Line2D([],[],color=blue,label='Predictive correction'),Line2D([],[],color=orange,label='S2'),Line2D([],[],color=gray,ls='--',label='Reference')]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,fontsize=7.5)
    mech.legend(handles=[handles[0],handles[1],Line2D([],[],color=violet,ls='--',label='Historical request')],loc='outside upper center',ncol=3,fontsize=7.5)
    for figure,name in [(fig,'local_trajectory_diagnostic'),(mech,'six_row_mechanism_diagnostic')]:
        for suffix in ['pdf','svg','png']:
            figure.savefig(OUT/f'{name}.{suffix}',dpi=240)
        plt.close(figure)
    import csv
    with (OUT/'physical_series.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['corner','method','relative_time_s','heading_deg','curvature_per_mm','speed_mm_s']);writer.writerows(csv_records)
    (OUT/'caption.md').write_text('Internal diagnostics of complete dense-case trajectories. Columns use the first registered135°,90° and45° corners; they are not selected by performance. Local paths share coordinate limits and equal scale, with the original1.25mm corridor. Mechanism traces are restricted to the registered spatial window, then aligned by each method\'s witnessed ordered-gate crossing. Therefore the displayed durations differ by method, and adjacent turns beyond the fixed window are excluded. Requested actions include the immediately preceding held decision. Top three rows show historical requests and final requests after the finite prediction search; bottom rows show measured velocity heading, signed curvature and speed. Curves are unsmoothed; S2 has no corresponding policy-action traces. One deterministic run per method; these figures do not establish independent repeatability or final learned-policy superiority.\n',encoding='utf-8')
    print(OUT)


if __name__=='__main__':main()
