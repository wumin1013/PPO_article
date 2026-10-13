"""Read-only diagnostics of saved learned trajectories, without XY smoothing."""
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from isolated_corner_shape import corner_shape
from complete_corner_metrics import complete_corner_metrics
from audit_xy_curvature_resolution import curvature_metrics
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--arm',default='fitted')
    parser.add_argument('--source',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--step',type=float)
    parser.add_argument('--paths',type=int,nargs='+',default=[0,4,8,15])
    args=parser.parse_args()
    parent=BASE/args.parent;source=BASE/args.source
    selected=next(r for r in read(parent/'results.json')['candidates']
                  if r.get('arm',r.get('tag'))==args.arm or r.get('model')==args.arm)
    data=read(source/'results.json')
    positive=(next(r for r in data['candidates'] if r['step']==args.step)
              if args.step is not None else data.get('chosen') or
              next(r for r in data['candidates'] if r['step']>0))
    stage=register(args.stage,[__file__,parent/'results.json',source/'results.json',
        source/'receipt.json',BASE/'isolated_corner_shape.py',BASE/'complete_corner_metrics.py'],
        dict(new_actor_calls=0,new_physics_ticks=0,wall_s=60,paths=args.paths),
        'Inspect learned parent and learned positive update at every saved1ms point. '
        'Panels link actual XY, signed physical heading, signed curvature, and speed. '
        'Show both0.05deg and0.5deg onset diagnostics with raw extrema, keeping the '
        'existing full shape metrics. This is exploratory diagnosis, not a manuscript '
        'figure or independent generalization result. No smoothed XY or ideal template.')
    start=time.perf_counter()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(len(args.paths),4,figsize=(15,2.9*len(args.paths)),
        squeeze=False,gridspec_kw={'width_ratios':[1.3,1,1,1]})
    records=[]
    for panel,path in enumerate(args.paths):
        ax=axes[panel]
        for label,color,candidate in [('Parent','#707070',selected),
                                     ('Learned positive update','#2878b5',positive)]:
            j=next(i for i,r in enumerate(candidate['results']) if r['original_path']==path)
            trace=BASE/candidate['traces'][j]
            with np.load(trace) as z:
                ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            assert len(ref)==3
            u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);sign=np.sign(angle)
            inward=sign*np.array([-u[1],u[0]])
            t=(np.arange(len(rows))+1)*.001
            speed=np.linalg.norm(rows[:,2:4],axis=1);moving=speed>5.
            heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@inward,rows[:,2:4]@u)))
            kappa=np.full(len(rows),np.nan)
            kappa[moving]=sign*(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
            apex=int(np.argmin(np.linalg.norm(rows[:,:2]-ref[1],axis=1)))
            before=(np.arange(len(rows))<=apex)&moving
            onset=[]
            for threshold in (.05,.5):
                ids=np.flatnonzero(before&(np.abs(heading)>threshold))
                onset.append(dict(threshold_deg=threshold,minimum_speed_mm_s=5.,
                    first_time_s=float(t[ids[0]]) if len(ids) else None,
                    first_direction=('inside' if heading[ids[0]]>0 else 'outside') if len(ids) else None,
                    reverse_heading_duration_s=float((before&(heading<-threshold)).sum()*.001)))
            record=dict(path=path,label=label,trace=str(trace.relative_to(BASE)),
                success=candidate['results'][j]['success'],onset=onset,
                corner=corner_shape(ref,rows,actions),complete=complete_corner_metrics(ref,rows),
                curvature=curvature_metrics(ref,rows))
            records.append(record)
            near=np.linalg.norm(rows[:,:2]-ref[1],axis=1)<8.
            ax[0].plot(rows[near,0],rows[near,1],color=color,lw=1.4,label=label)
            ax[1].plot(t,np.where(moving,heading,np.nan),color=color,lw=1.2)
            ax[2].plot(t,kappa,color=color,lw=1.1)
            ax[3].plot(t,speed,color=color,lw=1.1)
        line=np.stack([ref[1]-8*u,ref[1],ref[1]+8*v])
        ax[0].plot(line[:,0],line[:,1],'--',color='#b0b0b0',lw=.8,zorder=0)
        ax[0].scatter(*ref[1],marker='x',s=18,color='#999999')
        ax[0].set_aspect('equal',adjustable='datalim')
        ax[0].set(xlabel='X (mm)',ylabel='Y (mm)',title=f'Path {path}: {np.rad2deg(angle):g} deg')
        for boundary in (0.,abs(np.rad2deg(angle))):
            ax[1].axhline(boundary,ls=':',lw=.8,color='#aaaaaa')
        ax[1].set(xlabel='Time (s)',ylabel='Signed heading (deg)')
        ax[2].axhline(0.,ls=':',lw=.8,color='#aaaaaa')
        ax[2].set(xlabel='Time (s)',ylabel='Signed curvature (1/mm)')
        ax[3].axhline(100.,ls=':',lw=.8,color='#aaaaaa')
        ax[3].set(xlabel='Time (s)',ylabel='Speed (mm/s)',ylim=(0,105))
        for item in ax:item.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2)
    fig.suptitle('Actual learned trajectories: shape, direction, turning and cruise',fontsize=12)
    fig.tight_layout(rect=(0,.035,1,.97))
    fig.savefig(stage/'actual_shape_comparison.png',dpi=180)
    plt.close(fig)
    write_json(stage/'results.json',dict(records=records,positions_smoothed=False,
        heading_and_curvature_speed_threshold_mm_s=5.,goal_complete=False))
    receipt(stage,dict(status='saved_actual_shape_composition_diagnostics',
        new_physics_ticks=0,wall_s=time.perf_counter()-start,visual_inspection_pending=True))


if __name__=='__main__':main()
