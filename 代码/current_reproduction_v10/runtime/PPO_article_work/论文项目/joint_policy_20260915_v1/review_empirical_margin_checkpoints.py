"""Actual XY review of completed learned checkpoints that failed promotion."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from complete_corner_metrics import complete_corner_metrics
from audit_xy_curvature_resolution import curvature_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    learning=BASE/'empirical_margin_policy_learning_20260917'
    selection=read(parent/'training_selection.json')
    stage=register('empirical_margin_checkpoint_xy_review_20260917',
        [__file__,parent/'training_selection.json',learning/'receipt.json',learning/'screen_results.json',
         BASE/'complete_corner_metrics.py',BASE/'audit_xy_curvature_resolution.py'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=60),
        'Inspect actual learned closed-loop trajectories after the bounded .5/1 updates. '
        'Both are29/29 but neither satisfies the registered promotion checks. Focus '
        'on40deg left (outgoing excursion worsens) and140deg left (remaining inward '
        'then reverse motion and slight full-step energy increase). Show raw XY, both '
        'edge distances, actual heading, signed physical curvature and actual speed; '
        'record onset at several thresholds. No intervention tapes or ideal templates.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    records=[]
    for path in (3,13):
        fig,axes=plt.subplots(2,3,figsize=(13.2,7),layout='constrained')
        for label,trace,color in [('Parent',BASE/selection['traces'][path],'#70777D'),
            ('Learned half-step',learning/'screen_0.5'/f'{path}.npz','#C9802E'),
            ('Learned full-step',learning/'screen_1'/f'{path}.npz','#286A9F')]:
            with np.load(trace) as z:ref,rows=z['reference'],z['rows']
            u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
            angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);sign=np.sign(angle)
            ni=sign*np.array([-u[1],u[0]]);no=sign*np.array([-v[1],v[0]])
            xy=rows[:,:2]-ref[1];t=np.arange(1,len(rows)+1)*.001
            speed=np.linalg.norm(rows[:,2:4],axis=1);moving=speed>5
            heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@ni,rows[:,2:4]@u)))
            curvature=sign*(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
            axes[0,0].plot(xy[:,0],xy[:,1],label=label,color=color,lw=1.2)
            axes[0,1].plot(t,xy@ni,color=color,lw=1.2)
            axes[0,2].plot(t,xy@no,color=color,lw=1.2)
            axes[1,0].plot(t[moving],heading[moving],color=color,lw=1.2)
            axes[1,1].plot(t[moving],curvature,color=color,lw=1.2)
            axes[1,2].plot(t,speed,color=color,lw=1.2)
            onset={}
            for threshold in (.05,.1,.2,.5,1.):
                ids=np.flatnonzero(moving&(abs(heading)>=threshold))
                onset[str(threshold)]=None if not len(ids) else dict(time_s=float(t[ids[0]]),
                    heading_deg=float(heading[ids[0]]),incoming_side_mm=float((xy@ni)[ids[0]]))
            records.append(dict(path=path,label=label,trace=str(trace.relative_to(BASE)),onset=onset,
                **complete_corner_metrics(ref,rows),**curvature_metrics(ref,rows)))
        reference=ref-ref[1]
        axes[0,0].plot(reference[:,0],reference[:,1],'--',color='#A0A0A0',lw=.8)
        axes[0,0].plot(0,0,'o',color='#333333',ms=3)
        axes[0,0].set_aspect('equal',adjustable='box')
        axes[0,0].set(xlim=(-12,10 if path==3 else 2),ylim=(-.5,9),
            xlabel='x from vertex (mm)',ylabel='y from vertex (mm)',title='a  Raw XY')
        axes[0,1].set(xlabel='Time (s)',ylabel='Incoming signed distance (mm)',
            ylim=(-.015,.05),title='b  Entry zoom: outside < 0')
        axes[0,2].set(xlabel='Time (s)',ylabel='Outgoing signed distance (mm)',
            ylim=(-.2,.08),title='c  Exit zoom: outside < 0')
        axes[1,0].set(xlabel='Time (s)',ylabel='Actual heading (deg)',title='d  Moving-path heading')
        axes[1,1].set(xlabel='Time (s)',ylabel='Signed curvature (1/mm)',title='e  Physical curvature')
        axes[1,2].set(xlabel='Time (s)',ylabel='Actual speed (mm/s)',ylim=(0,104),title='f  Complete-path speed')
        axes[1,0].axhline(abs(np.rad2deg(angle)),color='#999999',ls=':',lw=.7)
        for ax in axes.flat:
            ax.grid(alpha=.15)
            if ax is not axes[0,0]:ax.axhline(0,color='#999999',lw=.6)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle(f'Training {abs(np.rad2deg(angle)):.0f} deg | learned checkpoints; neither promoted',fontsize=12)
        fig.savefig(stage/f'path{path}.png',dpi=150);plt.close(fig)
    write_json(stage/'metrics.json',dict(records=records,goal_complete=False))
    receipt(stage,dict(status='rendered_pending_visual_review',new_physics_ticks=0,new_actor_decisions=0))


if __name__=='__main__':main()
