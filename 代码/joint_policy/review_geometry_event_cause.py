"""Exploratory raw-XY diagnostic, not a manuscript or performance figure."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from complete_corner_metrics import complete_corner_metrics
from study_io import BASE, register, receipt, write_json


def main():
    parent=BASE/'exit_local_credit_parameter_learning_20260916/training_1/13.npz'
    updated=BASE/'matched_single_decision_credit_learning_20260917/independent_zero_return_0.25/13.npz'
    probe=BASE/'preswing_sampled_geometry_causal_probe_20260917/p13_quarter_22_111.npz'
    paths=[parent,updated,probe]
    stage=register('sampled_geometry_causal_xy_review_20260917',[__file__]+paths,
        dict(new_physics_ticks=0,new_decisions=0,wall_s=30),
        'Exploratory inspection of raw physical XY, both signed edge distances, first heading '
        'direction, full curvature and speed after one identified feature-event intervention. '
        'Equal XY aspect, explicit distance/time zooms, unsmoothed records. Three trajectories '
        'are deterministic conditional examples, not seeds, learned-intervention policies '
        'or final acceptance. No journal styling or visual-precedent claim.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),layout='constrained')
    labels=['Retained learned actor','Rejected learned update','One-decision diagnostic intervention']
    colors=['#606970','#CF702E','#276D9E'];metrics=[]
    for path,label,color in zip(paths,labels,colors):
        with np.load(path) as z:ref,rows,actions=z['reference'],z['rows'],z['actions']
        xy=rows[:,:2]-ref[1];relative=ref-ref[1]
        u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
        angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);sign=np.sign(angle)
        ni=sign*np.array([-u[1],u[0]]);no=sign*np.array([-v[1],v[0]])
        t=np.arange(1,len(rows)+1)*.001;speed=np.linalg.norm(rows[:,2:4],axis=1)
        moving=speed>5;heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@ni,rows[:,2:4]@u)))
        curvature=sign*(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
        axes[0,0].plot(xy[:,0],xy[:,1],color=color,lw=1.4,label=label)
        axes[0,1].plot(t,xy@ni,color=color,lw=1.5)
        axes[0,2].plot(t,xy@no,color=color,lw=1.5)
        axes[1,0].plot(t[moving],heading[moving],color=color,lw=1.3)
        axes[1,1].plot(t[moving],curvature,color=color,lw=1.2)
        axes[1,2].plot(t,speed,color=color,lw=1.3)
        metrics.append(dict(label=label,trace=str(path.relative_to(BASE)),
            **complete_corner_metrics(ref,rows)))
    axes[0,0].plot(relative[:,0],relative[:,1],'--',color='#999999',lw=.8)
    axes[0,0].plot(0,0,'o',color='#222222',ms=3)
    axes[0,0].set(xlim=(-12,2),ylim=(-1,8),xlabel='x from vertex (mm)',ylabel='y from vertex (mm)',title='a  Actual XY: corner crop')
    axes[0,0].set_aspect('equal',adjustable='box')
    axes[0,1].set(xlim=(0,.55),ylim=(-.04,.06),xlabel='Time (s)',ylabel='Incoming signed distance (mm)',title='b  Entry distance zoom: negative = outside')
    axes[0,2].set(xlim=(.5,.86),ylim=(-.12,.12),xlabel='Time (s)',ylabel='Outgoing signed distance (mm)',title='c  Exit distance zoom: negative = outside')
    axes[1,0].set(xlim=(.05,.52),ylim=(-2,6),xlabel='Time (s)',ylabel='Physical heading (deg)',title='d  Entry heading zoom: negative = reverse')
    axes[1,1].set(xlabel='Time (s)',ylabel='Signed curvature (1/mm)',title='e  Complete moving-path curvature')
    axes[1,2].set(xlabel='Time (s)',ylabel='Actual speed (mm/s)',title='f  Complete-path speed')
    for ax in axes.flat:
        ax.grid(alpha=.15)
        if ax is not axes[0,0]:ax.axhline(0,color='#777777',lw=.6)
    for ax in [axes[0,1],axes[1,0],axes[1,1],axes[1,2]]:
        ax.axvspan(.44,.46,color='#CF702E',alpha=.10)
    handles,_=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('A sampled-geometry event amplifies reverse entry; the intervention does not solve the full turn',fontsize=12)
    fig.savefig(stage/'diagnostic.png',dpi=160);fig.savefig(stage/'diagnostic.svg')
    write_json(stage/'metrics.json',dict(records=metrics,goal_complete=False))
    receipt(stage,dict(status='rendered_pending_visual_review',new_physics_ticks=0,new_decisions=0))


if __name__=='__main__':main()
