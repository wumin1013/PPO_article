"""Unsmoothed equal-aspect evidence separating learned policy and intervention."""
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_io import BASE,register,receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--learned',action='store_true');args=parser.parse_args()
    groups=[('Learned balanced policy',BASE/'balanced_shape_time_learning_20260916/validation_5/3.npz','#777777'),
            ('Learned state residual',BASE/'shape_pareto_candidate_review_20260916/validation_1/3.npz','#245f96'),
            ('Finite intervention F=40, D=8, steering=0',BASE/'wide_turn_reachability_probe_20260916/feed40_D8.npz','#bc722b')]
    if args.learned:groups[1]=('Feedback network epoch50 (9/12)',BASE/'consistent_teacher_feedback_learning_20260916/validation_50/3.npz','#245f96')
    stage=register('learned_wide_turn_transfer_review_20260916' if args.learned else 'wide_turn_reachability_review_20260916',[__file__]+[f for _,f,_ in groups],
        dict(physics_ticks=0,training_updates=0),
        'True XY, heading and speed review of a finite diagnostic intervention versus learned policies. No smoothing; no claim that the wide diagnostic trajectory has been learned.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for label,file,color in groups:
        with np.load(file) as d:ref,rows=d['reference'],d['rows']
        xy=rows[:,:2]-ref[1];speed=np.linalg.norm(rows[:,2:4],axis=1)
        apex=int(np.argmin(np.linalg.norm(xy,axis=1)));time=(np.arange(len(rows))-apex)*.001
        heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,3],rows[:,2])))
        moving=speed>5;curvature=np.full(len(rows),np.nan)
        curvature[moving]=(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
        axes[0,0].plot(*xy.T,color=color,label=label,lw=1.8)
        for ax,y in [(axes[0,1],heading),(axes[1,0],curvature),(axes[1,1],speed)]:
            ax.plot(time,y,color=color,lw=1.6);ax.set_xlim(-.35,.3);ax.axvline(0,color='.85',lw=.7)
    rr=ref-ref[1];axes[0,0].plot(*rr.T,color='.65',ls='--',lw=1);axes[0,0].scatter([0],[0],s=20,color='.3')
    axes[0,0].set(xlim=(-12,2),ylim=(-1,6.5),xlabel='x relative to vertex (mm)',ylabel='y relative to vertex (mm)',title='150-degree turn: actual equal-aspect XY')
    axes[0,0].set_aspect('equal',adjustable='box')
    for ax,label in [(axes[0,1],'Velocity heading (deg)'),(axes[1,0],'Curvature proxy (1/mm)'),(axes[1,1],'Speed (mm/s)')]:
        ax.set(xlabel='Time from nearest vertex approach (s)',ylabel=label)
    axes[1,1].set_ylim(0,105);axes[1,1].axhline(100,color='.6',ls='--',lw=.8)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.955))
    fig.suptitle('Learned wider 150-degree turn; full-path robustness remains unresolved' if args.learned else 'A wider inward transition is physically reachable; it has not yet been learned',fontsize=14,y=.995)
    fig.text(.5,.015,'All three shown single corners finish and stop. Blue fails three other development paths; not accepted. Orange is an offline intervention.' if args.learned else 'Orange: six-cell offline reachability probe, fixed 24-decision intervention. All shown paths fully stop. Development evidence only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.92));fig.savefig(stage/'diagnostic.png',dpi=170)
    receipt(stage,dict(status='rendered',physics_ticks=0,training_updates=0,visual_qa_pending=True))


if __name__=='__main__':main()
