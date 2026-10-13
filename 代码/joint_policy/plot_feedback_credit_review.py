"""Saved physical evidence only: no interpolation smoothing or new rollouts."""
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_io import BASE,register,receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',action='store_true');args=parser.parse_args()
    groups=[('Feedback initialization (10/12)',BASE/'interior_teacher_feedback_learning_20260916/validation_50','#8b8d92'),
            ('PPO19716 (12/12)',BASE/'bounded_shape_full_feedback_ppo_20260916/validation_19716','#216ca7'),
            ('History ablation200 (8/12)',BASE/'history_independent_feedback_learning_20260916/validation_200','#b46b31')]
    if args.step:
        groups=[('Earlier PPO19716 (12/12)',BASE/'bounded_shape_full_feedback_ppo_20260916/validation_19716','#8b8d92'),
                ('Learned parameter step0.5 (12/12)',BASE/'constrained_parameter_step_probe_20260916/validation_1','#216ca7'),
                ('Heading feedback2000 (12/12)',BASE/'heading_frame_feedback_learning_20260916/validation_2000','#b46b31')]
    stage=register('constrained_step_xy_review_20260916' if args.step else 'feedback_mechanism_xy_review_20260916',[__file__]+[directory/f'{i}.npz' for _,directory,_ in groups for i in (0,1,3,11)],
        dict(physics_ticks=0,training_updates=0),
        'Inspect actual unsmoothed XY, heading/curvature and longest development-path speed across existing learned models. All are preliminary development candidates; no accepted model.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,8.5))
    for label,directory,color in groups:
        for col,index in enumerate((0,1,3)):
            with np.load(directory/f'{index}.npz') as z:ref,rows=z['reference'],z['rows']
            u,v=np.diff(ref,axis=0);theta=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);alpha=-theta/2
            rotation=np.array([[np.cos(alpha),np.sin(alpha)],[-np.sin(alpha),np.cos(alpha)]])
            xy=(rows[:,:2]-ref[1])@rotation;rr=(ref-ref[1])@rotation
            axes[0,col].plot(*xy.T,c=color,lw=1.5,label=label)
            if label==groups[0][0]:axes[0,col].plot(*rr.T,c='.65',ls='--',lw=1)
            axes[0,col].set_aspect('equal',adjustable='box')
            extent=9 if index<3 else 6
            axes[0,col].set(xlim=(-extent,extent),ylim=(-.4,7 if index<3 else 11),xlabel='Rotated x (mm)',ylabel='Rotated y (mm)',title=f'{abs(np.rad2deg(theta)):.0f} degree: actual XY')
            if index!=3:continue
            speed=np.linalg.norm(rows[:,2:4],axis=1);apex=np.argmin(np.linalg.norm(rows[:,:2]-ref[1],axis=1));tt=(np.arange(len(rows))-apex)*.001
            heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,3],rows[:,2])))
            k=np.full(len(rows),np.nan);valid=speed>5;k[valid]=(rows[valid,2]*rows[valid,5]-rows[valid,3]*rows[valid,4])/speed[valid]**3
            axes[1,0].plot(tt,heading,c=color,lw=1.5);axes[1,1].plot(tt,k,c=color,lw=1.5)
        with np.load(directory/'11.npz') as z:rows=z['rows']
        axes[1,2].plot(np.arange(1,len(rows)+1)*.001,np.linalg.norm(rows[:,2:4],axis=1),c=color,lw=1.5)
    for i,y in enumerate(('150-degree heading (deg)','150-degree curvature proxy (1/mm)')):
        axes[1,i].set(xlim=(-.4,.3),xlabel='Time from nearest vertex approach (s)',ylabel=y)
    axes[1,2].set(xlabel='Complete-path time (s)',ylabel='Longest development path speed (mm/s)',ylim=(0,105))
    axes[1,2].axhline(95,c='.7',ls='--',lw=.8)
    for ax in axes.flat:ax.grid(alpha=.16)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.965))
    fig.suptitle('Learned step backtracking preserves broader inward turns; cruising still requires improvement' if args.step else 'Completion recovery does not yet yield broad smooth turns or high-speed cruising',y=.995)
    fig.text(.5,.01,'1 ms XY without smoothing; equal metric scale. Blue:12/12 complete, small0.02-0.03mm approach excursions remain. No model has final acceptance.' if args.step else '1 ms XY is shown without smoothing. Rotation preserves metric scale. First turns are inward; remaining reversals and sharp bends still require work.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.935));fig.savefig(stage/'diagnostic.png',dpi=160)
    receipt(stage,dict(status='rendered_for_visual_inspection',physics_ticks=0,training_updates=0,visual_qa_pending=True))


if __name__=='__main__':main()
