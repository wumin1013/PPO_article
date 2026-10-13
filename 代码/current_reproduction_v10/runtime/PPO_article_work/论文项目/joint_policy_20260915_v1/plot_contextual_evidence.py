"""Inspect saved XY and physical derivatives without any new rollout."""
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from isolated_corner_shape import corner_shape
from study_io import BASE, register, receipt, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selected-local',action='store_true');args=parser.parse_args()
    groups=[('Shape initialization',BASE/'constrained_parameter_step_probe_20260916/validation_1','#92979d'),
            ('Contextual PPO22509',BASE/'contextual_checkpoint_recovery_20260916/validation_22509','#196ba4')]
    if args.selected_local:
        groups[1]=('Local-geometry PPO18645',BASE/'local_geometry_continuous_reward_ppo_20260916/validation_18645','#196ba4')
    stage=register('local_geometry_selected_xy_review_20260916' if args.selected_local else 'contextual_xy_review_20260916',[__file__]+[d/f'{i}.npz' for _,d,_ in groups for i in (0,1,2,3,11)],
        dict(physics_ticks=0,new_decisions=0),
        'Review actual1ms XY, signed heading/curvature and long development speed. Equal XY axes; '
        'only rigid rotations, no trajectory smoothing. All outputs remain development evidence.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,12));gs=fig.add_gridspec(4,4,height_ratios=[1.6,1,1,.9])
    axes=[[fig.add_subplot(gs[r,c]) for c in range(4)] for r in range(3)]
    cruise=fig.add_subplot(gs[3,:]);metrics=[]
    for label,directory,color in groups:
        for i in range(4):
            with np.load(directory/f'{i}.npz') as z:ref,rows,actions=z['reference'],z['rows'],z['actions']
            u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u)
            angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);sign=np.sign(angle)
            alpha=-angle/2;rotation=np.array([[np.cos(alpha),np.sin(alpha)],[-np.sin(alpha),np.cos(alpha)]])
            xy=(rows[:,:2]-ref[1])@rotation;rr=(ref-ref[1])@rotation
            axes[0][i].plot(*xy.T,color=color,lw=1.6,label=label)
            if label==groups[0][0]:axes[0][i].plot(*rr.T,color='.6',ls='--',lw=.9)
            axes[0][i].set_aspect('equal',adjustable='box')
            extent=7 if i<3 else 4
            axes[0][i].set(xlim=(-extent,extent),ylim=(-.3,8),xlabel='Rotated x (mm)',ylabel='Rotated y (mm)',title=f'{abs(np.rad2deg(angle)):.0f} degree / actual XY')
            speed=np.linalg.norm(rows[:,2:4],axis=1)
            apex=np.argmin(np.linalg.norm(rows[:,:2]-ref[1],axis=1));tt=(np.arange(len(rows))-apex)*.001
            normal=sign*np.array([-u[1],u[0]])
            heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u)))
            kappa=np.full(len(rows),np.nan);valid=speed>5
            kappa[valid]=sign*(rows[valid,2]*rows[valid,5]-rows[valid,3]*rows[valid,4])/speed[valid]**3
            axes[1][i].plot(tt,heading,color=color,lw=1.4)
            axes[2][i].plot(tt,kappa,color=color,lw=1.4)
            shape=corner_shape(ref,rows,actions);metrics.append(dict(model=label,path=i,**shape))
            if label==groups[-1][0]:
                axes[0][i].text(.02,.98,f"outward {shape['outward_excursion_mm']:.4f} mm\n< -0.02mm: {shape['outward_position_duration_s']*1000:.0f} ms\nreverse {shape['reverse_turning_total_deg']:.2f} deg",va='top',transform=axes[0][i].transAxes,fontsize=8)
        with np.load(directory/'11.npz') as z:rows=z['rows']
        cruise.plot(np.arange(1,len(rows)+1)*.001,np.linalg.norm(rows[:,2:4],axis=1),c=color,lw=1.4,label=label)
    for row in (1,2):
        for ax in axes[row]:ax.set(xlim=(-.35,.25),xlabel='Time from closest vertex approach (s)',ylabel='Signed heading (deg)' if row==1 else 'Signed curvature (1/mm)')
    cruise.set(xlabel='Complete-path time (s)',ylabel='Speed (mm/s)',ylim=(0,103));cruise.axhline(95,color='.6',ls='--',lw=.8)
    for ax in [a for row in axes for a in row]+[cruise]:ax.grid(alpha=.16)
    cruise.legend(frameon=False,loc='lower center',ncol=2)
    fig.suptitle('Both complete12/12 development paths; inward entry retains residual reversal and insufficient cruising',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(stage/'diagnostic.png',dpi=145)
    write_json(stage/'metrics.json',dict(metrics=metrics))
    receipt(stage,dict(status='rendered_pending_visual_review',new_decisions=0,physics_ticks=0))


if __name__=='__main__':main()
