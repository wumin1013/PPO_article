"""Full-path physical review of the selected local-credit learned actor."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from complete_corner_metrics import complete_corner_metrics
from study_io import BASE,register,receipt,write_json


def main():
    groups=[('PPO18645',BASE/'local_geometry_continuous_reward_ppo_20260916/validation_18645','#9AA0A6'),
            ('Local-credit learned actor',BASE/'exit_local_credit_parameter_learning_20260916/development','#256B9B')]
    stage=register('exit_local_credit_full_xy_review_20260916',
        [__file__]+[d/f'{i}.npz' for _,d,_ in groups for i in (0,1,2,3,11)],
        dict(new_decisions=0,new_physics_ticks=0,wall_s=30),
        'Inspect complete physical XY, both edge-side errors, full heading and curvature, and full longest '
        'development speed. No trajectory smoothing. All XY axes equal aspect; outgoing side error '
        'panel explicitly zooms the signed distance, while XY and heading span the full path.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,13));grid=fig.add_gridspec(5,4,height_ratios=[1.8,1,1,1,.85],hspace=.52,wspace=.35)
    axes=[[fig.add_subplot(grid[r,c]) for c in range(4)] for r in range(4)]
    speed_ax=fig.add_subplot(grid[4,:]);records=[];cruise=[]
    for label,folder,color in groups:
        for i in range(4):
            with np.load(folder/f'{i}.npz') as z:ref,rows=z['reference'],z['rows']
            u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);sign=np.sign(angle)
            normal=sign*np.array([-u[1],u[0]]);outnormal=sign*np.array([-v[1],v[0]])
            xy=rows[:,:2]-ref[1];local=np.column_stack([xy@u,xy@normal]);rr=ref-ref[1]
            rr=np.column_stack([rr@u,rr@normal]);a=-abs(angle)/2
            rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            local=local@rotation.T;rr=rr@rotation.T
            axes[0][i].plot(*local.T,color=color,lw=1.6,label=label)
            if label==groups[0][0]:axes[0][i].plot(*rr.T,'--',color='#666666',lw=.8)
            axes[0][i].set_aspect('equal',adjustable='box')
            axes[0][i].set(xlabel='Rotated x (mm)',ylabel='Rotated y (mm)',title=f'{abs(np.rad2deg(angle)):.0f} deg: complete XY')
            arc=np.cumsum(np.linalg.norm(np.diff(np.vstack([ref[0],rows[:,:2]]),axis=0),axis=1))
            apex=int(np.argmin(np.linalg.norm(xy,axis=1)));arc-=arc[apex]
            speed=np.linalg.norm(rows[:,2:4],axis=1);valid=speed>5
            heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u)))
            kappa=sign*(rows[valid,2]*rows[valid,5]-rows[valid,3]*rows[valid,4])/speed[valid]**3
            axes[1][i].plot(arc,xy@outnormal,color=color,lw=1.5)
            axes[2][i].plot(arc[valid],heading[valid],color=color,lw=1.4)
            axes[3][i].plot(arc[valid],kappa,color=color,lw=1.2)
            records.append(dict(model=label,path=i,**complete_corner_metrics(ref,rows)))
        with np.load(folder/'11.npz') as z:rows=z['rows']
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        speed_ax.plot(np.arange(1,len(rows)+1)*.001,speed,color=color,lw=1.5,label=label)
        cruise.append(dict(model=label,peak=float(speed.max()),ge95_s=float((speed>=95).sum()*.001),
                           ge98_s=float((speed>=98).sum()*.001),ge99_s=float((speed>=99).sum()*.001)))
    for i in range(4):
        axes[1][i].axhline(0,color='#444444',lw=.8,ls='--');axes[1][i].set(ylim=(-.6,.6),ylabel='Outgoing signed distance (mm)')
        axes[1][i].set_title('Distance zoom: negative = outside',fontsize=9)
        axes[2][i].axhline([50,95,125,150][i],color='#444444',ls='--',lw=.8)
        axes[2][i].set_ylabel('Physical heading (deg)')
        axes[3][i].axhline(0,color='#444444',lw=.8,ls='--');axes[3][i].set_ylabel('Signed curvature (1/mm)')
        for row in (1,2,3):axes[row][i].set_xlabel('Full arc from closest approach (mm)')
    speed_ax.axhline(98,color='#666666',ls='--',lw=.8)
    speed_ax.set(xlabel='Complete-path time (s)',ylabel='Speed (mm/s)',ylim=(0,103))
    speed_ax.legend(frameon=False,loc='lower center',ncol=2)
    for ax in [a for row in axes for a in row]+[speed_ax]:ax.grid(alpha=.18)
    fig.suptitle('Learned local credit improves shallow exit; residual overshoot and sharp-turn concentration remain',fontsize=13,y=.99)
    fig.subplots_adjust(left=.065,right=.98,bottom=.055,top=.955)
    fig.savefig(stage/'diagnostic.png',dpi=150);fig.savefig(stage/'diagnostic.pdf')
    write_json(stage/'metrics.json',dict(shapes=records,cruise=cruise,goal_complete=False))
    receipt(stage,dict(status='rendered_pending_visual_review',new_decisions=0,new_physics_ticks=0))
    print(cruise)


if __name__=='__main__':main()
