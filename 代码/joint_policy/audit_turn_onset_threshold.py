"""Check direction-label sensitivity against raw signed XY displacement."""
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from isolated_corner_shape import corner_shape,first_run
from bounded_shape_return import terminal_shape_cost
from study_io import BASE,register,receipt,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    ids=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    source=BASE/'matched_endpoint_representation_learning_20260916'
    groups=[('original',BASE/'constrained_parameter_step_probe_20260916/training_3','#92979d'),
            ('full50',source/'full_features_training_50','#d08a31'),
            ('masked50',source/'mask_endpoint_bearing_training_50','#176ca5')]
    stage=BASE/'turn_onset_label_audit_20260916' if args.render_only else register('turn_onset_label_audit_20260916',[__file__,BASE/'isolated_corner_shape.py',BASE/'bounded_shape_return.py',source/'receipt.json']+
        [d/f'{i}.npz' for _,d,_ in groups for i in range(10)],dict(physics_ticks=0,new_decisions=0),
        'Audit threshold-dependent first-turn labels on saved complete training trajectories. '
        'Compare heading onset thresholds and actual signed lateral displacement; expose exact original '
        'reward contribution of its discrete first-turn label. Do not replace full-shape assessment by a new scalar.')
    records=[];fig,axes=plt.subplots(3,3,figsize=(14,10))
    for label,directory,color in groups:
        for i in range(10):
            with np.load(directory/f'{i}.npz') as z:ref,rows,actions=z['reference'],z['rows'],z['actions']
            u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);sign=np.sign(u[0]*v[1]-u[1]*v[0]);normal=sign*np.array([-u[1],u[0]])
            xy=rows[:,:2]-ref[1];inside=xy@normal;longitudinal=xy@u
            speed=np.linalg.norm(rows[:,2:4],axis=1);heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u)))
            apex=np.argmin(np.linalg.norm(xy,axis=1));inbound=np.arange(len(rows))<=apex
            onset=[]
            for threshold in [.05,.1,.2,.5,1.]:
                pos=first_run(inbound&(speed>5)&(heading>threshold));neg=first_run(inbound&(speed>5)&(heading<-threshold))
                first=min([v for v in [pos,neg] if v is not None],default=None)
                onset.append(dict(threshold_deg=threshold,first=('inside' if first==pos else 'outside') if first is not None else None,
                                  first_inside_tick=pos,first_outside_tick=neg))
            displacement=[]
            for threshold in [.0001,.001,.005,.01,.02]:
                pos=np.flatnonzero(inbound&(inside>threshold));neg=np.flatnonzero(inbound&(inside<-threshold))
                a=int(pos[0]) if len(pos) else None;b=int(neg[0]) if len(neg) else None
                first=min([v for v in [a,b] if v is not None],default=None)
                displacement.append(dict(threshold_mm=threshold,first=('inside' if first==a else 'outside') if first is not None else None))
            s=corner_shape(ref,rows,actions);cost,details=terminal_shape_cost(ref,rows,actions)
            q_without_first=details['quality']-.5*(s['first_physical_turn']!='inside')
            records.append(dict(model=label,path=ids[i],original_shape=s,heading_onset=onset,displacement_onset=displacement,
                original_cost=float(cost.sum()),same_trace_cost_without_discrete_first_label=float(-np.expm1(-q_without_first))))
            if i not in [2,3,4]:continue
            c=[2,3,4].index(i)
            axes[0,c].plot(longitudinal,inside,c=color,label=label)
            axes[0,c].set(xlim=(-30,0),ylim=(-.06,.12),xlabel='Incoming longitudinal position (mm)',ylabel='Signed inside displacement (mm)',title=f'Training path{ids[i]}')
            axes[0,c].axhline(0,color='.6',lw=.7)
            axes[1,c].plot(longitudinal,heading,c=color)
            axes[1,c].set(xlim=(-30,-3),ylim=(-1.3,1.3),xlabel='Incoming longitudinal position (mm)',ylabel='Signed heading (deg)')
            axes[1,c].axhline(.5,color='.6',ls='--',lw=.7);axes[1,c].axhline(-.5,color='.6',ls='--',lw=.7)
            angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);alpha=-angle/2
            rotation=np.array([[np.cos(alpha),np.sin(alpha)],[-np.sin(alpha),np.cos(alpha)]])
            rotated=xy@rotation;rr=(ref-ref[1])@rotation
            rotated[:,1]*=sign;rr[:,1]*=sign
            axes[2,c].plot(*rotated.T,c=color)
            if label=='original':axes[2,c].plot(*rr.T,c='.6',ls='--',lw=.8)
            axes[2,c].set_aspect('equal',adjustable='box');axes[2,c].set(xlim=(-7,7),ylim=(-.3,6),xlabel='Rotated x (mm)',ylabel='Rotated y (mm)')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(stage/'diagnostic.png',dpi=150)
    if not args.render_only:
        write_json(stage/'results.json',dict(records=records,goal_complete=False))
        receipt(stage,dict(status='saved_XY_onset_and_reward_label_audit_complete',physics_ticks=0,new_decisions=0,visual_review_pending=True))
    for r in records:
        if r['model']=='masked50':print(r['path'],r['original_shape']['first_physical_turn'],[x['first'] for x in r['heading_onset']],[x['first'] for x in r['displacement_onset']],r['original_cost'],r['same_trace_cost_without_discrete_first_label'],flush=True)


if __name__=='__main__':main()
