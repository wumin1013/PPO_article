"""Exploratory true-XY review; no smoothing or publication claims."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_io import BASE, register, receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--latest',action='store_true');parser.add_argument('--bending',action='store_true');args=parser.parse_args()
    groups = [('PPO source (12/12)', BASE/'ar_conditional_joint_20260916/final_validation', '#6b7280'),
              ('Return search, gen. 1 (11/12)', BASE/'complete_return_search_20260916/validation_1', '#c07b31'),
              ('Return search, gen. 3 (12/12)', BASE/'complete_return_search_20260916/validation_3', '#245f96')]
    if args.latest:
        groups=[('PPO source: 20.236 s',BASE/'ar_conditional_joint_20260916/final_validation','#6b7280'),
                ('Reflected shape learning: 20.676 s',BASE/'reflection_shape_search_20260916/validation_5','#c07b31'),
                ('Correlated search: 20.503 s',BASE/'correlated_parameter_search_20260916/validation_4','#245f96')]
    if args.bending:
        groups=[('Balanced control: 20.503 s',BASE/'balanced_shape_time_learning_20260916/validation_5','#6b7280'),
                ('Bending objective g5: 20.252 s',BASE/'spatial_bending_learning_20260916/validation_5','#c07b31'),
                ('Bending objective g6: 20.178 s',BASE/'spatial_bending_learning_20260916/validation_6','#245f96')]
    files = [folder/f'{i}.npz' for _, folder, _ in groups for i in (1,2,3)]
    stage = register('spatial_bending_shape_review_20260916' if args.bending else 'learned_inner_shape_review_20260916' if args.latest else 'complete_return_shape_review_20260916', [__file__]+files,
        dict(trace_reads=9, physics_ticks=0, training_updates=0),
        'Exploratory visual review of actual learned XY, heading, curvature and speed; no smoothing, optimality or generalization claim.')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), gridspec_kw={'height_ratios':[1.6,1,1,1]})
    for col, index in enumerate((1,2,3)):
        for label, folder, color in groups:
            with np.load(folder/f'{index}.npz') as data:
                ref, rows = data['reference'], data['rows']
            xy = rows[:,:2]-ref[1]; speed = np.linalg.norm(rows[:,2:4],axis=1)
            apex = np.argmin(np.linalg.norm(xy,axis=1)); time = (np.arange(len(rows))-apex)*.001
            heading = np.rad2deg(np.unwrap(np.arctan2(rows[:,3],rows[:,2])))
            cross = rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4]
            curvature = np.full(len(rows), np.nan); moving=speed>5
            curvature[moving]=cross[moving]/speed[moving]**3
            axes[0,col].plot(xy[:,0],xy[:,1],color=color,lw=1.6,label=label)
            for ax,y in zip(axes[1:,col],(heading,curvature,speed)):
                ax.plot(time,y,color=color,lw=1.25)
                ax.set_xlim(-.3,.22);ax.axvline(0,color='.8',lw=.7);ax.grid(alpha=.13)
        rr=ref-ref[1]
        axes[0,col].plot(rr[:,0],rr[:,1],color='.6',ls='--',lw=1,zorder=0)
        axes[0,col].scatter([0],[0],s=18,color='.25',zorder=5)
        axes[0,col].set(xlim=(-12,4),ylim=(-1.7,8),xlabel='x relative to vertex (mm)',title=f'{[95,125,150][col]} degree turn')
        axes[0,col].set_aspect('equal',adjustable='box');axes[0,col].grid(alpha=.13)
        axes[1,col].axhline(0,color='.65',lw=.8);axes[2,col].axhline(0,color='.65',lw=.8)
        axes[3,col].axhline(100,color='.6',ls='--',lw=.8);axes[3,col].set_ylim(0,105)
        axes[3,col].set_xlabel('Time from nearest vertex approach (s)')
    for ax,label in zip(axes[:,0],('y relative to vertex (mm)','Velocity heading (deg)','Discrete curvature proxy (1/mm)','Speed (mm/s)')):
        ax.set_ylabel(label)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.975))
    fig.suptitle('Bending-energy objective: sharp-turn concentration and exit corrections remain' if args.bending else 'Learned inward approach; sharp-turn speed and cruising remain unresolved' if args.latest else 'Complete-time improvement does not establish acceptable corner shape',y=.998,fontsize=14)
    fig.text(.5,.008,'Recorded 1 ms physical samples; equal-aspect local XY zoom. All three methods: 12/12 development completion. No independent test.' if args.latest or args.bending else 'Recorded 1 ms physical samples. XY uses equal aspect. Gen. 1 failed a separate mixed path. Development evidence only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.95))
    fig.savefig(stage/'diagnostic.png',dpi=180)
    receipt(stage,dict(status='exploratory_diagnostic_rendered',trace_reads=9,physics_ticks=0,training_updates=0,visual_qa_pending=True))


if __name__=='__main__':main()
