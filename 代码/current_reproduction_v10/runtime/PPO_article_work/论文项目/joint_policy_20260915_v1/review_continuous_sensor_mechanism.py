"""Inspect actual saved XY, signed heading and speed for the sensor mechanism."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    audit=BASE/'large_step_feedback_event_audit_20260917'
    probe=BASE/'large_step_single_sensor_causal_probe_20260917'
    learned=BASE/'continuous_sensor_matched_function_transfer_20260917'
    selected=next(r for r in read(learned/'results.json')['candidates'] if r['arm']=='updated')
    data=read(audit/'results.json');counterfactual=read(probe/'results.json')['records']
    stage=register('continuous_sensor_mechanism_xy_review_20260917',
        [__file__,audit/'results.json',probe/'results.json',learned/'results.json'],
        dict(new_physics_ticks=0,new_actor_calls=0,wall_s=60),
        'Actual saved1ms XY, velocity heading and speed around the two diagnosed '
        'events. Distinguish old learned parent, regressed learned parameter update, '
        'single-sensor diagnostic intervention, and new continuous-sensor learned '
        'transfer. The intervention is never a learned result. No smoothing of '
        'positions, no fitted ideal template and no promotion by visual comparison.')
    start=time.perf_counter();fig,axes=plt.subplots(2,3,figsize=(13,7.2))
    colors=['#555555','#bd4b3c','#168566','#2676bc'];records=[]
    labels=['Learned parent','Regressed learned update','One-decision probe (diagnostic)','Continuous-sensor learned transfer']
    for row,(path,step,anchor) in enumerate(((8,1.,29),(15,.5,35))):
        record=next(r for r in data['records'] if r['path']==path)
        paths=[BASE/next(t['trace'] for t in record['traces'] if t['step']==s) for s in (0.,step)]
        paths.append(probe/next(r['trace'] for r in counterfactual if r['path']==path and r['recipient_step']==step))
        j=next(i for i,r in enumerate(selected['results']) if r['original_path']==path)
        paths.append(BASE/selected['traces'][j])
        for label,color,trace in zip(labels,colors,paths):
            with np.load(trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
            u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            turn=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);normal=np.sign(turn)*np.array([-u[1],u[0]])
            speed=np.linalg.norm(rows[:,2:4],axis=1);t=(np.arange(len(rows))+1)*.001
            heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u)))
            heading[speed<=5]=np.nan
            near=np.linalg.norm(rows[:,:2]-ref[1],axis=1)<8
            axes[row,0].plot(rows[near,0],rows[near,1],color=color,lw=1.5,label=label)
            axes[row,1].plot(t,heading,color=color,lw=1.3)
            axes[row,2].plot(t,speed,color=color,lw=1.3)
            records.append(dict(path=path,label=label,trace=str(trace.relative_to(BASE))))
        reference=np.stack([ref[1]-8*u,ref[1],ref[1]+8*v])
        axes[row,0].plot(reference[:,0],reference[:,1],'--',color='#999999',lw=1,zorder=0)
        axes[row,0].scatter(*ref[1],c='#777777',marker='x',s=20,zorder=4)
        axes[row,0].set_aspect('equal',adjustable='datalim')
        axes[row,0].set(xlabel='Actual X (mm)',ylabel='Actual Y (mm)',title=f'Path {path}: {abs(np.rad2deg(turn)):.0f} degree turn')
        axes[row,1].axhline(abs(np.rad2deg(turn)),ls=':',color='#888888',lw=1)
        axes[row,1].set(xlabel='Time (s)',ylabel='Signed velocity heading (deg)',xlim=(anchor*.02-.25,anchor*.02+.35))
        axes[row,2].set(xlabel='Time (s)',ylabel='Vector speed (mm/s)',xlim=(anchor*.02-.25,anchor*.02+.35),ylim=(0,105))
        for col in (1,2):axes[row,col].axvline(anchor*.02,ls=':',color='#333333',lw=.8)
        for ax in axes[row]:ax.grid(alpha=.18)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle('A discrete sensing event has a causal effect; continuous sensing still requires learned shape repair',fontsize=12)
    fig.tight_layout(rect=(0,.09,1,.96));fig.savefig(stage/'actual_xy_heading_speed.png',dpi=180)
    plt.close(fig)
    write_json(stage/'sources.json',dict(records=records,positions_smoothed=False,goal_complete=False))
    receipt(stage,dict(status='rendered_actual_saved_sensor_mechanism',wall_s=time.perf_counter()-start,
        visual_inspection_pending=True,new_physics_ticks=0))


if __name__=='__main__':main()
