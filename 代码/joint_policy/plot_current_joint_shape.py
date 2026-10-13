"""Compact raw-trajectory diagnostic, not a manuscript or acceptance figure."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joint_runtime import FILTER_TAU,wrap
from search_complete_return import read
from study_io import BASE,write_json,sha


def main():
    stage=BASE/'canonical_gate_band_direct_observable_joint_learning_20260920'
    result=read(stage/'results.json')
    selected=next(c for c in result['candidates'] if c['tag']==result['best'])
    records=selected['records']
    audit=read(stage/'raw_result_audit.json')
    assert audit['model_sha256']==sha(stage/'research_candidate.pt')
    trace_hashes={r['path']:r['sha256'] for r in audit['raw_trace_sha256']}
    fig,axes=plt.subplots(2,3,figsize=(12,6.4),layout='constrained')
    sources=[]
    for row,path in enumerate((206,304)):
        r=next(r for r in records if r['path']==path)
        assert sha(r['trace'])==trace_hashes[path]
        with np.load(r['trace']) as z:ref,raw=z['reference'].copy(),z['rows'].copy()
        u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
        sign=np.sign(u[0]*v[1]-u[1]*v[0]);normal=sign*np.array([-u[1],u[0]])
        xy=np.column_stack([(raw[:,:2]-ref[1])@u,(raw[:,:2]-ref[1])@normal])
        reference=np.column_stack([(ref-ref[1])@u,(ref-ref[1])@normal])
        t=np.arange(1,len(raw)+1)*.001;speed=np.linalg.norm(raw[:,2:4],axis=1)
        filtered=np.arctan2(raw[:,9],raw[:,8]);origin=math.atan2(u[1],u[0])
        previous=np.r_[origin,filtered[:-1]]
        target=previous+raw[:,10]*.001/(1-math.exp(-.001/FILTER_TAU))
        target=np.degrees([sign*wrap(x-origin) for x in target]);target[raw[:,12]!=0]=np.nan
        heading=np.degrees(np.unwrap(np.arctan2(raw[:,2:4]@normal,raw[:,2:4]@u)))
        heading[speed<=5]=np.nan
        apex=int(np.linalg.norm(raw[:,:2]-ref[1],axis=1).argmin())
        ax=axes[row,0];ax.plot(reference[:,0],reference[:,1],'--',color='.55',label='Reference')
        ax.plot(xy[:,0],xy[:,1],color='#12627e',label='Actual XY')
        ax.scatter(*xy[apex],s=22,color='#bc4937',label='Closest approach')
        ax.set(xlim=(-6,1),ylim=(-.4,3.6),aspect='equal',xlabel='Incoming coordinate (mm)',ylabel='Inside coordinate (mm)',title=f'Path {path}: physical corner')
        ax.legend(fontsize=7)
        ax=axes[row,1];ax.plot(t,heading,color='#12627e',label='Velocity heading')
        ax.plot(t,target,color='#bc4937',alpha=.8,label='Combined requested heading')
        ax.axhline(abs(r['shape']['turn_deg']),color='.55',linestyle='--',linewidth=.8)
        ax.set(xlabel='Time (s)',ylabel='Heading (deg)',title=f"Reverse {r['shape']['full_physical_reverse_deg']:.3f} deg; exit {r['shape']['out_edge_outward_mm']:.4f} mm")
        ax.legend(fontsize=7)
        ax=axes[row,2];ax.plot(t,speed,color='#12627e');ax.axhline(99,color='.55',linestyle='--',linewidth=.8)
        ax.fill_between(t,0,speed,where=speed>=99,color='#5ba88b',alpha=.4)
        ax.set(xlabel='Time (s)',ylabel='Speed (mm/s)',ylim=(0,104),title=f"Complete {r['complete_time_s']:.3f} s; >=99 cruise {r['motion']['cruise_ge99_s']:.3f} s")
        for ax in axes[row]:ax.grid(alpha=.15)
        sources.append(dict(path=path,trace=r['trace'],sha256=sha(r['trace'])))
    fig.suptitle('Current research candidate: saved 1 ms trajectories (shape not accepted)',fontsize=12)
    fig.savefig(stage/'current_shape_evidence_verified.png',dpi=180)
    write_json(stage/'current_shape_evidence_verified_sources.json',dict(sources=sources,raw_xy_unsmoothed=True,
        model_sha256=audit['model_sha256'],selected=result['best'],
        result_sha256=sha(stage/'results.json'),
        supersedes='current_shape_evidence.png used baseline parent trajectories despite its candidate title; retained as historical evidence only.',
        moving_heading_threshold_mm_s=5.,combined_target_reconstructed_from_recorded_filter_increment=True))
    plt.close(fig)
    print(stage/'current_shape_evidence_verified.png')


if __name__=='__main__':main()
