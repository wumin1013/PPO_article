"""Diagnostic view bound to verified raw traces; not a publication figure."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometric_turn_metrics import spatial_turn_profile
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920'
    reference=read(stage/'raw_result_audit.json')['changed_metrics'][0]['before']
    selected=read(stage/'raw_result_audit.json')['changed_metrics'][0]['after']
    exploratory=read(BASE/'canonical_gate_band_coupled_peak_memory_learning_20260920/raw_result_audit.json')['changed_metrics'][0]['after']
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    bindings=[]
    for label,r,color in [('Protected parent',reference,'#333333'),('Held: duration regression',exploratory,'#bb6633'),('Guarded postpeak learning',selected,'#2266bb')]:
        with np.load(r['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        xy=rows[:,:2]-ref[1];u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
        sign=np.sign(u[0]*v[1]-u[1]*v[0]);no=sign*np.array([-v[1],v[0]])
        time=np.arange(1,len(rows)+1)*.001;speed=np.linalg.norm(rows[:,2:4],axis=1)
        axes[0,0].plot(xy[:,0],xy[:,1],color=color,label=label,lw=1.3)
        axes[0,1].plot(time,-xy@no,color=color,lw=1.3)
        axes[1,0].plot(time,speed,color=color,lw=1.3)
        profile=spatial_turn_profile(np.vstack([ref[0],rows[:,:2]]))
        axes[1,1].plot(profile['grid'][1:-1],np.abs(profile['turn'])/.1,color=color,lw=1.3)
        bindings.append(dict(label=label,trace=r['trace'],sha256=sha(r['trace'])))
    axes[0,0].plot((ref-ref[1])[:,0],(ref-ref[1])[:,1],'k--',lw=.7)
    axes[0,0].set(xlim=(-7,1),ylim=(-4,1),xlabel='X relative to vertex (mm)',ylabel='Y relative to vertex (mm)',title='Actual XY: unresolved outgoing crossing')
    axes[0,0].set_aspect('equal');axes[0,0].legend(fontsize=7)
    axes[0,1].set(xlim=(.3,.55),ylim=(-.1,.19),xlabel='Time (s)',ylabel='Outgoing outward excursion (mm)',title='Amplitude and duration must both improve')
    axes[0,1].axhline(.02,color='gray',ls=':',lw=.7);axes[0,1].axhline(0,color='gray',lw=.7)
    axes[1,0].set(xlim=(.2,.65),ylim=(20,102),xlabel='Time (s)',ylabel='Actual speed (mm/s)',title='Corner speed and recovery')
    axes[1,1].set(xlim=(18,32),ylim=(0,1.6),xlabel='Travelled arc length (mm)',ylabel='Curvature magnitude (1/mm)',title='Raw XY sampled at 0.1 mm')
    for ax in axes.flat:ax.grid(alpha=.18)
    fig.suptitle('Training path 304: full physical traces; shape goal remains unmet',fontsize=12)
    file=stage/'raw_shape_duration_audit.png';fig.savefig(file,dpi=150);plt.close(fig)
    write_json(stage/'raw_shape_duration_audit_sources.json',dict(image_sha256=sha(file),
        candidate_model_sha256=sha(stage/'research_candidate.pt'),traces=bindings,new_rollouts=0,
        purpose='Diagnostic inspection only; no smoothing or ideal curve substituted for actual XY.'))
    print(file)


if __name__=='__main__':main()
