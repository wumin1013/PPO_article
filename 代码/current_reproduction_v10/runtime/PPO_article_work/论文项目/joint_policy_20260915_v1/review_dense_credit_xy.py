"""Raw physical shape review for the learned dense-credit candidate."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from complete_corner_metrics import complete_corner_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protected',action='store_true');parser.add_argument('--residual',action='store_true')
    parser.add_argument('--parent');parser.add_argument('--stage');args=parser.parse_args()
    old=BASE/'exit_local_credit_parameter_learning_20260916'
    new=BASE/('protected_residual_complete_function_credit_learning_20260917' if args.residual else 'protected_complete_function_credit_learning_20260917' if args.protected else 'matched_dense_corner_credit_learning_20260917')
    if args.parent:new=BASE/args.parent
    selected=read(new/'training_selection.json')
    cases=[('Training140deg left',old/'training_1/13.npz',BASE/selected['traces'][13]),
           ('Development50deg left',old/'development/0.npz',new/'development/0.npz'),
           ('Development150deg left',old/'development/3.npz',new/'development/3.npz')]
    stage=register(args.stage or ('residual_credit_learned_xy_review_20260917' if args.residual else 'protected_credit_learned_xy_review_20260917' if args.protected else 'dense_credit_learned_xy_review_20260917'),
        [__file__,new/'receipt.json',new/selected['model']]+[p for _,a,b in cases for p in (a,b)],
        dict(new_physics_ticks=0,new_decisions=0,wall_s=60),
        'Inspect learned-policy trajectories, not action interventions: raw equal-aspect XY, '
        'entry and exit signed-distance zooms, heading, curvature and speed. Reuse64/64 '
        'training and12/12 development evidence and review known remaining defects. '
        'This is exploratory diagnostics; no final shape, cruise or generalization claim.')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    metrics=[]
    for ci,(title,oldfile,newfile) in enumerate(cases):
        fig,axes=plt.subplots(2,3,figsize=(13.5,7.3),layout='constrained')
        for label,file,color in [('Previous learned actor',oldfile,'#747B81'),('Residual-credit learned actor' if args.residual else 'Protected-credit learned actor' if args.protected else 'Dense-credit learned actor',newfile,'#276D9E')]:
            with np.load(file) as z:ref=z['reference'];rows=z['rows']
            xy=rows[:,:2]-ref[1];u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v);sign=np.sign(angle)
            ni=sign*np.array([-u[1],u[0]]);no=sign*np.array([-v[1],v[0]])
            t=np.arange(1,len(rows)+1)*.001;speed=np.linalg.norm(rows[:,2:4],axis=1);moving=speed>5
            heading=np.rad2deg(np.unwrap(np.arctan2(rows[:,2:4]@ni,rows[:,2:4]@u)))
            curvature=sign*(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
            axes[0,0].plot(xy[:,0],xy[:,1],color=color,lw=1.5,label=label)
            axes[0,1].plot(t,xy@ni,color=color,lw=1.3)
            axes[0,2].plot(t,xy@no,color=color,lw=1.3)
            axes[1,0].plot(t[moving],heading[moving],color=color,lw=1.3)
            axes[1,1].plot(t[moving],curvature,color=color,lw=1.3)
            axes[1,2].plot(t,speed,color=color,lw=1.3)
            metric=complete_corner_metrics(ref,rows)
            # Record onset at several headings; a single arbitrary threshold is not acceptance.
            onset={}
            for threshold in (.05,.1,.2,.5,1.):
                ids=np.flatnonzero(moving&(abs(heading)>=threshold))
                onset[str(threshold)]=None if not len(ids) else dict(time_s=float(t[ids[0]]),
                    heading_deg=float(heading[ids[0]]),signed_incoming_mm=float((xy@ni)[ids[0]]))
            metrics.append(dict(case=title,label=label,trace=str(file.relative_to(BASE)),onset=onset,**metric))
        reference=ref-ref[1];axes[0,0].plot(reference[:,0],reference[:,1],'--',color='#A0A0A0',lw=.8)
        axes[0,0].plot(0,0,'o',color='#333333',ms=3);axes[0,0].set_aspect('equal',adjustable='box')
        axes[0,0].set(xlabel='x from vertex (mm)',ylabel='y from vertex (mm)',title='a  Raw XY (corner crop)')
        axes[0,0].set_xlim(-12,10 if ci==1 else 2);axes[0,0].set_ylim(-.5,9)
        axes[0,1].set(xlabel='Time (s)',ylabel='Incoming signed distance (mm)',title='b  Entry distance zoom: outside < 0',ylim=(-.04,.08))
        axes[0,2].set(xlabel='Time (s)',ylabel='Outgoing signed distance (mm)',title='c  Exit distance zoom: outside < 0',ylim=(-.32,.1))
        axes[1,0].set(xlabel='Time (s)',ylabel='Physical heading (deg)',title='d  Complete moving-path heading')
        axes[1,1].set(xlabel='Time (s)',ylabel='Signed curvature (1/mm)',title='e  Complete moving-path curvature')
        axes[1,2].set(xlabel='Time (s)',ylabel='Actual speed (mm/s)',title='f  Complete-path speed',ylim=(0,104))
        axes[1,0].axhline(abs(np.rad2deg(angle)),color='#B0B0B0',ls=':',lw=.8)
        for ax in axes.flat:
            ax.grid(alpha=.15)
            if ax is not axes[0,0]:ax.axhline(0,color='#999999',lw=.6)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
        fig.suptitle(title+' | a learned correction, with remaining reversal / exit crossing',fontsize=12)
        fig.savefig(stage/f'case{ci}.png',dpi=150);fig.savefig(stage/f'case{ci}.svg');plt.close(fig)
    cruise=[]
    for label,folder in [('previous',old/'development'),('dense',new/'development')]:
        for i in range(12):
            with np.load(folder/f'{i}.npz') as z:speed=np.linalg.norm(z['rows'][:,2:4],axis=1)
            cruise.append(dict(model=label,path=i,peak_mm_s=float(speed.max()),
                ge95_s=float((speed>=95).sum()*.001),ge98_s=float((speed>=98).sum()*.001),ge99_s=float((speed>=99).sum()*.001)))
    write_json(stage/'metrics.json',dict(shape=metrics,cruise=cruise,goal_complete=False))
    receipt(stage,dict(status='rendered_pending_visual_review',new_physics_ticks=0,new_decisions=0,
        model_sha256=sha(new/selected['model'])))


if __name__=='__main__':main()
