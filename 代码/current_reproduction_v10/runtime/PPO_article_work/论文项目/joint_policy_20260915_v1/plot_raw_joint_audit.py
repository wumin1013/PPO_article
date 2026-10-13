"""Raw XY, heading, physical curvature and speed for unadopted training actors."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_io import BASE
from search_complete_return import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--paths',nargs='+',type=int,default=[105,204])
    args=p.parse_args();stage=BASE/args.stage
    records={r['path']:r for r in read(stage/'shadow_training_result.json')['records']}
    baseline={r['path']:r for r in read(stage/'baseline.json')['baseline']['records']}
    fig,axes=plt.subplots(len(args.paths),4,figsize=(15,3.5*len(args.paths)),squeeze=False,layout='constrained')
    for row,path in enumerate(args.paths):
        for label,record,color in [('Protected research',baseline[path],'#718096'),('Training candidate',records[path],'#be3f4d')]:
            with np.load(record['trace']) as z:ref,rows=z['reference'],z['rows']
            u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            sign=np.sign(u[0]*v[1]-u[1]*v[0]);normal=sign*np.array([-u[1],u[0]])
            basis=np.stack([u,normal],axis=1);xy=(rows[:,:2]-ref[1])@basis
            speed=np.linalg.norm(rows[:,2:4],axis=1);moving=speed>5.
            heading=np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u))
            curvature=np.full(len(rows),np.nan)
            curvature[moving]=sign*(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
            t=(np.arange(len(rows))+1)*.001;heading[~moving]=np.nan
            axes[row,0].plot(xy[:,0],xy[:,1],color=color,label=label,lw=1.4)
            axes[row,1].plot(t,np.degrees(heading),color=color,lw=1.)
            axes[row,2].plot(t,curvature,color=color,lw=1.)
            axes[row,3].plot(t,speed,color=color,lw=1.)
        ref_xy=(ref-ref[1])@basis
        axes[row,0].plot(ref_xy[:,0],ref_xy[:,1],'k--',lw=.8)
        axes[row,0].set(xlim=(-9,1),ylim=(-.5,5),xlabel='Incoming tangent x (mm)',ylabel=f'Path {path}: inward y (mm)')
        axes[row,0].set_aspect('equal',adjustable='box');axes[row,0].legend(fontsize=8)
        angle=np.degrees(np.arccos(np.clip(u@v,-1.,1.)))
        axes[row,1].axhline(angle,color='k',ls='--',lw=.6)
        axes[row,1].set(xlim=(0,.8),xlabel='Time (s)',ylabel='Physical heading (deg)')
        axes[row,2].axhline(0,color='k',ls='--',lw=.6)
        axes[row,2].set(xlim=(.2,.65),xlabel='Time (s)',ylabel='Signed V/A curvature (1/mm)')
        axes[row,3].axhline(99,color='k',ls='--',lw=.6)
        axes[row,3].set(xlim=(0,1.2),xlabel='Time (s)',ylabel='Speed (mm/s)',ylim=(0,105))
        for ax in axes[row]:ax.grid(alpha=.2)
    fig.suptitle('Unadopted training actor — actual 1 ms traces, no position smoothing',fontsize=12)
    target=stage/'raw_xy_heading_curvature_speed.png';fig.savefig(target,dpi=150);plt.close(fig)
    print('RAW_GEOMETRY_FIGURE',target,flush=True)


if __name__=='__main__':main()
