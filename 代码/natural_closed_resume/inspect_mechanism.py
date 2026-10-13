"""Exploratory raw execution-chain inspection, not a frozen manuscript figure."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from study_io import sha,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--joint',required=True);ap.add_argument('--fixed',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    target=Path(args.output);target.mkdir(exist_ok=False)
    case=BASE/'paper_evidence_review/natural_closed_case'
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    sources={'Joint':Path(args.joint),'Fixed':Path(args.fixed)}
    write_json(target/'mechanism_inspection_contract.json',dict(status='exploratory_not_final_manuscript',
        conclusion='Inspect whether learned requests correspond to inward, approximately tangent executed turns; numerical acceptance is a separate gate.',
        backend='Python',layout='User-prescribed135/90/45 columns and six synchronized signal rows',
        time_alignment='Each method independently aligned to the same registered corner gate crossing',
        raw_data_not_smoothed=True,N_path=1,N_backbone=1,independent_PPO_replications=0,
        sources={k:{'trace':str(v/'full_trace.npz'),'sha256':sha(v/'full_trace.npz')} for k,v in sources.items()}))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(6,3,figsize=(8.1,10.0),sharex=True,layout='constrained')
    labels=['Steering request (deg)','Feed request (mm/s)','Preview request (mm)','Actual heading (deg)','Curvature (1/mm)','Actual speed (mm/s)']
    colors={'Joint':'#246A9A','Fixed':'#43816B'}
    for name,path in sources.items():
        z=np.load(path/'full_trace.npz');rows=z['rows'];actions=z['actions'];t=(np.arange(len(rows))+1)*.001
        m=json.loads((path/'full_metrics.json').read_text(encoding='utf-8'))
        for col,cid in enumerate(geometry['mechanism_columns']):
            c=next(c for c in geometry['sharp_corners'] if c['id']==cid)
            metric=next(c for c in m['corners'] if c['id']==cid);tc=metric['crossing_time_s']
            keep=(t>=tc-.35)&(t<=tc+.35);rr=rows[keep];v=rr[:,2:4];a=rr[:,4:6];speed=np.linalg.norm(v,axis=1)
            k=c['reference_vertex_index'];u=z['reference'][k]-z['reference'][k-1];u/=np.linalg.norm(u)
            normal=np.array([-u[1],u[0]])*np.sign(c['signed_heading_change_deg'])
            heading=np.rad2deg(np.unwrap(np.arctan2(v@normal,v@u)))
            curvature=np.sign(c['signed_heading_change_deg'])*(v[:,0]*a[:,1]-v[:,1]*a[:,0])/np.maximum(speed,5.)**3
            for row,y in ((3,heading),(4,curvature),(5,speed)):
                axes[row,col].plot(t[keep]-tc,y,color=colors[name],label=name,lw=1.1,ls='-' if name=='Joint' else '--')
            ta=np.arange(len(actions))*.02;ka=(ta>=tc-.37)&(ta<=tc+.37)
            for row in range(3):
                y=actions[ka,row]*(180/np.pi if row==0 else 1.)
                axes[row,col].step(ta[ka]-tc,y,where='post',color=colors[name],lw=1.1,ls='-' if name=='Joint' else '--')
            axes[0,col].set_title(str(int(c['interior_angle_deg']))+'° included angle')
            if name=='Joint':
                axes[3,col].axhline(abs(c['signed_heading_change_deg']),color='.6',ls=':',lw=.7)
    for row in range(6):
        for col in range(3):
            axes[row,col].axvline(0,color='.75',lw=.7);axes[row,col].set_xlim(-.35,.35)
            axes[row,col].grid(axis='y',alpha=.12)
        axes[row,0].set_ylabel(labels[row])
    for ax in axes[-1]:ax.set_xlabel('Time from ordered gate crossing (s)')
    axes[3,0].legend(loc='upper left',fontsize=7)
    fig.suptitle('Development diagnostic: actual execution chain; method not frozen',fontsize=10)
    for suffix in ('png','pdf'):
        fig.savefig(target/('mechanism_inspection.'+suffix),dpi=180)
    plt.close(fig)
    print('Saved raw mechanism inspection.')


if __name__=='__main__':main()
