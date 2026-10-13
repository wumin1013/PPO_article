"""Raw local XY audit of every registered corner, with no trajectory smoothing."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from study_io import sha,write_json
from natural_case_metrics import ordered_projection

ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('output');args=ap.parse_args()
stage=Path(args.output);stage.mkdir(exist_ok=False);candidate=Path(args.candidate)
z=np.load(candidate/'full_trace.npz');reference=z['reference'];rows=z['rows'];progress=ordered_projection(reference,rows[:,:2])
geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
m=json.loads((candidate/'full_metrics.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
for page in range(4):
    fig,axes=plt.subplots(3,3,figsize=(9,9),layout='constrained')
    for ax,c in zip(axes.flat,geometry['sharp_corners'][9*page:9*(page+1)]):
        k=c['reference_vertex_index'];u=reference[k]-reference[k-1];u/=np.linalg.norm(u)
        sign=np.sign(c['signed_heading_change_deg']);normal=sign*np.array([-u[1],u[0]])
        rot=np.stack([u,normal],axis=1);out=reference[k+1]-reference[k];out/=np.linalg.norm(out);out=out@rot
        ref=np.array([[-30,0],[0,0],30*out]);ax.plot(ref[:,0],ref[:,1],ls='--',color='.5',lw=.7)
        for a,b in zip(ref[:-1],ref[1:]):
            edge=b-a;n=np.array([-edge[1],edge[0]])/np.linalg.norm(edge)*1.25
            ax.add_patch(Polygon([a+n,b+n,b-n,a-n],color='#E7EEF3',zorder=0))
        ax.add_patch(Circle((0,0),1.25,color='#E7EEF3',zorder=0))
        keep=abs(progress-c['reference_s_mm'])<=30.;xy=(rows[keep,:2]-reference[k])@rot
        ax.plot(xy[:,0],xy[:,1],color='#246A9A',lw=1)
        cm=next(x for x in m['corners'] if x['id']==c['id'])
        ax.set(xlim=(-15,15),ylim=(-3,17),aspect='equal',xlabel='Incoming axis (mm)',ylabel='Inward normal (mm)',
            title=f"{c['id']} | included {c['interior_angle_deg']:.1f}°")
        ax.text(.03,.97,f"entry {cm['heading_entry_deg']:+.2f}° | exit {cm['heading_exit_error_deg']:+.2f}°\nreverse integral {cm['reverse_heading_deg']:.2f}°",transform=ax.transAxes,va='top',fontsize=7)
    count=len(geometry['sharp_corners'][9*page:9*(page+1)])
    for ax in list(axes.flat)[count:]:ax.axis('off')
    fig.suptitle('All-corner development audit | raw 1 ms XY | identical local scales',fontsize=11)
    fig.savefig(stage/f'corners_{page+1}.png',dpi=160);fig.savefig(stage/f'corners_{page+1}.pdf');plt.close(fig)
write_json(stage/'source_binding.json',dict(candidate=str(candidate),trace_sha256=sha(candidate/'full_trace.npz'),
    metrics_sha256=sha(candidate/'full_metrics.json'),all34_included=True,raw_data_not_smoothed=True,
    display='Common incoming/inward coordinates; right turns reflected for shape comparison only; all numerical audits use original XY.',visual_qa_complete=False))
print(stage)
