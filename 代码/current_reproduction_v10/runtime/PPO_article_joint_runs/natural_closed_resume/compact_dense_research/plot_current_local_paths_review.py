"""Three actual local trajectory comparisons, bound to successful full runs."""
import os
os.environ['MPLBACKEND']='Agg'
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon,Patch
from evaluate_historical_dense import ROOT,BASE,write,sha
from natural_case_metrics import ordered_projection


def main():
    stage=ROOT/'direct14_actual_arrival_fit_round55/mechanism_0001/local_paths_review'
    stage.mkdir(exist_ok=False);shutil.copy2(__file__,stage/Path(__file__).name)
    case=BASE/'paper_evidence_review/natural_closed_case'
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    bindings={str(case/'geometry.json'):sha(case/'geometry.json'),str(case/'reference.csv'):sha(case/'reference.csv')}
    old=json.loads((case/'paper_result_binding.json').read_text(encoding='utf-8'))['primary_methods']['S2']
    specs=[('S2',BASE/old['trace'],BASE/old['result'],'#BE783F','--'),
        ('Two-action',ROOT/'direct14_capacity_two_action_selection_round60/eval_0000_original_1ms.npz',ROOT/'direct14_capacity_two_action_selection_round60/eval_0000_original_1ms.json','#82748C',':'),
        ('Joint policy',ROOT/'direct14_actual_arrival_fit_round55/eval_0001_original_1ms.npz',ROOT/'direct14_actual_arrival_fit_round55/eval_0001_original_1ms.json','#27628E','-')]
    assert sha(BASE/old['trace'])==old['trace_sha256']
    datasets=[]
    for name,trace,result,color,style in specs:
        res=json.loads(result.read_text(encoding='utf-8'));assert res['success']
        with np.load(trace) as d:rows=d['rows'];ref=d['reference']
        assert np.array_equal(ref,reference)
        if 'trace_sha256' in res:assert res['trace_sha256']==sha(trace)
        datasets.append((name,rows,ordered_projection(reference,rows[:,:2]),color,style))
        bindings[str(trace)]=sha(trace);bindings[str(result)]=sha(result)
    write(stage/'figure_contract.json',dict(conclusion='Successful policies use visibly different local transition shapes within the same Euclidean corridor.',
        relationship='Paired local spatial comparison',geometry='User-prescribed representative135/90/45degree columns;equal-scale local windows,not fullpathoverlays.',
        precedent='Layout follows explicit user-selected local comparison and fixed representative angles;no outside visual-precedent claim.',
        backend='Python matplotlib',exports=['pdf','svg','png'],statistics='One complete nominal execution per method;no errorbars or seed claim.',
        corridor='Exact union of width2.5mm straight segment strips and vertexdisks in these isolated line-line corners.',
        scope='Research review,not final manuscript figure. Two-action initializer is a learned historical PPO controller with zero added head,after further training selection opportunities.'))
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.8));fig.subplots_adjust(left=.075,right=.99,bottom=.20,top=.75,wspace=.24)
    source_rows={}
    for col,ident in enumerate(geometry['mechanism_columns']):
        corner=next(c for c in geometry['sharp_corners'] if c['id']==ident);k=corner['reference_vertex_index'];point=reference[k]
        t=reference[k]-reference[k-1];t/=np.linalg.norm(t);n=np.sign(corner['signed_heading_change_deg'])*np.array([-t[1],t[0]])
        frame=np.stack([t,n],axis=1);out=(reference[k+1]-point)@frame;out/=np.linalg.norm(out)
        line=np.array([[-40.,0.],[0.,0.],40*out]);ax=axes[col]
        for start,end in zip(line[:-1],line[1:]):
            d=end-start;d/=np.linalg.norm(d);normal=1.25*np.array([-d[1],d[0]])
            ax.add_patch(Polygon(np.array([start+normal,end+normal,end-normal,start-normal]),closed=True,facecolor='#EDEDED',edgecolor='none',zorder=0))
        ax.add_patch(Circle((0,0),1.25,facecolor='#EDEDED',edgecolor='none',zorder=0))
        ax.plot(line[:,0],line[:,1],color='#555555',linewidth=.7,label='Reference',zorder=1)
        for name,rows,s,color,style in datasets:
            mask=abs(s-corner['reference_s_mm'])<=35.
            xy=(rows[mask,:2]-point)@frame
            source_rows[ident+'_'+name.replace(' ','_')]=xy
            ax.plot(xy[:,0],xy[:,1],color=color,linestyle=style,linewidth=1.05 if name=='Joint policy' else .95,label=name)
        ax.set(aspect='equal',xlim=(-10,10),ylim=(-2,13),xticks=[-10,0,10],yticks=[0,5,10])
        ax.tick_params(width=.6,length=2.5,labelsize=7)
        ax.set_title(f'({chr(97+col)}) '+f'{corner["interior_angle_deg"]:.0f}'+r'$^\circ$',fontsize=9.5,fontweight='bold',pad=5)
        ax.set_xlabel(r'$x_{\mathrm{local}}$ (mm)',labelpad=3)
    axes[0].set_ylabel(r'$y_{\mathrm{local}}$ (mm)',labelpad=3)
    handles,labels=axes[0].get_legend_handles_labels();handles.append(Patch(facecolor='#EDEDED'));labels.append('Tolerance corridor')
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.98),ncol=3,columnspacing=1.5,handlelength=2.5,fontsize=7.5)
    paths=[]
    for ext in ['pdf','svg','png']:
        p=stage/f'local_paths_review.{ext}';fig.savefig(p,dpi=250,facecolor='white');paths.append(p)
    plt.close(fig);np.savez_compressed(stage/'local_source_rows.npz',**source_rows)
    write(stage/'data_binding.json',dict(source_sha256=bindings,script_sha256=sha(__file__),
        outputs={p.name:sha(p) for p in paths},local_data_sha256=sha(stage/'local_source_rows.npz'),
        visual_inspection_pending=True,submission_ready=False,manuscript_and_original_three_figures_unchanged=True))
    print('LOCAL_PATHS_REVIEW',stage,flush=True)


if __name__=='__main__':main()
