"""Draw both matched modes at the completed 60k node from saved commands."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from study_io import BASE,ROOT,register,receipt,write_json,sha
from summarize_conditioned_training import distance_to_commands


def main():
    analysis=BASE/'minimal_factors_60000_analysis_v1'
    converter_path=ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources=[Path(__file__),BASE/'minimal_corner_figure_contract.md',
        BASE/'v7_checkpoint_figure_contract.md',BASE/'v7_60000_corner_figure_v1/visual_qa.json',
        analysis/'receipt.json',analysis/'results.json',converter_path,
        BASE/'summarize_conditioned_training.py']
    for arm in ('base','state','reward'):
        for mode in ('joint','fixed'):
            folder=BASE/f'minimal_{arm}_{mode}_pilot_v1'
            sources += [folder/'receipt.json',folder/'final_validation/summary.json']
            sources += [folder/'final_validation'/f'{i}.npz' for i in range(4)]
    if not all(p.exists() for p in sources):
        raise RuntimeError('Complete the common 60k analysis before drawing')
    sources += [BASE/'minimal_factor_corner_figures_v1/failure.json']
    stage=register('minimal_factor_corner_figures_v2',sources,
        dict(saved_trace_files=24,svg_conversions=2,models=0,physics_ticks=0),
        'Compare all isolated corners for each execution mode, preserving real aspect, speeds and complete-time context.')
    data=json.loads((analysis/'results.json').read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('minimal_corner_converter',converter_path)
    converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    mpl.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','axes.linewidth':.65})
    factors=[('base','Original 72','#444444','-'),('state','Spatial 22','#245784','--'),
        ('reward','Unshaped 72','#cd7825','-.')]
    rows=[]
    outputs=[]
    for mode in ('joint','fixed'):
        fig,axes=plt.subplots(2,2,figsize=(7.1,7.5))
        traces={}
        for arm,_,_,_ in factors:
            for index in range(4):
                with np.load(BASE/f'minimal_{arm}_{mode}_pilot_v1/final_validation/{index}.npz') as d:
                    traces[arm,index]={key:d[key].copy() for key in ('reference','rows')}
        for index,(ax,angle) in enumerate(zip(axes.ravel(),(50,95,125,150))):
            reference=traces['base',index]['reference']
            localref=reference-reference[1]
            region=LineString(localref).buffer(1.25,resolution=192)
            x,y=region.exterior.xy
            ax.fill(x,y,color='#edf4fa',zorder=0)
            ax.plot(x,y,color='#8db2d2',lw=.65,ls='--')
            ax.plot(localref[:,0],localref[:,1],color='#888888',ls=':',lw=.9,label='Reference')
            ax.plot(0,0,'o',ms=3,color='#333333')
            for j,(arm,label,color,style) in enumerate(factors):
                trace=traces[arm,index]
                assert np.array_equal(trace['reference'],reference)
                xy=trace['rows'][:,:2]-reference[1]
                speed=np.linalg.norm(trace['rows'][:,2:4],axis=1)
                local=np.linalg.norm(xy,axis=1)<10
                group=next(r for r in data['groups'] if r['arm']==arm and r['mode']==mode)
                result=group['cases'][index]['result']
                row=dict(arm=arm,mode=mode,turn_deg=angle,success=result['success'],
                    completion_time_s=result['completion_time_s'],
                    vertex_distance_mm=distance_to_commands(np.zeros(2),np.vstack([localref[0],xy])),
                    local_median_speed_mm_s=float(np.median(speed[local])) if local.any() else None)
                rows.append(row)
                ax.plot(xy[:,0],xy[:,1],color=color,lw=1.15,ls=style,label=label)
                speedlabel='n/a' if row['local_median_speed_mm_s'] is None else f"{row['local_median_speed_mm_s']:.1f}"
                timelabel='prefix' if row['completion_time_s'] is None else f"{row['completion_time_s']:.3f} s"
                ax.text(.02,-.36-.10*j,f"{label}: {row['vertex_distance_mm']:.2f} / {speedlabel} / {timelabel}",
                    transform=ax.transAxes,color=color,fontsize=7.2)
            ax.set(xlim=(-10,10),ylim=(-3,12),xlabel='x from vertex (mm)',ylabel='y from vertex (mm)')
            ax.set_aspect('equal',adjustable='box')
            ax.set_title(f'({chr(97+index)}) {angle}° turn',loc='left',fontsize=9)
            ax.grid(color='#e7eaed',lw=.35)
            ax.text(.02,-.26,'Distance (mm) / local speed (mm/s) / full time',transform=ax.transAxes,fontsize=7)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=4,frameon=False,fontsize=8)
        fig.suptitle('Joint control' if mode=='joint' else 'Fixed preview: D = 3 mm',fontsize=10,y=.997)
        fig.subplots_adjust(left=.09,right=.985,top=.89,bottom=.17,hspace=.90,wspace=.28)
        svg=stage/f'{mode}_corners.svg'
        fig.savefig(svg);fig.savefig(stage/f'{mode}_corners.png',dpi=180);plt.close(fig)
        external=Path('D:/PPO_article_joint_runs')/stage.name/mode
        external.parent.mkdir(parents=True,exist_ok=True)
        converter.convert(svg,external)
        pdf=stage/f'{mode}_corners.pdf'
        shutil.copy2(external/pdf.name,pdf)
        outputs.append(dict(mode=mode,pdf_sha256=sha(pdf),svg_sha256=sha(svg),preview=str(external/'preview.png')))
    with (stage/'source_data.csv').open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(stage/'figure_manifest.json',dict(outputs=outputs,full_pdf_visual_QA_pending=True,
        caption='All four isolated corners at the same 60000-decision development endpoint. Each factor is compared within its execution mode. Distances alone do not establish shape or efficiency superiority; local speeds and complete times are retained.',
        actual_evaluation_repeats=0,goal_complete=False))
    receipt(stage,dict(status='converted_two_mode_corner_figures_pending_visual_review',saved_trace_files=24,models=0,physics_ticks=0))
    print(json.dumps(outputs))


if __name__=='__main__':
    main()
