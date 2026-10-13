"""Plot the four saved complete corner transitions, without new rollouts."""
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


def main():
    groups=[('J-NNC','joint_remaining_credit_training_v5','#245784'),
            ('Fixed D = 3 mm','fixed3_remaining_credit_training_v5','#cd7825'),
            ('Fixed D = 4.6 mm','fixed4p6_remaining_credit_training_v5','#38815d')]
    converter_path=ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources=[Path(__file__),converter_path,BASE/'remaining_credit_analysis_v5/results.json']
    for _,name,_ in groups:
        sources += [BASE/name/'final_validation'/f'{i}.npz' for i in range(4)]
    stage=register('v5_corner_comparison_figure_v1',sources,
        dict(saved_trace_files=12,svg_conversions=1,model_calls=0,physics_ticks=0),
        'Show all four completed v5 isolated-corner transitions; keep unfavourable results and do not imply final superiority.')
    results=json.loads((BASE/'remaining_credit_analysis_v5/results.json').read_text(encoding='utf-8'))['groups']
    mpl.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none',
        'axes.linewidth':.65,'mathtext.fontset':'stix'})
    fig,axes=plt.subplots(2,2,figsize=(7.1,7.0))
    for i,(ax,angle) in enumerate(zip(axes.ravel(),[50,95,125,150])):
        with np.load(BASE/groups[0][1]/'final_validation'/f'{i}.npz') as data:
            reference=data['reference']-data['reference'][1]
        region=LineString(reference).buffer(1.25,resolution=192)
        x,y=region.exterior.xy
        ax.fill(x,y,color='#edf4fa',zorder=0)
        ax.plot(x,y,color='#8db2d2',lw=.65,ls='--',zorder=1)
        ax.plot(reference[:,0],reference[:,1],color='#62676b',ls=':',lw=.9,label='Reference',zorder=2)
        ax.plot(0,0,'o',ms=3,color='#333333',zorder=4)
        values=[]
        for j,(label,name,color) in enumerate(groups):
            with np.load(BASE/name/'final_validation'/f'{i}.npz') as data:
                xy=data['rows'][:,:2]-data['reference'][1]
            ax.plot(xy[:,0],xy[:,1],color=color,lw=1.25,label=label,zorder=3)
            t=results[j]['final_traces'][i]
            values.append(f"{t['nearest_corner_mm']:.2f} / {t['corner_speed_median']:.1f}")
        ax.set(xlim=(-10,10),ylim=(-3,12),xlabel='x from vertex (mm)',ylabel='y from vertex (mm)')
        ax.set_aspect('equal',adjustable='box')
        ax.set_title(f'({chr(97+i)}) {angle}° turn',loc='left',fontsize=9)
        ax.grid(color='#e7eaed',lw=.35)
        ax.text(.02,-.29,'Distance from vertex (mm) / local median speed (mm/s)',
            transform=ax.transAxes,fontsize=6.8)
        for j,((_,_,color),value) in enumerate(zip(groups,values)):
            ax.text(.02+j*.33,-.39,value,transform=ax.transAxes,color=color,fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1),ncol=4,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,top=.94,bottom=.12,hspace=.55,wspace=.28)
    svg=stage/'v5_corner_comparison.svg'
    fig.savefig(svg)
    plt.close(fig)
    spec=importlib.util.spec_from_file_location('v5_corner_svg_converter',converter_path)
    converter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    external=Path('D:/PPO_article_joint_runs/v5_corner_comparison_figure_v1')
    converter.convert(svg,external)
    shutil.copy2(external/'v5_corner_comparison.pdf',stage/'v5_corner_comparison.pdf')
    write_json(stage/'caption.json',dict(text='Saved deterministic final transitions after the v5 development training. All twelve development paths were completed by each method. Values below each panel follow legend order and show minimum distance to the complete command segments and median executed speed within 10 mm of the vertex. The tolerance is the Euclidean 1.25 mm neighbourhood of the finite reference. This is a development diagnosis, not a final comparison or a claim of maximum tolerance utilization.',
        svg_sha256=sha(svg),pdf_sha256=sha(stage/'v5_corner_comparison.pdf'),
        preview=str(external/'preview.png'),full_pdf_visual_QA_pending=True))
    receipt(stage,dict(status='saved_corner_figure_converted_pending_visual_QA',figures=1,model_calls=0,physics_ticks=0))
    print(external/'preview.png')


if __name__=='__main__':
    main()
