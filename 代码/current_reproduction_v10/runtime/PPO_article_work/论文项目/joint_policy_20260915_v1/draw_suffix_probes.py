"""Render all three directed approach probes using the accepted true-XY layout."""
import csv
import importlib.util
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from probe_coordinated_suffix import read,shape
from study_io import BASE,ROOT,register,receipt,write_json,sha


def main():
    prior=BASE/'coordinated_suffix_probe_20260916';inner=BASE/'inner_approach_amplitude_probe_20260916'
    baseline=BASE/'ar_conditional_joint_20260916/final_validation'
    converter_path=ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources=[Path(__file__),BASE/'v7_checkpoint_figure_contract.md',BASE/'probe_coordinated_suffix.py',
        prior/'receipt.json',prior/'results.json',inner/'receipt.json',inner/'results.json',converter_path]
    sources += [p for i in (1,2,3) for p in (baseline/f'{i}.npz',prior/f'path{i}_111.npz',inner/f'{i}.npz')]
    stage=register('coordinated_suffix_figures_20260916',sources,
        dict(saved_traces=9,svg_conversions=1,model_calls=0,physics_ticks=0),
        'Show actual inside/outside approach and full completion for all three finite intervention cases, not learned policy curves.')
    (stage/'figure_contract.md').write_text('# 后缀可达性图\n\n沿用用户已确认的真实XY等比例、1.25mm允差带和配对三线图型。问题是有限动作干预能否产生可行内侧过渡并减少完整时间；不把干预当作学得方法。全部95/125/150度保留。Python绘图，SVG实际转PDF、PNG及CSV；目视验收后另写QA。\n',encoding='utf-8')
    mpl.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none','axes.linewidth':.65})
    fig,axes=plt.subplots(2,2,figsize=(7.1,7.5));records=[]
    colors=['#444444','#245784','#cd7825'];styles=['-','--','-.'];labels=['Actor','Probe 2°','Probe 8°']
    for ax,i,angle in zip(axes.ravel(),(1,2,3),(95,125,150)):
        paths=[baseline/f'{i}.npz',prior/f'path{i}_111.npz',inner/f'{i}.npz']
        for j,p in enumerate(paths):
            with np.load(p) as d:ref,rows,actions=(d[k].copy() for k in ('reference','rows','actions'))
            if j==0:
                reference=ref.copy();localref=ref-ref[1]
                region=LineString(localref).buffer(1.25,resolution=192);x,y=region.exterior.xy
                ax.fill(x,y,color='#edf4fa',zorder=0);ax.plot(x,y,color='#8db2d2',lw=.65,ls='--')
                ax.plot(localref[:,0],localref[:,1],color='#888888',ls=':',lw=.9,label='Reference')
                ax.plot(0,0,'o',ms=3,color='#333333')
            assert np.array_equal(ref,reference)
            xy=rows[:,:2]-ref[1];ax.plot(xy[:,0],xy[:,1],color=colors[j],ls=styles[j],lw=1.15,label=labels[j])
            diag=shape(ref,rows,actions);distance=diag['corners'][0]['minimum_vertex_distance_mm']
            y=diag['approach_minus10_to_minus2_median_signed_y_mm'];total=len(rows)*.001
            records.append(dict(path=i,turn_deg=angle,method=labels[j],complete_time_s=total,
                approach_median_signed_y_mm=y,vertex_distance_mm=distance,source=str(p),sha256=sha(p)))
            ax.text(.02,-.36-.1*j,f'{labels[j]}: {y:+.2f} / {distance:.2f} / {total:.3f} s',
                    transform=ax.transAxes,color=colors[j],fontsize=7.2)
        ax.set(xlim=(-10,10),ylim=(-3,12),xlabel='x from vertex (mm)',ylabel='y from vertex (mm)')
        ax.set_aspect('equal',adjustable='box');ax.grid(color='#e7eaed',lw=.35)
        ax.set_title(f'({chr(96+i)}) {angle}° turn',loc='left',fontsize=9)
        ax.text(.02,-.26,'Approach y (mm) / distance (mm) / full time',transform=ax.transAxes,fontsize=7)
    ax=axes.ravel()[3];ax.axis('off')
    note=('Finite action interventions\n\n'
          'Actor: original conditional-exploration policy.\n'
          'Probes: +5 mm/s feed and +2 mm preview,\n'
          'with either +2° or +8° steering.\n\n'
          'Offsets last eight 20 ms decisions from the\n'
          'saved approach state; then the original actor\n'
          'runs until complete braking and rest.\n\n'
          'All displayed paths pass the same corridor,\n'
          'ordered coverage and planar V/A/J checks.\n\n'
          'Positive approach y is inside for these left turns.\n'
          'The 150° case still has a slightly negative\n'
          'approach median under the 8° probe.\n\n'
          'These probes demonstrate finite reachability.\n'
          'They are not learned behavior or an optimum.')
    ax.text(.0,.98,note,va='top',fontsize=8,linespacing=1.65,transform=ax.transAxes)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.97),ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Feasible approach changes under finite action probes',fontsize=10,y=.997)
    fig.subplots_adjust(left=.09,right=.985,top=.89,bottom=.17,hspace=.90,wspace=.28)
    svg=stage/'suffix_probes.svg';fig.savefig(svg);fig.savefig(stage/'suffix_probes.png',dpi=180);plt.close(fig)
    spec=importlib.util.spec_from_file_location('suffix_converter',converter_path);converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
    external=Path('D:/PPO_article_joint_runs')/stage.name;converter.convert(svg,external)
    shutil.copy2(external/'suffix_probes.pdf',stage/'suffix_probes.pdf')
    with (stage/'source_data.csv').open('w',encoding='utf-8',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(stage/'figure_manifest.json',dict(pdf_sha256=sha(stage/'suffix_probes.pdf'),
        svg_sha256=sha(svg),preview=str(external/'preview.png'),visual_qa_pending=True))
    receipt(stage,dict(status='suffix_probe_figure_pending_visual_review',saved_traces=9,model_calls=0,physics_ticks=0))
    print(str(external/'preview.png'))


if __name__=='__main__':main()
