"""Render the audited complete development curve once; no policy evaluation."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from study_io import BASE, ROOT, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',choices=('v1','v2'),default='v1')
    args=parser.parse_args()
    audit=BASE/'v7_complete_curve_audit_v1'
    precedent=BASE/'v7_curve_visual_precedent_v1'
    converter_path=ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources=[Path(__file__),audit/'receipt.json',audit/'results.json',audit/'learning_curve.csv',
        precedent/'figure_contract.md',precedent/'core_fetch.json',precedent/'core2023_icml.html',
        BASE/'credit_baseline_fulltext_v1/tucker18a.pdf',converter_path]
    if not all(path.exists() for path in sources):
        raise RuntimeError('The complete v7 audit and visual-source verification must exist')
    data=json.loads((audit/'results.json').read_text(encoding='utf-8'))
    assert len(data['curves'])==28
    stage=register('v7_complete_curve_figure_'+args.version,sources,
        dict(saved_summary_reads=1,svg_conversions=1,models=0,physics_ticks=0),
        'Display all audited development nodes and selected checkpoints; no independent-seed uncertainty or final superiority claim.')
    mpl.rcParams.update({'font.family':'Arial','font.size':8,'svg.fonttype':'none',
        'axes.linewidth':.65,'mathtext.fontset':'stix','axes.spines.top':False,
        'axes.spines.right':False})
    fig=plt.figure(figsize=(7.1,6.3))
    grid=fig.add_gridspec(2,2,height_ratios=[1.25,1.],hspace=.46,wspace=.37)
    axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
    groups=[('Joint independent','joint_independent_refinement_v7','#245784','-','o'),
        ('Fixed3 independent','fixed3_independent_refinement_v7','#cd7825','-','s'),
        ('Joint gSDE8','joint_gsde_refinement_v7','#628cae','--','o'),
        ('Fixed3 gSDE8','fixed3_gsde_refinement_v7','#875324','--','s')]
    metrics=('full_time_s','line_time_s','corner150_distance_mm')
    for label,name,color,style,marker in groups:
        rows=sorted((r for r in data['curves'] if r['stage']==name),key=lambda r:r['decisions'])
        assert [r['decisions'] for r in rows]==[0,60000,120000,180000,240000,300000,359999]
        chosen=next(r for r in data['selected'] if r['stage']==name)
        x=np.array([r['decisions']/1000 for r in rows])
        for ax,key in zip(axes,metrics):
            y=np.array([r[key] if r[key] is not None else np.nan for r in rows])
            ax.plot(x,y,label=label,color=color,ls=style,lw=1.15,marker=marker,ms=3.5)
            if chosen[key] is not None:
                ax.plot(chosen['decisions']/1000,chosen[key],marker='o',ms=8.5,
                    mfc='none',mec=color,mew=.9,linestyle='none')
    axes[1].axhline(data['s2_eight_path_time_s'],color='#444444',ls=':',lw=1.,label='Conventional S2')
    axes[1].set_ylim(bottom=data['s2_eight_path_time_s']-.10)
    axes[1].text(.04,.08,f"S2: {data['s2_eight_path_time_s']:.3f} s",
        transform=axes[1].transAxes,ha='left',va='bottom',color='#444444',fontsize=7.5)
    labels=['Twelve-path time (s)','Eight-path time (s)','Distance from vertex (mm)']
    titles=['(a) Complete development motion','(b) Common paths with S2','(c) 150° corner']
    for ax,ylabel,title in zip(axes,labels,titles):
        ax.set(xlim=(-8,368),xticks=[0,60,120,180,240,300,360],
            xlabel='New training decisions (thousands)',ylabel=ylabel)
        ax.set_title(title,loc='left',fontsize=8.5,pad=7)
        ax.grid(color='#e7eaed',lw=.35)
        ax.tick_params(labelsize=7)
    all_complete=all(r['successes']==12 for r in data['curves'])
    axes[0].text(.015,.035,'12/12 complete at every node' if all_complete else 'Incomplete-node total times omitted',
        transform=axes[0].transAxes,fontsize=7.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,.995),ncol=2,
        frameon=False,fontsize=8)
    fig.subplots_adjust(left=.1,right=.98,top=.87,bottom=.1)
    svg=stage/'v7_complete_curve.svg'
    fig.savefig(svg)
    fig.savefig(stage/'v7_complete_curve.png',dpi=180)
    plt.close(fig)
    shutil.copy2(audit/'learning_curve.csv',stage/'source_data.csv')
    spec=importlib.util.spec_from_file_location('v7_curve_svg_converter',converter_path)
    converter=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    external=Path('D:/PPO_article_joint_runs')/stage.name
    converter.convert(svg,external)
    pdf=stage/'v7_complete_curve.pdf'
    shutil.copy2(external/pdf.name,pdf)
    caption=('Development learning curves after a common pretrained initialization. All registered checkpoints, '
        'including the initial policy, are retained. The actual last saved decision is 359,999. '
        'Circles around markers identify the checkpoint selected by complete-success count and then total time. '
        'Panel (a) uses twelve development paths; panel (b) uses the same eight paths available for conventional S2; '
        'panel (c) reports distance from the 150-degree vertex, which alone does not measure inward-transition quality. '
        'The initial shape value was not recomputed. Each method has one development run from the same pretraining '
        'lineage, so no independent-seed error bands are shown. These curves do not establish final adaptive-preview superiority.')
    write_json(stage/'caption.json',dict(text=caption,pdf_sha256=sha(pdf),svg_sha256=sha(svg),
        preview=str(external/'preview.png'),actual_final_decisions=359999,
        full_pdf_visual_QA_pending=True,final_manuscript_figure=False))
    receipt(stage,dict(status='converted_audited_curve_pending_visual_review',figures=1,
        models=0,physics_ticks=0,training_decisions=0))
    print(external/'preview.png')


if __name__=='__main__':
    main()
