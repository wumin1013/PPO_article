"""Draft Figure9 from accepted selected models; no inference about uniform repeats."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluate_historical_dense import ROOT, BASE, write, sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-audit', type=Path,
        default=ROOT/'supplement_margin_main_923201_ppo24/evaluation/current_two_action_comparison.json')
    args = parser.parse_args()
    delivery = BASE/'paper_evidence_review/verified_results/lowdim_original_revision'
    binding_path = delivery/'paper_result_binding.json'
    acceptance_path = ROOT/'experimental_supplement_20260923/three_selected_student_acceptance.json'
    binding, acceptance, audit = map(read, [binding_path, acceptance_path, args.baseline_audit])
    for item in [binding, acceptance, audit]:
        assert all(sha(p) == h for p, h in item['source_sha256'].items())
    assert acceptance['all_three_selected_checkpoints_pass']
    envelope = np.array(audit['dense_envelope_s'])
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    write(output/'figure_contract.json', dict(
        conclusion='The frozen controller reduces complete time; all three selected independently initialized students exceed5percent dense gains under the current envelope.',
        panel_logic={'a':'Complete original lap, selected controllers', 'b':'Complete dense case, selected controllers',
            'c':'Dense gains of frozen model and three accepted students at both steps'},
        relationship='Paired quantitative comparison; numerical steps and student histories explicitly separated.',
        archetype='quantitative grid', backend='Python matplotlib',
        visual_precedent='Retain the author-directed existing Figure9 paired-point grammar and palette; no external journal precedent is claimed.',
        scope='Working draft pending completed paired-action supplement and final envelope reconciliation; shared teacher data/maps and differing later refinement. No error bars or uniform-training success-rate claim.',
        export='184.15mm wide vectorPDF/SVG and PNG preview; editable text',
        sources={str(p):sha(p) for p in [binding_path, acceptance_path, args.baseline_audit, Path(__file__)]}))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none',
        'legend.frameon':False})
    colors={'S2':'#D78021','Two-action':'#82748C','J-NNC':'#186C9A'}
    fig=plt.figure(figsize=(7.25,5.4))
    grid=fig.add_gridspec(2,2,height_ratios=[1.,1.12],hspace=.72,wspace=.57)
    records=[]
    for col,case in enumerate(['original','dense']):
        ax=fig.add_subplot(grid[0,col])
        for row,name in enumerate(['S2','Two-action','J-NNC']):
            values=[v['complete_time_s'] for v in binding['methods'][name][case]]
            color=colors[name]
            for j,(v,offset) in enumerate(zip(values,[-.08,.08])):
                ax.plot(v,row+offset,marker='o' if j==0 else 's',mfc=color if j==0 else 'white',mec=color,ms=4)
                records.append(dict(panel=chr(97+col),model=name,case=case,dt_ms=1 if j==0 else .5,complete_time_s=v,gain_percent=''))
            ax.text(max(values)+(.027 if case=='original' else .10),row,
                f'{values[0]:.3f} / {values[1]:.4f}',fontsize=6.5,va='center',color=color)
        ax.set(yticks=range(3),yticklabels=['S2','Two-action','J-NNC'],xlabel='Complete time (s)',ylim=(2.5,-.5))
        ax.set_xlim((444.05,446.0) if case=='original' else (14.25,23.7))
        ax.grid(axis='x',color='#E5E9EC',lw=.45)
        ax.set_title('(a) Long path' if case=='original' else '(b) Dense case',loc='left',pad=10)
    ax=fig.add_subplot(grid[1,:])
    students=[('Main model',[v['complete_time_s'] for v in binding['methods']['J-NNC']['dense']])]
    students += [(f"Seed {r['seed']}",r['dense_times_s']) for r in acceptance['records']]
    for row,(name,times) in enumerate(students):
        gains=100*(1-np.array(times)/envelope)
        for j,offset in enumerate([-.10,.10]):
            ax.plot(gains[j],row+offset,marker='o' if j==0 else 's',mfc=colors['J-NNC'] if j==0 else 'white',mec=colors['J-NNC'],ms=4.5)
            records.append(dict(panel='c',model=name,case='dense',dt_ms=1 if j==0 else .5,complete_time_s=times[j],gain_percent=float(gains[j])))
        ax.text(max(gains)+.04,row,f'{gains[0]:.2f}% / {gains[1]:.2f}%',fontsize=8,va='center')
    ax.axvline(5.,ls='--',color='#9CA5AC',lw=.8)
    ax.text(5.025,-.56,'5% criterion',fontsize=7,color='#566773')
    ax.set(xlim=(4.85,6.9),ylim=(3.5,-.75),yticks=range(4),yticklabels=[r[0] for r in students],
        xlabel='Complete-time reduction vs. best valid two-action envelope (%)')
    ax.grid(axis='x',color='#E5E9EC',lw=.45)
    ax.set_title('(c) Dense-case efficiency across selected student histories',loc='left',pad=10)
    handles=[Line2D([],[],color='#334D60',marker='o',ls='none',ms=4,label='1 ms'),
        Line2D([],[],color='#334D60',marker='s',mfc='white',ls='none',ms=4,label='0.5 ms')]
    fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.58,1.))
    fig.subplots_adjust(left=.18,right=.98,top=.89,bottom=.105)
    for ext in ['pdf','svg','png']:
        fig.savefig(output/f'Figure_9.{ext}',dpi=250)
    plt.close(fig)
    with (output/'Figure_9_source_data.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    caption='Complete efficiency and selected student histories. (a,b) Complete times of the selected controllers. (c) Dense-case reductions relative to the fastest valid two-action time at each resolution, for the frozen main model and three independently initialized students. Students share teacher data and static feature design, with differing subsequent refinement; they are not three repetitions of a uniform training procedure. Filled circles and open squares show two numerical integration intervals, not additional training seeds. All displayed executions complete the corresponding task.'
    (output/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    write(output/'figure_data_binding.json',dict(sources={str(p):sha(p) for p in [binding_path,acceptance_path,args.baseline_audit,Path(__file__)]},
        dense_envelope_s=envelope.tolist(),selected_student_count=3,uniform_fixed_recipe_repetitions=False,
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},visual_qa_complete=False,final_manuscript_accepted=False))
    print('STUDENT_EFFICIENCY_DRAFT',output)


if __name__=='__main__':
    main()
