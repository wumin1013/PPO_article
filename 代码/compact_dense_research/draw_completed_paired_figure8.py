"""Keep the author's Figure8 geometry; show completed paired endpoints in panel b."""
from pathlib import Path
import importlib.util
import numpy as np
from evaluate_historical_dense import ROOT,BASE,write,sha
from fit_retained_basis_repair import read

source=ROOT/'draw_lowdim_manuscript_figures.py'
code=source.read_text(encoding='utf-8')
start=code.index('    counts = [8,16,24,32,40,48]')
end=code.index('    local_rows = {}',start)
code=code[:start]+'''    paired_path=ROOT/'experimental_supplement_20260923/common_two_action_expansion/completed_endpoint_selection.json'
    paired=read(paired_path)
    assert paired['all_three_seeds_complete_registered_paired_opportunities']
    ax.axvline(0.,color='#89959D',lw=.7)
    for i, row in enumerate(paired['paired_comparisons']):
        for j,(offset,marker) in enumerate([(-.10,'o'),(.10,'s')]):
            value=row['gain_percent'][2+j]
            ax.plot(value,i+offset,marker=marker,ms=4,mec=common.COLORS['Joint'],
                mfc=common.COLORS['Joint'] if j==0 else 'white',ls='none',label=('1 ms' if j==0 else '0.5 ms') if i==0 else None)
        ax.text(1.49,i,f"{row['gain_percent'][2]:.2f} / {row['gain_percent'][3]:.2f}",fontsize=6.7,va='center')
    ax.set(yticks=range(3),yticklabels=['923201','923202','923203'],ylim=(2.5,-.7),
        xlim=(-.30,2.15),xticks=[0,.5,1.,1.5,2.],xlabel='Dense-time reduction (%)',ylabel='Seed')
    ax.legend(fontsize=7,loc='upper right',bbox_to_anchor=(1.,1.25),ncol=2)
    ax.grid(axis='x',color='#E7EAED',lw=.4)
    common.panel(ax,'b','Matched PPO refinement')
''' +code[end:]
code=code.replace("checkpoint_curve='All six scheduled checks from48additional PPO updates;not independent seeds.'", "paired_panel='Three common-start pairs;equal48PPOupdates;fixed0/24/48selection;not the separately developed final-controller comparison.'")
ns={'__name__':'figure8_completed','__file__':str(source)}
exec(compile(code,str(source),'exec'),ns)
delivery=BASE/'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'
out=delivery/'paired_ablation';out.mkdir(exist_ok=False)
binding=read(delivery/'paper_result_binding.json')
paired_path=ROOT/'experimental_supplement_20260923/common_two_action_expansion/completed_endpoint_selection.json'
helper=ROOT/'lowdim_original_manuscript_work/selected923202/lowdim_figure_helpers.py'
spec=importlib.util.spec_from_file_location('figure8_helpers',helper)
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
import sys
sys.path.insert(0,str(ROOT.parent))
import draw_all34_publication_figures as previous
for mod in [common,previous,previous.common]:
    mod.NAMES['Fixed']='Two-action';mod.COLORS['Fixed']='#82748C';mod.NAMES['Joint']='J-NNC'
ref=np.loadtxt(binding['cases']['dense']['reference'],delimiter=',',skiprows=1)
geometry=read(binding['cases']['dense']['geometry'])
data={name:previous.load_trace(binding['methods'][name]['dense'][0]['trace'],ref) for name in ['J-NNC','Two-action']}
write(out/'figure_contract.json',dict(conclusion='Developed controllers differ in local trajectory;equal-opportunity PPO intervention has small mixed gains.',
    panels='a densecase,b matched refinement,c/d developed-controller local shapes;roles explicitly separate',
    backend='Python matplotlib',layout='Retain author-prescribed existing Figure8 layout and local scales;no external figure precedent claimed',
    evidence={str(p):sha(p) for p in [Path(__file__),source,helper,delivery/'paper_result_binding.json',paired_path]}))
ns['figure8'](binding,ref,geometry,data,out,common)
write(out/'paired_panel_data.json',read(paired_path)['paired_comparisons'])
print('PAIRED_FIGURE8',out)
