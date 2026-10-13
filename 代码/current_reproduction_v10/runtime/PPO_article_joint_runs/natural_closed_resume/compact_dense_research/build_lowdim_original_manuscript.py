"""Build only from the approved anonymous source, retaining prior working files."""
import difflib
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / 'lowdim_original_manuscript_work'
BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
DELIVERY = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision'
ORIGINAL = Path(r'D:\Nutstore\1\基于强化学习的运动规划方法\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous')
SRC = Path(r'D:\PPO_article_work\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main_table(binding):
    text = r'''\begin{table}[htbp]
\centering
\caption{Complete successful execution at two physical intervals. Maximum Euclidean distance is evaluated over command intervals. All entries pass vector kinematic and stopping checks.}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}
\toprule
Case & Method & \multicolumn{2}{c}{Complete time (s)} & \multicolumn{2}{c}{Maximum distance (mm)}\\
 & & 1 ms & 0.5 ms & 1 ms & 0.5 ms\\
\midrule
'''
    for case, label in [('original','Long route'),('dense','Dense case')]:
        for i, name in enumerate(['S2','Two-action','J-NNC']):
            a,b=binding['methods'][name][case]
            text += f'{label if i==0 else ""} & {name} & {a["complete_time_s"]:.3f} & {b["complete_time_s"]:.4f} & {a["maximum_distance_mm"]:.6f} & {b["maximum_distance_mm"]:.6f}'+r'\\'+'\n'
        if case=='original': text+=r'\midrule'+'\n'
    return text+r'\bottomrule\end{tabular}\end{table}'

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata-dir',type=Path,default=DELIVERY)
    args=parser.parse_args()
    args.metadata_dir.mkdir(parents=True,exist_ok=True)
    binding=read(DELIVERY/'paper_result_binding.json')
    for p,h in binding['source_sha256'].items(): assert sha(p)==h, p
    assert sha(ORIGINAL/'manuscript.tex')=='7e396a937555d8b63a48427213a93198797540767be3ec0cd8fc04ff078f8880'
    assert sha(binding['original_pdf'])=='074724d783318f227720ce09fe07410eb67221c572d661962c4ff2d7db64caa7'
    base=(ORIGINAL/'manuscript.tex').read_text(encoding='utf-8')
    text=base[:base.index(r'\section{Methodology}')]
    text=text.replace(r'\graphicspath{{./figures/}}',r'\graphicspath{{./figures_lowdim_revision/}}')
    abstract_start=text.index('Experiments on square, circular, and butterfly-shaped paths')
    abstract_end=text.index(r'\end{abstract}',abstract_start)
    text=text[:abstract_start]+('The evaluated controller uses 14 current inputs and three actions, with PPO-derived parameters refined by offline fitting. On a 44.420514 m closed path, it completes ordered coverage, braking and physical rest in 444.178 s under the common planar limits of 100 mm/s, 2000 mm/s$^2$ and 62500 mm/s$^3$. It reduces complete time by 1.007 s relative to a two-step planner and has no greater speed deficit at any of the 34 corners than that planner or a trained two-action controller. On a fixed 84-corner dense case, it is 6.04\\% faster than the strongest valid two-action result and 25.54\\% faster than the two-step planner. These advantages persist at a 0.5 ms execution interval. Recorded trajectories and synchronized action--motion traces connect the gains to anticipatory turning, coordinated feed recovery and sustained near-limit cruising within a one-sided 1.25 mm Euclidean corridor.\n')+text[abstract_end:]
    text=text.replace('limited transition length~\\cite{Li2020}.','limited transition length~\\cite{Yan2023}.')
    text=text.replace('same CAM path~\\cite{Li2020,Zhao2013}.','same CAM path~\\cite{Yan2023,Zhao2013}.')
    start=text.index('In a closely related direction, Li et al.')
    end=text.index('\n\nThese one-step methods',start)
    text=text[:start]+r'In a closely related direction, Li et al. studied reinforcement learning for CNC trajectory smoothing~\cite{Li2020}. This direction motivates learned motion decisions that connect local trajectory shaping with feedrate regulation.'+text[end:]
    start=text.index('In NNC, the neural-network action')
    end=text.index('\n\nThis paper addresses',start)
    text=text[:start]+'For learned one-step planning, contour tracking and path advancement must be accompanied by explicit execution-side velocity, acceleration and jerk constraints. Steering, feedrate and the forward geometric range also interact during corner transitions. Their contribution should therefore be assessed from executed trajectories and complete traversal time under common contour-tolerance and kinematic limits.'+text[end:]
    text=text.replace('The policy is trained using PPO~\\cite{Schulman2017PPO} and outputs','The policy is trained using PPO~\\cite{Schulman2017PPO} with subsequent offline refinement and outputs')
    text=text.replace('tracking, kinematic, corner, and multi-scale forward-geometric information, and the reward accounts for path progress, contour error, direction consistency, and motion smoothness.','tracking, kinematic and multi-scale forward-geometric information, and the learning objective accounts for physical time and progress under contour and motion constraints.')
    text=text.replace('while satisfying velocity, acceleration, and jerk limits for both linear and angular motion.','while satisfying planar vector velocity, acceleration and jerk limits.')
    text=text.replace('Experiments on square, circular, and butterfly-shaped paths show that J-NNC completes the full path and removes the linear-jerk-limit violations observed in the NNC baseline and the no-KCM ablation.','Complete executions on a long closed route and a corner-dense case demonstrate reduced speed loss and shorter traversal time than a two-step planner and a trained two-action controller, with paired numerical-resolution checks.')
    text=text.replace(r'P_{\mathrm{raw}}={p_1,p_2,\dots,p_N}',r'P_{\mathrm{raw}}=\{p_1,p_2,\dots,p_N\}')
    text=text.replace(r'\Omega={x\mid d(x,P_{\mathrm{raw}})\leq \varepsilon/2}',r'\Omega=\{x\mid d(x,P_{\mathrm{raw}})\leq \varepsilon/2\}')
    text=text.replace('where $d(x,P_{\\mathrm{raw}})$ denotes the minimum distance from point $x$ to the reference path.','where $d(x,P_{\\mathrm{raw}})$ denotes the minimum Euclidean distance from point $x$ to the finite reference path, including its line and arc segments. Here $\\varepsilon$ denotes the full band width, so the one-sided tolerance is $\\varepsilon/2$.')
    text=text.replace('local geometric quantities are extracted, including the normal error, tangential direction, distance to the next corner, and turning angle of the next corner. These quantities are used in the state representation, reward weighting, and corner-mode identification.','tracking and forward-geometric quantities are extracted from the local reference and upcoming points. These quantities are used in the current observation. An ordered gate sequence preserves traversal order where different portions of the closed path are spatially close.')
    text=text[:text.index(r'\subsection{Constrained Action Decision Formulation}')]+(WORK/'formulation.tex').read_text(encoding='utf-8')
    methods=(WORK/'methods.tex').read_text(encoding='utf-8')
    experiments=(WORK/'experiments.tex').read_text(encoding='utf-8').replace('@@MAIN_TABLE@@',main_table(binding))
    conclusion=r'''
\section{Conclusion}
\label{sec:conclusion}
This paper presented J-NNC, a reinforcement learning-based CNC motion-planning method for mixed CAM-generated tool paths subject to contour-tolerance and kinematic constraints. The method treats geometric shaping and feedrate regulation as a closed-loop constrained decision process. A 14-dimensional current observation supports steering, feedrate and Adaptive Preview decisions without policy-history memory, and a vector KCM converts their combined request into executable motion. PPO-derived learning and subsequent offline refinement are distinguished in the method description.

The same selected model completes the 44.420514 m route and the 84-corner dense case, including ordered coverage, braking and physical rest. On the long route, all 34 registered corner deficits are no greater than those of S2 or the trained two-action controller, and the complete time is 444.178 s. On the dense case, complete time is 14.860 s, a 6.04\% reduction relative to the strongest valid two-action result and a 25.54\% reduction relative to S2. These benefits remain above the numerical acceptance margins at 0.5 ms. Local trajectory views and action--motion records associate the efficiency gains with upstream turning, context-dependent feed and preview, and reduced cumulative speed loss.

Future work will address servo lag, axis-level constraints and real machining experiments. Independent repetitions of the complete learning and refinement pipeline will complement the present selected-model evidence.

'''
    revised=text+methods+'\n'+experiments+conclusion+(WORK/'appendix.tex').read_text(encoding='utf-8')+r'''
\clearpage
\bibliographystyle{unsrt}
\bibliography{references}
\end{document}
'''
    assert '@@' not in revised
    assert r'\subsection{PPO' not in revised
    assert revised.count(r'\includegraphics')==10 and revised.count(r'\ContinuedFloat')==1
    assert re.search(r'\\title\{([^}]+)',revised).group(1)==re.search(r'\\title\{([^}]+)',base).group(1)
    assert re.findall(r'\\subsection\{([^}]+)\}',experiments)==['Experimental Settings','Main Comparative Results','Ablation Results']
    assert not re.search(r'generaliz|unseen|held.out',experiments,re.I)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history=ROOT/'lowdim_manuscript_history'/stamp
    history.mkdir(parents=True)
    for name in ['manuscript.tex','references.bib','manuscript.bbl','elsarticle.cls','unsrt.bst']:
        if (SRC/name).exists(): shutil.copy2(SRC/name,history/name)
    (SRC/'manuscript.tex').write_text(revised,encoding='utf-8',newline='\n')
    bibliography=(ORIGINAL/'references.bib').read_text(encoding='utf-8').replace('Ishizaki, Kazunori','Ishizaki, Kosuke')
    bibliography=bibliography.replace('  eprint        = {1707.06347},','  eprint        = {1707.06347},\n  note          = {arXiv:1707.06347},')
    (SRC/'references.bib').write_text(bibliography,encoding='utf-8',newline='\n')
    for name in ['elsarticle.cls','unsrt.bst']: shutil.copy2(ORIGINAL/name,SRC/name)
    figures=SRC/'figures_lowdim_revision'
    if figures.exists(): shutil.copytree(figures,history/'figures_lowdim_revision')
    figures.mkdir(exist_ok=True)
    for p in (DELIVERY/'figures_publication').glob('*.pdf'): shutil.copy2(p,figures/p.name)
    for i in range(1,4): assert sha(figures/f'Figure_{i}.pdf')==sha(ORIGINAL/'figures'/f'Figure_{i}.pdf')
    (args.metadata_dir/'original_to_revised.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),revised.splitlines(True),fromfile='approved_18page/manuscript.tex',tofile='lowdim_revision/manuscript.tex')),encoding='utf-8')
    write(args.metadata_dir/'revision_origin.json',dict(base_source=str(ORIGINAL/'manuscript.tex'),base_source_sha256=sha(ORIGINAL/'manuscript.tex'),base_pdf=binding['original_pdf'],base_pdf_sha256=sha(binding['original_pdf']),previous_source_history=str(history),source=str(SRC/'manuscript.tex'),source_sha256=sha(SRC/'manuscript.tex'),bibliography_sha256=sha(SRC/'references.bib'),protected_figures={str(i):sha(figures/f'Figure_{i}.pdf') for i in range(1,4)},title_unchanged=True,preamble_change='Graphics directory only',main_figure_count=9,figure4_continued=True,main_sections=re.findall(r'\\section\{([^}]+)\}',revised.split(r'\appendix')[0]),templates={p.name:sha(p) for p in WORK.glob('*.tex')},binding_sha256=sha(DELIVERY/'paper_result_binding.json'),final_manuscript_accepted=False))
    print('SOURCE_READY',SRC/'manuscript.tex',flush=True)

if __name__=='__main__': main()
