"""Concise evidence integration only after all paired endpoints and audits exist."""
from datetime import datetime,timezone
from pathlib import Path
import re
import shutil
import numpy as np
from evaluate_historical_dense import ROOT,BASE,sha,write
from audit_current_two_comparison import read


def main():
    sup=ROOT/'experimental_supplement_20260923'
    paired_path=sup/'common_two_action_expansion/completed_endpoint_selection.json'
    paired=read(paired_path)
    assert paired['all_three_seeds_complete_registered_paired_opportunities']
    accepted=read(sup/'three_selected_student_acceptance.json')
    evaluations=[ROOT/'direct14_dictionary02_resume_round73']+[Path(r['evaluation']) for r in accepted['records']]
    audits=[p/'current_two_action_comparison_v2.json' for p in evaluations]
    for path in audits:
        report=read(path)
        assert report['all_new_two_action_comparisons_pass']
        assert all(sha(p)==h for p,h in report['source_sha256'].items())
        assert np.allclose(report['dense_envelope_s'],[15.798,15.8755],rtol=0,atol=1e-8), 'Refresh all envelope figures/text before integration'
    source=Path(r'D:/PPO_article_work/论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous/manuscript.tex')
    old=source.read_text(encoding='utf-8');new=old
    paragraph=r"\figref{fig:efficiency} summarizes complete time at both resolutions for the selected controllers. The paired executions check numerical sensitivity of the same models. The controllers share a PPO-derived initialization and undergo separate fitting and checkpoint selection, with different development budgets (Appendix~\ref{app:learning}). These comparisons establish controller-level efficiency gains under common physical conditions; equal-budget training comparisons are needed to quantify the effect of preview availability alone. Independent complete-pipeline repetitions have not yet been established."
    replacement=r"\figref{fig:efficiency} summarizes complete time for the selected controllers. Their common physical conditions permit a controller-level efficiency comparison; their separate fitting and checkpoint selection use different development budgets (\ref{app:learning}). To examine preview availability under matched refinement budgets, a separate comparison starts each two-/three-action pair from the same steering and feed policy, gives both arms 48 PPO updates, and applies the same complete-success checkpoint rule."
    assert new.count(paragraph)==1
    new=new.replace(paragraph,replacement)
    start=new.index(r'\begin{table}[htbp]',new.index('Three independently initialized student models'))
    end=new.index(r'\end{table}',start)+len(r'\end{table}')
    table=r'''\begin{table}[htbp]
\centering
\caption{Selected student performance and matched paired refinement. All displayed models complete both cases at both integration intervals. Panel (a) reports the three selected main-method students; panel (b) reports the separate two-/three-action pairs after equal PPO opportunities and a common checkpoint-selection rule.}
\label{tab:selected_students}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrrr}
\toprule
\multicolumn{5}{l}{(a) Selected main-method students}\\
Seed & \multicolumn{2}{c}{Long route (s)} & \multicolumn{2}{c}{Dense case (s)}\\
 & 1 ms & 0.5 ms & 1 ms & 0.5 ms\\
\midrule
'''
    for row in accepted['records']:
        a,b=row['original_times_s'];c,d=row['dense_times_s']
        table+=f"{row['seed']} & {a:.3f} & {b:.4f} & {c:.3f} & {d:.4f}\\\\\n"
    table+=r'''\midrule
\multicolumn{5}{l}{(b) Matched paired refinement: dense-case time}\\
Seed & \multicolumn{2}{c}{Two-action (s)} & \multicolumn{2}{c}{Three-action (s)}\\
 & 1 ms & 0.5 ms & 1 ms & 0.5 ms\\
\midrule
'''
    changes=[]
    for row in paired['paired_comparisons']:
        a,b=row['two_times_s'][2:];c,d=row['three_times_s'][2:]
        table+=f"{row['seed']} & {a:.3f} & {b:.4f} & {c:.3f} & {d:.4f}\\\\\n"
        changes+=row['gain_percent'][2:]
    table+='\\bottomrule\n\\end{tabular}\n\\end{table}'
    new=new[:start]+table+new[end:]
    if not paired['all_pairs_show_credible_dense_gain']:
        statement=(f"\n\nThe matched paired refinement gives dense-case time reductions ranging from {min(changes):.2f}\\% to {max(changes):.2f}\\% across seeds and integration intervals (negative values indicate an increase). These smaller, seed-dependent differences do not consistently exceed the numerical margin. The gains above 5\\% in the developed-controller comparison therefore characterize the complete learning and refinement pipeline, rather than the marginal effect of enabling preview in this paired PPO stage.\n\n")
    else:
        raise AssertionError('Reassess wording against the newly stronger paired evidence')
    marker='\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.98\\linewidth]{Figure_9.pdf}'
    assert new.count(marker)==1;new=new.replace(marker,statement+marker)
    new=new.replace('Independent complete-pipeline repetitions established & 0 & 0\\\\\n','')
    new=new.replace('Appendix~\\ref{app:learning}',r'\ref{app:learning}')
    append=r'''
The supplementary students use independent random initializations with shared teacher data and feature design. Their selected refinements comprise additional PPO and fitting for seed 923201, offline fitting for seed 923202, and local static-feature fitting with per-center support radii for seed 923203. The separate paired study expands a successful two-action initializer by adding a constant preview output with zero initial correction; corresponding steering/feed parameters and initial executions agree. Seed 923202 uses the same per-center support widths in both arms. Each arm receives 48 PPO updates of 2048 decisions, split equally between the two physical intervals, with 75\% dense-case decisions, KL limit 0.01 and auxiliary weight 0.25. Checkpoints 0, 24 and 48 receive complete evaluation. Among successful endpoints, selection minimizes the worst S2-normalized time over both cases and both intervals, then the mean and the earlier endpoint. Initialization and offline preparation differ between seeds; only the subsequent paired PPO budgets and checkpoint opportunities are matched.

'''
    marker=r'\begin{table}[H]'+'\n'+r'\centering'+'\n'+r'\caption{Training provenance of the reported models.'
    assert new.count(marker)==1;new=new.replace(marker,append+marker)
    method=lambda t:t.split(r'\section{Methodology}',1)[1].split(r'\section{Experiments and Results}',1)[0]
    assert method(old)==method(new)
    for pattern in [r'\\(?:sub)?section\{[^}]+\}',r'\\cite\{[^}]+\}',r'\\includegraphics[^\n]+']:
        assert re.findall(pattern,old)==re.findall(pattern,new)
    history=ROOT/'lowdim_original_manuscript_work/editorial_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history.mkdir();shutil.copy2(source,history/'manuscript_before_completed_pairing.tex')
    source.write_text(new,encoding='utf-8',newline='\n')
    write(history/'editorial_change_receipt.json',dict(before_sha256=sha(history/'manuscript_before_completed_pairing.tex'),after_sha256=sha(source),
        paired_selection=str(paired_path),paired_selection_sha256=sha(paired_path),current_audits={str(p):sha(p) for p in audits},
        changes='Two resulttablesinbody;combined selected-model/pairedtable;one concise attribution paragraph;matching parameters inappendix;redundant zero-repeat rowremoved;doubleAppendixprefixfixed.',
        method_and_first3figures_unchanged=True,final_manuscript_accepted=False))
    print('COMPLETED_PAIRING_INTEGRATED',history)


if __name__=='__main__':main()
