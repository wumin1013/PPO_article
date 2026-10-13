"""Integrate verified selected-controller results into the original-format source."""
from pathlib import Path
from datetime import datetime, timezone
import json,re,shutil,hashlib,difflib
ROOT=Path(__file__).resolve().parent
BASE=Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1')
SRC=Path('D:/PPO_article_work/论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous')
OUT=BASE/'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'
SUP=ROOT/'experimental_supplement_20260923'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    acceptance=read(SUP/'selected_main_923202_evidence/selected_acceptance_20260924.json')
    assert acceptance['core_joint_numeric_goal_achieved']
    assert sha(acceptance['checkpoint'])==acceptance['checkpoint_sha256']
    for c in ['original','dense']:
        d=read(SUP/f'original_nnc_current_cases/{c}/result.json')
        assert d['first_common_violation'] is not None and not d['complete_time_comparison_eligible']
    assert read(SUP/'common_two_action_expansion/completed_endpoint_selection.json')['all_three_seeds_complete_registered_paired_opportunities']
    origin=read(OUT.parent/'revision_origin.json')
    hist=ROOT/'lowdim_original_manuscript_work/editorial_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    hist.mkdir(parents=True)
    text=(SRC/'manuscript.tex').read_text(encoding='utf-8');before=text
    shutil.copy2(SRC/'manuscript.tex',hist/'manuscript.tex')
    shutil.copytree(SRC/'figures_lowdim_revision',hist/'figures_lowdim_revision')
    def replace(a,b):
        nonlocal text
        assert text.count(a)==1,(a,text.count(a))
        text=text.replace(a,b)
    def paragraph(prefix,new):
        nonlocal text
        match=re.search(re.escape(prefix)+r'[^\n]*',text)
        assert match,prefix
        text=text[:match.start()]+new+text[match.end():]
    paragraph('Computer numerical control (CNC) motion planning',r'''Computer numerical control (CNC) motion planning converts computer-aided manufacturing (CAM)-generated tool paths into executable trajectories under contour tolerance and kinematic limits. Conventional smooth-then-schedule pipelines fix the geometry before regulating feedrate, limiting the use of geometric freedom during cornering. We propose the Jerk-constrained Neural Numerical Controller (J-NNC), a closed-loop reinforcement learning framework that coordinates tolerance-band trajectory shaping and feedrate regulation. A learned policy selects steering, feedrate and Adaptive Preview, and a kinematic constraint module (KCM) maps their combined request to feasible planar motion. The controller uses 14 current inputs without policy-history memory, with PPO learning followed by offline refinement. On a 44.420514 m closed path, it completes ordered coverage, braking and physical rest in 444.177 s under planar limits of 100 mm/s, 2000 mm/s$^2$ and 62500 mm/s$^3$. Complete time is 1.008 s shorter than that of a two-step planner, and speed deficit is no greater at any of the 34 corners than for that planner or a trained two-action controller. On an 84-corner dense case, complete time falls by 25.65\% relative to the two-step planner and by 6.07\% relative to the fastest valid two-action result. Both advantages persist at a 0.5 ms execution interval. Local trajectories and synchronized action--motion records associate these gains with anticipatory turning, coordinated feed recovery and sustained near-limit cruising within a one-sided 1.25 mm Euclidean corridor.''')
    replace('This representation is evaluated as a working input set; dimensional minimality is not assumed. ','')
    paragraph('PPO~\\cite{Schulman2017PPO} updates use',r'''The policy is learned with PPO~\cite{Schulman2017PPO} and refined offline against feasible action targets and recorded trajectory anchors. The resulting controller is a fixed function of the current 14-dimensional observation. Network representation and learning parameters are given in \ref{app:learning}.''')
    paragraph('We evaluate J-NNC in simulation',r'''We evaluate J-NNC on a complex closed path and a corner-dense closed case. The experiments examine physical feasibility, complete efficiency and the coordination between trajectory shape and feedrate. Comparisons use direct learned control (NNC), a traditional two-step planner (S2) and a trained two-action controller. The same selected J-NNC model is used on both cases.''')
    target='The traditional two-step baseline, S2, smooths the reference geometry'
    where=text.index(target)
    text=text[:where]+r'''The NNC baseline retains the direct two-action network and execution scheme used in the original comparison, motivated by learned CNC smoothing~\cite{Li2020}. It uses its original 14-input observation and 1 ms decision period without KCM projection. The planar adaptation removes angular-derived feed caps while preserving the learned action mapping. This implemented baseline tests command feasibility; successful constrained methods are then compared for complete efficiency.

'''+text[where:]
    paragraph('The principal ablation removes',r'''The principal ablation removes the learned preview output, retaining steering and feedrate actions, the same 14 observation channels and the same physical executor. Corresponding static-feature capacity and separate fitting, parameter selection and PPO opportunities are retained. The selected two-action model completes both cases at both resolutions. We also compare with the fastest valid dense-case time among all evaluated two-action models, retaining valid dense runs from models unsuccessful on the long route. This gives a conservative comparison against the best time at each resolution.''')
    paragraph('\\tabref{tab:main_results} reports',r'''J-NNC completes both cases within the common geometric and planar kinematic limits (\tabref{tab:main_results}). On the long route, it takes 444.177 s at 1 ms and 444.195 s at 0.5 ms, saving 1.008 and 0.990 s relative to S2. The corresponding savings over the selected two-action model are 0.537 and 0.536 s. Both exceed the numerical margins of 0.100 s against S2 and 0.175 s against the two-action model. On the dense case, J-NNC takes 14.839 and 14.8785 s, compared with S2's 19.957 and 19.9565 s. The reductions are 25.65\% and 25.45\%.''')
    ta=text.index(r'\begin{table}',text.index(r'\subsection{Main Comparative Results}'))
    tb=text.index(r'\end{table}',ta)+len(r'\end{table}')
    text=text[:ta]+r'''\begin{table}[htbp]
\centering
\caption{Feasibility and complete execution. Successful methods satisfy ordered coverage, command-interval geometry, vector kinematic limits and terminal rest. Maximum distance is measured over complete command intervals.}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lllrrrr}
\toprule
Case & Method & Feasible & \multicolumn{2}{c}{Complete time (s)} & \multicolumn{2}{c}{Max. distance (mm)}\\
 & & & 1 ms & 0.5 ms & 1 ms & 0.5 ms\\
\midrule
Long & NNC & No & --- & --- & --- & ---\\
route & S2 & Yes & 445.185 & 445.1850 & 1.243758 & 1.243758\\
 & Two-action & Yes & 444.714 & 444.7310 & 1.095198 & 1.173555\\
 & J-NNC & Yes & 444.177 & 444.1950 & 1.221862 & 1.140903\\
\midrule
Dense & NNC & No & --- & --- & --- & ---\\
case & S2 & Yes & 19.957 & 19.9565 & 1.149675 & 1.149675\\
 & Two-action & Yes & 15.875 & 15.8955 & 1.121741 & 1.163994\\
 & J-NNC & Yes & 14.839 & 14.8785 & 1.154901 & 1.249077\\
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.98\linewidth}\footnotesize
NNC is checked at its native 1 ms interval and fails the acceleration and jerk limits. Dashes denote absence of an admissible complete run; no failed-prefix time or partial-path maximum is entered as a complete result.
\end{minipage}
\end{table}'''+text[tb:]
    target='The complete vector records in '
    pos=text.index(target)
    text=text[:pos]+r'''The direct NNC commands exceed both acceleration and jerk limits from the first interpolation step on each case. In the retained first second, peak vector jerk reaches $5.261\times10^7$ mm/s$^3$ on the long route and $7.193\times10^7$ mm/s$^3$ on the dense case. Violations persist after the initial 20 ms. These partial records establish infeasibility under the common bounds, so they are excluded from complete-time rankings. J-NNC instead supplies feasible commands throughout both tasks. This comparison addresses the execution-layer requirement separately from efficiency among feasible planners.

'''+text[pos:]
    replace('The highest observed J-NNC corridor distance is 1.243948 mm on the nominal dense run','The highest observed J-NNC corridor distance is 1.249077 mm on the 0.5 ms dense run')
    replace("At C02, J-NNC's integrated speed deficit is 1.392 mm", "At C02, J-NNC's integrated speed deficit is 1.403 mm")
    replace('At C06, the corresponding values are 4.343, 13.321 and 5.963 mm.','At C06, the corresponding values are 4.253, 13.321 and 5.963 mm.')
    replace('to 13.965 mm for J-NNC.','to 13.902 mm for J-NNC.')
    replace('J-NNC accumulates 442.554 s','J-NNC accumulates 442.552 s')
    replace('interval is 111.737 s','interval is 111.753 s')
    paragraph('The stationary tail is included',r'''The stationary tail is included in $D_{\mathrm{all}}$. On the long route, J-NNC travels 44390.406 mm with total speed deficit 27.294 mm; S2 travels 44392.151 mm with deficit 126.349 mm. Of the 1.008 s saving, about 0.017 s comes from shorter travel and 0.991 s from reduced speed loss. Against the two-action model, the 0.537 s saving comprises 0.234 s from shorter travel and 0.303 s from lower speed loss. The percentage improvement is limited on a 44.4 m route already dominated by 100 mm/s cruise. Repeated transitions make the relative benefit larger on the dense case.''')
    a=text.index(r'\subsection{Ablation Results}')
    b=text.index(r'\section{Conclusion}',a)
    text=text[:a]+r'''\subsection{Ablation Results}
The three-action controller achieves shorter complete motion than the trained two-action configuration under the same physical constraints. On the dense case, the selected two-action model takes 15.875 and 15.8955 s at the two intervals. J-NNC reduces these times by 6.53\% and 6.40\%, respectively. The fastest valid two-action times across all evaluated models are 15.798 and 15.8755 s. Even against these values, J-NNC saves 0.959 and 0.997 s (6.07\% and 6.28\%), exceeding the 0.585 s numerical margin.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{Figure_8.pdf}
\caption{Two-action comparison: (a) the dense closed reference and local comparison locations; (b) dense-time reduction for three-action versus two-action controllers after matched PPO refinement, shown separately for three paired seeds; (c,d) executed $90^\circ$ and $45^\circ$ transitions of the developed main and two-action controllers. Panel (b) concerns a separate common-start intervention; panels (c,d) use the selected controllers in \tabref{tab:main_results}.}
\label{fig:ablation_dense}
\end{figure}

The local views in \figref{fig:ablation_dense}(c,d) show how the controllers use the tolerance band differently. At 1 ms, J-NNC sustains 4.368 s of near-limit cruise on the dense route, compared with 0.114 s for the selected two-action model. Total speed deficit falls from 213.867 to 132.502 mm. The 1.036 s time saving comprises 0.222 s from shorter travel and 0.814 s from reduced speed loss. Thus the efficiency benefit involves both local geometry and feedrate use over the complete 84-corner route.

A separate comparison starts each two-/three-action pair from equivalent physical requests and gives both arms the same PPO budget and selection opportunities (\ref{app:learning}). The selected dense-time changes in \figref{fig:ablation_dense}(b) range from $-0.12\%$ to $1.25\%$ across three seeds and both resolutions; they do not exceed the numerical acceptance margins. The larger gains above characterize the developed controllers, whose learning and refinement histories differ.

Three independently initialized student models each meet the physical, complete-time and all-34-corner criteria on both cases and at both intervals (\tabref{tab:selected_students}). Their dense-case improvements over the fastest valid two-action results range from 5.94\% to 6.29\%. They share teacher data and feature design, with different subsequent refinement; these are three successful selected models rather than repeated runs of one uniform training procedure. Seed 923202 is the main model used throughout the figures and comparisons.

\begin{table}[htbp]
\centering
\caption{Complete times of three selected student models. All pass both cases, both physical intervals and the all-corner comparisons.}
\label{tab:selected_students}
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lrrrr}
\toprule
Student seed & \multicolumn{2}{c}{Long route (s)} & \multicolumn{2}{c}{Dense case (s)}\\
 & 1 ms & 0.5 ms & 1 ms & 0.5 ms\\
\midrule
923201 & 444.178 & 444.1955 & 14.859 & 14.8980\\
923202 (main) & 444.177 & 444.1950 & 14.839 & 14.8785\\
923203 & 444.178 & 444.1955 & 14.839 & 14.8775\\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.98\linewidth]{Figure_9.pdf}
\caption{Complete efficiency: (a) long-route time and (b) dense-case time for the selected feasible controllers; (c) dense-case gains of the three selected models against the fastest valid two-action result at each interval. Filled circles and open squares denote integration intervals. Seed 923202 is the main model.}
\label{fig:efficiency}
\end{figure}

Together, the comparisons address the two requirements of joint motion planning. KCM provides feasible planar execution, while the learned actions use the tolerance band to improve complete efficiency. Anticipatory turning and coordinated recovery reduce cumulative speed loss. The dense case shows the benefit when transitions dominate; the original long route confirms sustained cruise and the all-corner result over a full lap.

'''+text[b:]
    paragraph('This paper presented J-NNC',r'''This paper presented J-NNC, a reinforcement learning-based CNC motion-planning method for mixed CAM-generated paths under contour-tolerance and kinematic constraints. It treats geometric shaping and feedrate regulation as a closed-loop constrained decision process. A 14-dimensional current observation supports steering, feedrate and Adaptive Preview decisions without policy-history memory. The planar KCM converts their combined request into executable motion.''')
    paragraph('The same selected model completes',r'''The same selected model completes the 44.420514 m route and the 84-corner dense case, including ordered coverage, braking and physical rest. All 34 long-route corner deficits are no greater than those of S2 or the trained two-action controller, and complete time is 444.177 s. Dense-case time is 14.839 s, a 6.07\% reduction relative to the fastest valid two-action result and a 25.65\% reduction relative to S2. These benefits persist at 0.5 ms and exceed the numerical acceptance margins. Local trajectory and action--motion records connect the efficiency improvement with upstream turning, coordinated feed and preview, and sustained cruising.''')
    paragraph('Future work will address',r'''Future work will address servo lag, axis-level constraints and machining experiments, and evaluate repeated training under a common fixed procedure.''')
    paragraph('The verified lineage begins',r'''The selected main model uses 88 PPO updates, arranged in stages of 16, 24, 24 and 24 updates, with offline fitting between stages. The last PPO stage uses 2048 fresh decisions per batch, comprising 1536 long-route and 512 dense-case decisions. Its time reward is given in Eq.~\eqref{eq:reward_total}; earlier stages additionally used a speed-deficit term. The final controller is obtained by offline fitting with the static feature map held fixed.''')
    paragraph('Offline parameter selection first',r'''Final offline refinement uses 12000 Adam iterations with learning rate $2\times10^{-4}$, multiplied by 0.3 after iterations 4000 and 8000. Each minibatch contains 256 recorded inputs from each case. All actor network and output coefficients are trainable, with gradient norm limited to 1; feature centers and support radii remain fixed. The coordinate loss scales are $(0.015,0.15,0.3)$. Feasible action targets and trajectory anchors are shared training data. The fixed final checkpoint is evaluated on both complete cases; neither teacher queries nor trajectory histories are available to the deployed policy.''')
    # Replace verbose provenance table and duplicated history with the settings needed to interpret comparisons.
    a=text.index('The two-action comparison retains')
    b=text.index(r'\clearpage',a)
    text=text[:a]+r'''The developed two-action comparator retains the 14-input encoder and corresponding local-feature capacity, with 5634 actor coefficients. Its upstream PPO-derived branch is followed by action conversion, offline fitting and additional PPO. The selected continuation checkpoints contribute 8 and 24 updates from stages offering 30 and 48 updates. This comparison therefore evaluates final-controller performance under common physical conditions, with unequal development histories.

The matched refinement experiment uses three seeds (923201--923203). Each pair starts with the same steering and feed requests and constant preview in the three-action arm. Each arm receives 48 PPO updates comprising 98304 fresh decisions. Checkpoints 0, 24 and 48 receive the same full evaluations. Among checkpoints passing both cases and both integration intervals, selection minimizes the worst S2-normalized complete time, followed by mean normalized time and then earlier checkpoint. The actor uses coordinate standard deviations $(0.002,0.015,0.02)$, auxiliary weight 0.25 and the common time objective. The initial model is retained if later checkpoints do not improve the selection criterion.

The three selected main-model students share teacher data and static features. Seed 923201 includes additional PPO with a soft corridor-margin cost, seed 923202 uses the final offline fit above, and seed 923203 uses local feature refinement with per-center support radii. Their input and output dimensions remain unchanged. Both benchmark cases participate in development and model selection.

The NNC comparison preserves the original two-action checkpoint, normalized observation and direct execution at 1 ms. The scalar angular-derived feed caps are removed for the common planar task; no kinematic projection is added. Vector velocity, acceleration and jerk are reconstructed independently from the recorded XY commands, starting from rest. After the first violation, a fixed prefix of up to 1 s is retained for diagnosis, ending earlier if the original environment terminates. These records determine feasibility only, with no complete-time or full-path peak claim. This is an adapted implementation of direct learned control, rather than an exact reproduction of all settings in Li et al.~\cite{Li2020}.

'''+text[b:]
    text=text.replace('Appendix~\\ref{app:learning}',r'\ref{app:learning}')
    for stale in ['444.178 s','14.860 s','25.54','6.03\\%','final step redistributes','remains incomplete','not yet been established']:
        assert stale not in text,stale
    assert re.findall(r'\\section\{([^}]+)\}',before)==re.findall(r'\\section\{([^}]+)\}',text)
    assert re.findall(r'\\includegraphics[^\n]+',before)==re.findall(r'\\includegraphics[^\n]+',text)
    assert re.search(r'\\title\{[^\n]+',before).group()==re.search(r'\\title\{[^\n]+',text).group()
    (SRC/'manuscript.tex').write_text(text,encoding='utf-8',newline='\n')
    for index in range(4,10):
        folder=OUT/('paired_ablation' if index==8 else 'student_efficiency' if index==9 else 'figures_publication')
        shutil.copy2(folder/f'Figure_{index}.pdf',SRC/'figures_lowdim_revision'/f'Figure_{index}.pdf')
    shutil.copy2(OUT/'figures_publication/Figure_4_execution.pdf',SRC/'figures_lowdim_revision/Figure_4_execution.pdf')
    for i in range(1,4):assert sha(SRC/'figures_lowdim_revision'/f'Figure_{i}.pdf')==origin['protected_figures'][str(i)]
    (hist/'working_change.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='before.tex',tofile='after.tex')),encoding='utf-8')
    write(OUT/'english_integration_receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),history=str(hist),
        source=str(SRC/'manuscript.tex'),source_sha256=sha(SRC/'manuscript.tex'),main_model=acceptance['checkpoint'],
        bibliography_sha256=sha(SRC/'references.bib'),protected_figures=origin['protected_figures'],
        original_title_preserved=True,original_main_section_structure_preserved=True,main_figures=9,
        source_words=len(text.split()),nnc_comparison_scope='Common-feasibility check of preserved implemented controller; recorded failed prefixes, no complete-time ranking.',final_pdf_qa_pending=True))
    print('ENGLISH_INTEGRATED',len(text.split()),'words',hist)
if __name__=='__main__':main()
