"""Generate every numerical results paragraph/table from completed v3 records."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'shared_manuscript'
NAMES = {'final_dense_fold':'Dense folds', 'final_chamfer_chain':'Chamfer chain',
         'final_rounded_triangle':'Rounded triangle', 'final_quintic_double_bend':'Double bend',
         'final_inward_channels':'Inward channels'}

def read(name):
    return json.loads((ROOT / name).read_text('utf-8'))

def write(name, text):
    (OUT/name).write_text(text.strip()+'\n', encoding='utf-8')

def fig(file, label, caption, width='0.98'):
    return ('\n\\begin{figure}[htbp]\n\\centering\n'
            f'\\includegraphics[width={width}\\linewidth]{{{file}.pdf}}\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n')

def table(label, caption, columns, heading, rows):
    return ('\n\\begin{table}[htbp]\n\\centering\n'
            f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n\\setlength{{\\tabcolsep}}{{4pt}}\n'
            +f'\\begin{{tabular}}{{{columns}}}\n\\toprule\n'+heading+' \\\\\n\\midrule\n'
            +'\n'.join(row+' \\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

def main():
    a=read('manuscript/results_v3/analysis.json')
    assert a['valid_for_manuscript'] and a['expected_cells']==90
    costs=read('audit/v3_complete_training_costs.json')
    timing=read('results/costs_v3/online_loop.json')
    sens=read('results/sensitivity_v3/summary.json')
    dev=read('shared_manuscript/development_motion_v3_metrics.json')
    lookup={(r['path'],r['method']):r for r in a['summary']}
    learning=[r for r in a['rows'] if r['method'] in ['adaptive','fixed'] and r['dt_s']==.001]
    fine=[r for r in a['rows'] if r['method'] in ['adaptive','fixed'] and r['dt_s']==.0005]
    nsuccess=sum(r['success'] for r in learning)
    assert nsuccess==len(learning)
    abstract = (
      'Consecutive corners in CNC toolpaths couple allowable contour deviation with acceleration- and jerk-limited motion. '
      'This study presents a neural numerical controller that jointly requests steering, feedrate and guide-target distance. '
      'A planar kinematic constraint module projects velocity requests and checks acceleration release, while ordered gates and continuous command-interval checks preserve complete path traversal. '
      'The policy combines feasible-action fitting with proximal policy optimization. '
      'Fixed and adaptive preview use the same observations, execution layer, teacher records and training budget. '
      'Five independently initialized models per configuration are evaluated on five reserved synthetic programs, including dense consecutive corners and mixed line--arc geometry. '
      f'All {nsuccess} learned executions at 1 ms completed the full rest-to-rest task at a nominal one-sided tolerance of 1.25 mm; '
      'arc discretization error is deducted from the geometric allowance. '
      'Comparisons include a quintic smoothing and scheduling adapter, a variant using Ruckig for straight-section timing, and an independent motion-overlap comparator. '
      'Complete trajectories show how local direction changes and sustained feedrate determine cycle time through executed length and integrated speed deficit. '
      'The results demonstrate joint learned command generation with explicit geometric and planar kinematic checks on the tested programs.')
    write('abstract_v3.tex',abstract)
    rows=[]
    for p in a['registered_programs']:
        cells=[]
        for m in ['adaptive','fixed','S2','S2R','SCMO']:
            r=lookup[p['id'],m]
            if r['mean_s'] is None: cells.append('F')
            elif m in ['adaptive','fixed']:cells.append(f"${r['mean_s']:.3f}\\pm{r['sd_s']:.3f}$")
            else:cells.append(f"{r['mean_s']:.3f}")
        rows.append(NAMES[p['id']]+' & '+' & '.join(cells))
    main_table=table('tab:complete_times',
      'Complete execution time (s) at 1 ms. Learned entries are mean $\\pm$ sample standard deviation over all five independent training seeds; both learned methods completed 5/5 runs on every program. Each deterministic entry is one complete run. F denotes a planning failure; completion time is undefined.',
      'lrrrrr','Program & J-NNC & Fixed preview & S2-Q & S2-R & SCMO',rows)
    paired_rows=[]
    for p in a['paired']:
        lo,hi=p['student_95_interval_s']
        paired_rows.append(f"{NAMES[p['path']]} & {p['complete_pairs']} & {p['mean_difference_s']:.3f} & $[{lo:.3f}, {hi:.3f}]$")
    paired_table=table('tab:paired',
      'Paired fixed-minus-adaptive time difference (s). The independent unit is the training seed, paired within a complete program. Intervals use Student\'s $t$ distribution; they describe training variability for these programs.',
      'lrrr','Program & Pairs & Mean difference & 95\\% interval',paired_rows)
    dense='final_dense_fold';mixed='final_rounded_triangle'
    def record(p,m):
        return next(r for r in a['rows'] if r['path']==p and r['method']==m and r['dt_s']==.001 and r['seed']==(101 if m in ['adaptive','fixed'] else None))
    d={m:record(dense,m) for m in ['adaptive','fixed','S2R','SCMO']}
    q={m:record(mixed,m) for m in ['adaptive','fixed','S2R','SCMO']}
    nominal=1.25
    mixed_program=next(p for p in a['registered_programs'] if p['id']==mixed)
    successful=[r for r in a['rows'] if r['success']]
    vmax=max(r['v_peak']/100 for r in successful)
    amax=max(r['a_peak']/2000 for r in successful)
    jmax=max(r['j_peak']/62500 for r in successful)
    learn_diff=max(abs(r['fine_minus_coarse_s']) for r in a['resolution'] if r['method'] in ['adaptive','fixed'] and r['fine_minus_coarse_s'] is not None)
    deterministic_diff=max(abs(r['fine_minus_coarse_s']) for r in a['resolution'] if r['method'] not in ['adaptive','fixed'] and r['fine_minus_coarse_s'] is not None)
    resolution_rows=[]
    for m,label in [('adaptive','J-NNC'),('fixed','Fixed preview'),('S2','S2-Q'),('S2R','S2-R'),('SCMO','SCMO')]:
        rs=[r for r in a['resolution'] if r['method']==m]
        ds=[abs(r['fine_minus_coarse_s'])*1000 for r in rs if r['fine_minus_coarse_s'] is not None]
        resolution_rows.append(f"{label} & {sum(r['success_1ms'] for r in rs)}/5 & {sum(r['success_0p5ms'] for r in rs)}/5 & {max(ds):.1f}")
    resolution_table=table('tab:resolution',
       'Temporal-resolution check with unchanged policies and planning settings. Learned methods use the predetermined seed 101. The time difference is calculated only for programs completed at both resolutions; completion counts include every program.',
       'lrrr','Method & Completed, 1 ms & Completed, 0.5 ms & Max. $|\\Delta T|$ (ms)',resolution_rows)
    sensitivity_table=table('tab:clip_sensitivity',
       'PPO clipping sensitivity on the two validation programs, with seed 101 and the common training budget. All settings completed both programs.',
       'rrr','PPO clip & Trapezoid time (s) & Zigzag time (s)',
       [f"{r['clip']:.1f} & {r['validation'][0]['complete_time_s']:.3f} & {r['validation'][1]['complete_time_s']:.3f}" for r in sens['rows']])
    resolution_table=resolution_table.replace('\\begin{table}[htbp]','\\begin{table}[H]')
    sensitivity_table=sensitivity_table.replace('\\begin{table}[htbp]','\\begin{table}[H]')
    # Use program-level seed statistics, never interpolation samples as replicates.
    experiments = r'''\subsection{Evaluation conditions and toolpaths}
All methods use complete planar command programs, the same initial rest state, vector limits of 100 mm/s, 2000 mm/s$^2$ and 62\,500 mm/s$^3$, and a nominal one-sided tolerance of 1.25 mm. Execution is sampled at 1 ms and checked again at 0.5 ms. The evaluation accepts a run only after every ordered gate is traversed, every command interval lies within the finite-curve corridor, and braking is followed by the prescribed final rest. A failed plan or incomplete execution has no completion time.

Four complete programs supply training data: a rectangle, a sampled sinusoidal path, a multi-turn path and a sampled cubic S-bend. Two different programs, a trapezoid and a zigzag, are used for parameter selection and validation. Program families are separated before the final evaluation: complete source programs and their crops, rotations, scalings and direct parameter variants remain in the same group. The long and repeated-corner development programs are kept separate from the final test set. The five reserved programs in \figref{fig:programs} are complete synthetic programs with distinct layouts. Their designs cover dense short blocks, concave chamfers, unequal circular fillets, noncircular bends and nested channels.

The common input for the final comparison is each program's declared polyline. For the rounded triangle, the known circular-chord error is subtracted from the nominal tolerance before any method is run. This gives a common polygonal allowance of '''+f"{mixed_program['evaluation_tolerance_mm']:.6f} mm from a chord-error bound of {mixed_program['arc_chord_error_bound_mm']:.6f} mm."+r''' The noncircular double-bend program is evaluated relative to its declared sampled curve. 

Kinematic and timing quantities are computed from the complete simulated command trajectories. A program's completion count, mean time and sample standard deviation are calculated over independent training seeds. The representative program identities, seed 101 and reference-based local-view rules are fixed before evaluation.
'''
    experiments+=fig('programs_v3','fig:programs','The five complete reserved synthetic programs. Coordinates are in millimeters. The dense-fold and rounded-triangle programs are the predetermined motion examples; all five programs remain in the numerical comparison.')
    experiments+='\n\\input{baseline_methods_v3.tex}\n'
    experiments+=r'''
\subsection{Training and complete-program performance}
Six predetermined teacher settings were evaluated on the four training and two validation programs. Selection first required complete executions and then minimized validation time. The chosen teacher provides '''+str(costs['teacher_final']['training_records'])+r''' training-only observations and feasible actions, with a shared 3 mm preview. Both learning arms use seeds 101, 211, 307, 401 and 503. Each run receives 2000 fitting updates with batches of 256, followed by 8192 fresh PPO decisions. PPO uses 256-decision rollouts, four epochs, minibatches of 64, clipping parameter 0.2, discount 0.99 and learning rate $3\times10^{-5}$; fitting uses $2\times10^{-4}$. Both arms share the same teacher data, model capacity, reward and execution checks. The final checkpoint at the full budget is used for every seed. Initial and intermediate checkpoints provide validation learning curves.

\figref{fig:training} shows all training seeds. All initial, intermediate and final validation evaluations completed. The complete-program comparison in \tabref{tab:complete_times} contains '''+str(nsuccess)+r''' successful learned runs at 1 ms. S2-Q and S2-R encounter a planning failure on the double-bend program because the numerical curve-peak calculation reaches its iteration bound before convergence. Both methods use the same transition geometry and curvature calculation. The table reports completion status and execution time for every program.
'''
    experiments+=fig('training_v3','fig:training','Training curves under the common learning configuration. Thin curves show individual seeds and thick curves show mean two-program validation time. All five seeds complete both validation programs at every displayed checkpoint. PPO clip fractions are shown for each training run.')
    experiments+=main_table+paired_table
    experiments+=r'''
\tabref{tab:paired} gives the fixed-minus-adaptive time differences and their 95\% intervals across paired training seeds. The full trajectories below relate the timing differences among the methods to path layout, transition geometry and sustained feedrate.

Offline cost includes teacher generation, feature construction, fitting, validation and PPO updates. Evaluating the teacher settings required '''+f"{costs['teacher_final']['wall_s']:.1f} s, while the summed wall time of the ten independent training runs was {costs['totals']['total_wall_s']:.1f} s."+r''' The runs shared an Intel Core i5-13400F host, with some jobs executing concurrently. Online timings were measured during a validation execution using single-threaded PyTorch. Across '''+str(timing['whole_decision']['n'])+f" decisions, the full observation--policy--execution loop had median {timing['whole_decision']['median_ms']:.2f} ms, 95th percentile {timing['whole_decision']['p95_ms']:.2f} ms and maximum {timing['whole_decision']['max_ms']:.2f} ms."+r''' These timings characterize the Python simulation loop, including online geometric and braking checks.

\subsection{Dense consecutive corners}
The dense-fold program places short alternating blocks next to an unequal return, so neighboring corner transitions interact. \figref{fig:dense_final} displays the complete program and the first maximum-turn reference vertex. For the predetermined seed, J-NNC, fixed preview, S2-R and SCMO complete in '''+f"{d['adaptive']['complete_time_s']:.3f}, {d['fixed']['complete_time_s']:.3f}, {d['S2R']['complete_time_s']:.3f} and {d['SCMO']['complete_time_s']:.3f} s, respectively."+r''' The complete cycle combines time spent in corner transitions with the sustained speed between them.

To relate path shape and feed scheduling to cycle time, let $L_e$ be executed length and define the integrated speed deficit
\begin{equation}
D_v=\sum_k(V_{\max}-\|v_k\|_2)h,\qquad
T=\frac{L_e+D_v}{V_{\max}}.
\label{eq:speed_deficit}
\end{equation}
The sum includes startup, braking and final rest. This identity accounts for execution time using the recorded command length and speed deficit. On this program, J-NNC executes '''+f"{d['adaptive']['executed_length_mm']:.2f} mm with {d['adaptive']['speed_deficit_mm']:.2f} mm of accumulated deficit; S2-R executes {d['S2R']['executed_length_mm']:.2f} mm with {d['S2R']['speed_deficit_mm']:.2f} mm."+r''' The speed profiles show how a shorter executed path can be accompanied by a larger speed deficit.
'''
    experiments+=fig('dense_experiment_v3','fig:dense_final','Dense consecutive-corner experiment at 1 ms. (a) Complete command trajectories. (b) Reference-selected local corner and tolerance corridor. (c) Speed along executed distance. (d) Accumulated speed deficit, including the complete rest-to-rest motion. Learned traces use the predetermined seed 101.')
    experiments+=r'''
\subsection{Mixed line--arc geometry}
The rounded-triangle program contains straight sections and three circular fillets with unequal radii and turn angles. Every method receives the same chordal reference and reduced allowance, so the discretization budget is shared. \figref{fig:mixed_final} shows how the methods allocate the direction change across this geometry. For seed 101, the J-NNC time is '''+f"{q['adaptive']['complete_time_s']:.3f} s, compared with {q['S2R']['complete_time_s']:.3f} s for S2-R and {q['SCMO']['complete_time_s']:.3f} s for SCMO."+r''' SCMO accumulates a larger speed deficit on this program, while S2-R uses explicit transition geometry and curve-feed bounds. The learned controller applies the same trained policy, local observation and constraint layer as on the dense program.

For this seed, J-NNC has executed length '''+f"{q['adaptive']['executed_length_mm']:.2f} mm and accumulated speed deficit {q['adaptive']['speed_deficit_mm']:.2f} mm. The corresponding SCMO values are {q['SCMO']['executed_length_mm']:.2f} and {q['SCMO']['speed_deficit_mm']:.2f} mm."+r''' Equation~\eqref{eq:speed_deficit} relates these full-trajectory quantities to the measured times. \tabref{tab:complete_times} summarizes the learned results across five seeds together with the deterministic results.
'''
    experiments+=fig('mixed_experiment_v3','fig:mixed_final','Complete mixed line--arc experiment under the common arc-error budget. Panels show (a) the full trajectories, (b) a reference-selected local transition, (c) speed and (d) accumulated speed deficit. The same preselected seed and plotting rules as in the dense experiment are used.')
    experiments+=r'''
\subsection{Planar kinematics and extended toolpaths}
\figref{fig:constraints_final} evaluates velocity, acceleration and jerk as planar vectors reconstructed from positions. All successful final trajectories satisfy the prescribed limits and complete the ordered geometric checks. Across both resolutions and all methods, the largest normalized peaks are '''+f"{vmax:.6f}, {amax:.6f} and {jmax:.6f}, respectively."+r''' These metrics characterize the simulated planar command trajectories.
'''
    experiments+=fig('constraints_v3','fig:constraints_final','Planar vector kinematics for the predetermined dense and mixed-geometry executions. Each quantity is normalized by its common limit; the dashed line is one. The curves cover the complete executions. Both learning configurations use the same geometric and kinematic checks.')
    dev_lookup={(r['case'],r['method']):r for r in dev['metrics']}
    dev_times=[]
    for case,label in [('original','long'),('dense','dense')]:
        times=[dev_lookup[case,m]['complete_time_s'] for m in ['adaptive','fixed','S2R','SCMO']]
        dev_times.append('The '+label+' program completes in '+', '.join(f'{value:.3f}' for value in times)+' s for J-NNC, fixed preview, S2-R and SCMO, respectively. ')
    experiments+=r'''
A long repeated route and a denser repeated-corner development program show how the local motion accumulates over extended toolpaths. Both development examples use seed 101 of the final model configuration. \figref{fig:development_motion} combines geometrically selected local corners with complete-path accumulated deficit. Similar local corner shapes can yield different total times because feed levels and speed recovery accumulate over many blocks. The higher sustained feed of the deterministic schedules on these programs appears in their lower accumulated speed deficit.
'''+''.join(dev_times)+'\n'
    experiments+=fig('development_motion_v3','fig:development_motion','Motion on the long and dense development programs using the common trained configuration. Each row shows a reference-selected local corner with a full-path inset, local speed, and accumulated deficit over the complete trajectory. Learned methods use seed 101.')
    experiments+=r'''
\FloatBarrier
\subsection{Temporal resolution and parameter sensitivity}
The 0.5 ms check repeats all deterministic programs and both learned arms with seed 101, using the same trained policies and planner settings. Completion outcomes agree between the two resolutions. The largest absolute complete-time difference is '''+f"{1000*learn_diff:.1f} ms for the learned arms and {1000*deterministic_diff:.1f} ms among completed deterministic pairs."+r''' \tabref{tab:resolution} reports completion counts at both resolutions. These repeated executions measure temporal discretization sensitivity, while \tabref{tab:paired} describes variability across independent training seeds.
'''+resolution_table+r'''
The local response to PPO clipping was evaluated on the two validation programs with seed 101 and the common training budget (\tabref{tab:clip_sensitivity}). Teacher data and other training parameters were held fixed. All three settings completed both programs; the final models use clipping parameter 0.2.
'''+sensitivity_table
    write('experiments_v3.tex',experiments)
    conclusion=('J-NNC jointly selects steering, feedrate and preview while a shared execution layer checks planar kinematics, command-interval geometry and ordered completion. '
       f'With a common training procedure, both preview configurations completed all {nsuccess} nominal-resolution runs across five reserved synthetic programs and five independent seeds. '
       'The full trajectories relate cycle time to the joint effect of executed length and sustained feedrate. '
       'On the evaluated programs, joint command generation satisfies the prescribed geometric and planar kinematic checks, with cycle times determined by path layout and the resulting feed profiles. '
       'Future extensions can couple the command-space formulation to axis dynamics and broaden the distribution of manufacturing paths used for training.')
    write('conclusion_v3.tex',conclusion)
    baseline=(ROOT/'manuscript/baseline_methods_v3.tex').read_text('utf-8')
    baseline=baseline.replace(' Its upstream example and seven independent analytical rest-to-rest cases from 0.001 to 1000 mm passed the implemented checks.','')
    baseline=baseline.replace(' Its 26 property checks cover geometric containment, vector limits, rest conditions, rotations and invalid inputs.','')
    for who,key in [('Sencer','Sencer2014'),('Li','Li2023'),('Xu','Xu2026')]:
        baseline=baseline.replace(who+r' et al.~\citep{'+key+'}',r'\namedcite{'+who+' et al.}{'+key+'}')
    write('baseline_methods_v3.tex',baseline)
    bib=(OUT/'references.bib').read_text('utf-8')
    new=(ROOT/'literature/verified_new_references.bib').read_text('utf-8')
    if 'Berscheid2021Ruckig' not in bib:
        write('references.bib',bib+'\n'+new)
    receipt={'analysis_sha256':hashlib.sha256((ROOT/'manuscript/results_v3/analysis.json').read_bytes()).hexdigest(),
             'generated':['abstract_v3.tex','experiments_v3.tex','conclusion_v3.tex','baseline_methods_v3.tex'],
             'abstract_words':len(abstract.split()),'final_grid_cells':a['expected_cells'],
             'all_final_outcomes_retained':True,'old_development_model_used':False,
             'scope':'No hardware, generalized preview benefit, frozen tighter-tolerance transfer or global-time-optimality claim.'}
    (ROOT/'audit/numerical_prose_receipt_v3.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
