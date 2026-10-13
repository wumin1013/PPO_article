"""Generate the corrected main-model figure, table and prose from one manifest."""
from pathlib import Path
import csv
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import sha, utc, write_json
from main_evidence_v5 import verify, read, MANIFEST, MODEL_ID, EXECUTOR_ID, METHODS, FIGURE_DEPENDENCIES
from generate_development_v3 import local_frame

OUT = ROOT / 'shared_manuscript/main_model_v5'
LABELS = dict(JNN='J-NNC', S2='S2-Q', S2R='S2-R', SCMO='SCMO')
COLORS = dict(JNN='#28688a', S2='#ba7944', S2R='#577862', SCMO='#806595')
STYLES = dict(JNN='-', S2=(0, (4, 1.5)), S2R=(0, (2, 1.3)), SCMO=(0, (5, 1.3, 1, 1.3)))
CASES = dict(original=('Long mixed path', 'corner_02'), dense=('Dense corner path', 'corner_01'))


def plot_arrays(run, frame):
    with np.load(ROOT / run['trace']) as z:
        p = z['positions'].copy()
    delta = p-frame['center']
    xy = np.column_stack([delta @ frame['incoming'], delta @ frame['normal']])
    progress = delta @ ((frame['incoming']+frame['outgoing'])/(1.+frame['incoming'] @ frame['outgoing']))
    padding = 2.*run['tolerance_mm']
    selected = np.flatnonzero((progress >= -frame['before']) & (progress <= frame['after']) &
        (xy[:, 0] >= -frame['before']-padding) & (xy[:, 0] <= padding) &
        (xy[:, 1] >= -padding) & (xy[:, 1] <= frame['after']+padding))
    visits = np.split(selected, np.flatnonzero(np.diff(selected)>1)+1)
    visits = [v for v in visits if len(v)]
    if not visits:
        raise ValueError('Registered geometric window was not traversed')
    indices = visits[0]
    distance = np.linalg.norm(np.diff(p, axis=0), axis=1)
    speed = np.r_[0., distance/run['dt_s']]
    return dict(positions_mm=p, speed_mm_s=speed, executed_distance_mm=np.r_[0., np.cumsum(distance)],
        deficit_mm=np.r_[0., np.cumsum(100.-speed[1:])*run['dt_s']], local_indices=indices,
        local_xy_mm=xy[indices], local_progress_mm=progress[indices], local_visits=len(visits))


def generate():
    manifest, records = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, frames, references, local_rows = {}, {}, {}, []
    for case, (_, corner) in CASES.items():
        record = records[f'{case}_JNN_dt1000us']
        reference = np.loadtxt(ROOT / record['reference'], delimiter=',', skiprows=1)
        references[case] = reference
        frames[case] = local_frame(reference, read(ROOT / record['geometry']), corner)
        for method in METHODS:
            key = f'{case}_{method}_dt1000us'
            arrays[key] = plot_arrays(records[key], frames[case])
            records[key]['local_visits'] = arrays[key]['local_visits']
            for i, xy, progress in zip(arrays[key]['local_indices'], arrays[key]['local_xy_mm'], arrays[key]['local_progress_mm']):
                local_rows.append([case, method, int(i), i*.001, *xy, progress, arrays[key]['speed_mm_s'][i]])
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial', 'DejaVu Sans'],
        'font.size':7.5, 'axes.labelsize':7.5, 'axes.titlesize':8., 'xtick.labelsize':7.,
        'ytick.labelsize':7., 'legend.fontsize':7.5, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65, 'path.simplify':False})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    fig.subplots_adjust(left=.075, right=.985, bottom=.09, top=.88, hspace=.50, wspace=.38)
    handles = [Line2D([0], [0], color='#4c4c4c', ls='--', lw=.85, label='Reference')]
    handles += [Line2D([0],[0],color=COLORS[m],ls=STYLES[m],lw=1.2,label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc='upper center',ncol=5,bbox_to_anchor=(.53,.997),frameon=False,
               columnspacing=1.2,handlelength=2.3)
    for row, (case, (label, _)) in enumerate(CASES.items()):
        spatial, speed_ax, accounting = axes[row]
        frame = frames[case]
        spatial.plot([-frame['before'],0,0],[0,0,frame['after']],c='#4c4c4c',lw=.9,ls='--')
        for method in reversed(METHODS):
            tr = arrays[f'{case}_{method}_dt1000us']
            style = dict(color=COLORS[method],ls=STYLES[method],lw=1.1)
            spatial.plot(*tr['local_xy_mm'].T,**style)
            speed_ax.plot(tr['local_progress_mm'],tr['speed_mm_s'][tr['local_indices']],**style)
            accounting.plot(tr['executed_distance_mm']/1000.,tr['deficit_mm']/1000.,**style)
        spatial.set(aspect='equal',xlim=(-frame['before']-.75,.75),ylim=(-.75,frame['after']+.75),
                    xlabel='Incoming-tangent coordinate (mm)',ylabel='Normal coordinate (mm)')
        speed_ax.set(xlim=(-frame['before'],frame['after']),ylim=(0,105),yticks=[0,25,50,75,100],
                     xlabel='Local projected progress (mm)',ylabel='Speed (mm/s)')
        speed_ax.axhline(100,c='#777777',ls=':',lw=.6)
        speed_ax.grid(axis='y',alpha=.16,lw=.5)
        accounting.set(xlim=(0,None),ylim=(0,None),xlabel='Executed distance (m)',ylabel='Accumulated speed deficit (m)')
        accounting.grid(axis='y',alpha=.16,lw=.5)
        for col, (ax,title) in enumerate(zip(axes[row],[label+': corner','Corner speed','Full-path speed deficit'])):
            ax.set_title(f"{'abcdef'[row*3+col]}  {title}",loc='left',fontweight='bold',pad=7)
            ax.tick_params(length=2.5,width=.6,pad=2)
        inset = spatial.inset_axes([.06,.59,.33,.35])
        inset.plot(*references[case].T,c='#a0a0a0',lw=.45)
        inset.plot(*frame['center'],'o',c='#222222',ms=2.5)
        inset.set_aspect('equal'); inset.set_axis_off()
        if case == 'original':
            zoom = accounting.inset_axes([.13,.50,.52,.39])
            for method in METHODS:
                tr = arrays[f'{case}_{method}_dt1000us']
                zoom.plot(tr['executed_distance_mm']/1000.,tr['deficit_mm']/1000.,
                          c=COLORS[method],ls=STYLES[method],lw=.8)
            zoom.set(xlim=(0,45),ylim=(0,.14),xticks=[0,20,40],yticks=[0,.07,.14])
            zoom.set_title('Expanded lower range',fontsize=6,pad=2)
            zoom.tick_params(labelsize=5.8,length=2,pad=1)
    for extension in ('pdf','svg','png'):
        fig.savefig(OUT / f'figure8_main_v5.{extension}',dpi=300,facecolor='white')
    plt.close(fig)
    np.savez_compressed(OUT / 'figure8_source.npz', **{key+'__'+k:v for key,a in arrays.items()
                        for k,v in a.items() if k!='local_visits'})
    with (OUT / 'figure8_local.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['case','method','sample','time_s','x_mm','y_mm','progress_mm','speed_mm_s'])
        writer.writerows(local_rows)
    def value(case,method,key='complete_time_s'):
        return records[f'{case}_{method}_dt1000us'][key]
    def saving(case,method):
        return 100.*(value(case,method)-value(case,'JNN'))/value(case,method)
    table = [r'\begin{table}[htbp]',r'\centering',
        r'\caption{Simulated cycle time, executed length and accumulated speed deficit at 1 ms. Each entry covers complete rest-to-rest motion under the common geometric and planar kinematic constraints.}',
        r'\label{tab:main_bound}',r'\begin{tabular}{llrrr}',r'\toprule',
        r'Program & Method & Time (s) & Length (mm) & Deficit (mm) \\',r'\midrule']
    for case in CASES:
        for method in METHODS:
            table.append(f"{'Long' if case=='original' else 'Dense'} & {LABELS[method]} & {value(case,method):.3f} & {value(case,method,'executed_length_mm'):.2f} & {value(case,method,'speed_deficit_mm'):.2f}"+r' \\')
        if case=='original': table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUT / 'main_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    text = (r'\subsection{Motion profiles and cycle time}'+'\n'
        r'The long mixed path and dense-corner path use the same trained J-NNC controller and execution layer. '
        r'The controller has a $14$--$64$--$64$--$3$ actor with $\tanh$ hidden activations, '
        r'96 compact-support preview features and 192 compact-support joint-action features. '
        r'The feature function is $\phi(r)=\max(1-r,0)^4(4r+1)$. '
        r'Steering, feedrate and preview distance are evaluated every 20 ms; the guidance direction and '
        r'planar kinematic projection are updated every 1 ms. Ordered gates and complete command-interval '
        r'checks enforce path traversal, followed by braking and final rest.'+'\n\n'
        r'Figure~\ref{fig:main_motion} shows corners selected from the reference geometry and the accumulated speed deficit '
        r'$D=\sum_i(100-v_i)\Delta t$ over each complete command trajectory. '
        r'With executed length $L=\sum_i v_i\Delta t$, the complete duration is $T=(L+D)/100$. '
        r'This accounts for both corner motion and the sustained speed between corners.'+'\n\n'
        f"J-NNC completed the long path in {value('original','JNN'):.3f} s, compared with {value('original','S2'):.3f}, "
        f"{value('original','S2R'):.3f} and {value('original','SCMO'):.3f} s for S2-Q, S2-R and SCMO. "
        f"Its mean speed over the complete path was {value('original','JNN','mean_speed_mm_s'):.2f} mm/s. "
        f"The accumulated deficit was {value('original','JNN','speed_deficit_mm'):.2f} mm, compared with "
        f"{value('original','S2R','speed_deficit_mm'):.2f} mm for S2-R. "+'\n\n'
        f"On the dense path, J-NNC completed the motion in {value('dense','JNN'):.3f} s, reducing time by "
        f"{saving('dense','S2'):.2f}\\%, {saving('dense','S2R'):.2f}\\% and {saving('dense','SCMO'):.2f}\\% "
        r'relative to S2-Q, S2-R and SCMO, respectively. The local speed profiles and complete-path deficit '
        r'show how the jointly generated transitions sustain higher cornering speed on this path.'+'\n\n'
        r'\input{main_table.tex}'+'\n'
        r'\begin{figure}[htbp]'+'\n'+r'\centering'+'\n'
        r'\includegraphics[width=\linewidth]{figure8_main_v5.pdf}'+'\n'
        r'\caption{Command motion on the long mixed path (top) and dense-corner path (bottom). '
        r'Each row shows a local corner with a full-path locator, local speed, and '
        r'complete-path accumulated speed deficit. The inset in (c) expands the lower range of the same curves. '
        r'Both J-NNC trajectories use the same trained controller. Initial acceleration, braking and final rest '
        r'are included in the complete-path quantities.}'+'\n'
        r'\label{fig:main_motion}'+'\n'+r'\end{figure}'+'\n')
    (OUT / 'main_results.tex').write_text(text,encoding='utf-8')
    claim = (f"On two development programs, the trained controller completes the long path in {value('original','JNN'):.3f} s "
        f"and the dense path in {value('dense','JNN'):.3f} s. Dense-path time is reduced by "
        f"{saving('dense','S2'):.2f}\\%, {saving('dense','S2R'):.2f}\\% and {saving('dense','SCMO'):.2f}\\% "
        'relative to S2-Q, S2-R and SCMO under the common geometric and planar kinematic conditions.\n')
    (OUT / 'abstract_numbers.tex').write_text(claim,encoding='utf-8')
    (OUT / 'highlights_numbers.txt').write_text(
        f"Dense-path time falls by {saving('dense','S2'):.2f}% relative to S2-Q in simulation.\n",encoding='utf-8')
    preview = r'''% !TeX program = xelatex
\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage{fontspec,amsmath,graphicx,booktabs}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{1em}
\setlength{\parskip}{3pt}
\emergencystretch=2em
\begin{document}
\setcounter{section}{4}
\setcounter{subsection}{5}
\setcounter{figure}{7}
\input{main_results.tex}
\end{document}
'''
    (OUT / 'main_model_section.tex').write_text(preview,encoding='utf-8')
    contract = '''# Main-model figure correction contract

Core claim: the restored main controller uses high sustained feed on the two existing development programs, and complete command-motion accounting explains the recorded times.
The quantitative grid retains the existing manuscript's local-geometry, local-speed and full-path-deficit structure. Reference corners remain original/corner_02 and dense/corner_01; the first chronological visit is shown. No case or corner is reselected by ranking. No external visual precedent is asserted for this data-source correction.
Python/matplotlib, 183mm wide, white background, Arial, vector PDF/SVG and 300dpi PNG. Line style and color encode methods. All full-resolution positions and speeds are exported; no smoothing or decimation is used.
All curves come from complete 1ms development executions. One fixed historical model is used; no seed standard deviation or independent-test claim is made. The table, prose, abstract-number fragment and highlight-number fragment use the identical manifest binding. The working section is an integration proof, not a final whole manuscript or a five-seed study.
'''
    (OUT / 'figure_contract.md').write_text(contract,encoding='utf-8')
    write_json(OUT / 'metrics.json',dict(main_model_id=MODEL_ID,executor_id=EXECUTOR_ID,
        manifest_sha256=sha(MANIFEST),runs=records,local_rule='first visit to the unchanged reference-selected corner'))
    files = [p for p in OUT.iterdir() if p.is_file() and p.name!='generation_receipt.json']
    write_json(OUT / 'generation_receipt.json',dict(created_utc=utc(),main_model_id=MODEL_ID,
        executor_id=EXECUTOR_ID,manifest_sha256=sha(MANIFEST),generator_sha256=sha(__file__),
        generator_dependency_sha256={relative:sha(ROOT / relative) for relative in FIGURE_DEPENDENCIES.values()},
        verifier_sha256=sha(ROOT / 'src/main_evidence_v5.py'),
        output_sha256={p.name:sha(p) for p in files},visual_qa='pending',publication_ready=False))
    print(json.dumps(dict(figure=str(OUT / 'figure8_main_v5.pdf'), main_model=MODEL_ID,
                         complete_traces=len(records)),indent=2))


if __name__=='__main__':
    generate()
