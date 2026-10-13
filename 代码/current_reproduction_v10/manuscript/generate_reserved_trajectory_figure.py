"""Plot predetermined successful reserved trajectories without executing experiments."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'manuscript'
PATHS = ['reserve_raster', 'reserve_keyhole', 'reserve_three_lobe']
NAMES = dict(zip(PATHS, ['Rounded raster', 'Keyhole', 'Three-lobe profile']))
METHODS = [('adaptive', 101), ('SCMO', None)]
COLORS = {'adaptive': '#246D91', 'SCMO': '#B18A45'}
DT = .001
ZOOM = (-2., 4., 2., 8.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    summary_file = ROOT/'results/reserved_v2/summary.json'
    freeze_file = ROOT/'audit/reserved_evaluation_freeze.json'
    source_hashes = {str(p.relative_to(ROOT).as_posix()): sha(p) for p in [summary_file, freeze_file]}
    summary, freeze = read(summary_file), read(freeze_file)
    vmax = float(freeze['constraints']['speed_mm_s'])
    require(vmax == 100., 'The registered time-accounting reference speed must be 100 mm/s')
    require(summary['valid_for_manuscript'] is True and summary['recorded'] == summary['expected'] == 80,
            'The complete 80-cell evaluation is required')
    model = [r for r in freeze['models'] if r['method'] == 'adaptive' and r['seed'] == 101]
    require(len(model) == 1 and sha(ROOT/model[0]['file']) == model[0]['sha256'], 'Frozen seed-101 model mismatch')
    source_hashes[model[0]['file']] = model[0]['sha256']
    trajectories, metrics = {}, []
    for path in PATHS:
        common_reference = None
        for method, seed in METHODS:
            stem = f'{method}_{seed}_{path}_{DT}'
            result_file = ROOT/'results/reserved_v2'/f'{stem}.json'
            trace_file = result_file.with_suffix('.npz')
            result = read(result_file)
            require(result['success'] is True and result['method'] == method and result['seed'] == seed
                    and result['path'] == path and result['dt_s'] == DT, 'Unexpected successful source identity: '+stem)
            require(float(result['limits'][0]) == vmax, 'Inconsistent physical speed bound: '+stem)
            require(sha(trace_file) == result['trace_sha256'], 'Trace hash mismatch: '+stem)
            for source in [result_file, trace_file]:
                source_hashes[source.relative_to(ROOT).as_posix()] = sha(source)
            with np.load(trace_file, allow_pickle=False) as data:
                reference = data['reference'].copy()
                positions = data['positions'].copy()
                if 'dt_s' in data:
                    require(float(data['dt_s']) == DT, 'Trace timestep mismatch: '+stem)
            reference_file = ROOT/'prospective/reserved_programs'/f'{path}.csv'
            reference_relative = reference_file.relative_to(ROOT).as_posix()
            require(sha(reference_file) == freeze['sha256'][reference_relative], 'Frozen reference hash mismatch')
            require(np.array_equal(reference, np.loadtxt(reference_file, delimiter=',', skiprows=1)), 'Reference mismatch')
            source_hashes[reference_relative] = sha(reference_file)
            require(len(positions) == result['samples'] and np.isfinite(positions).all(), 'Incomplete positions: '+stem)
            require(np.array_equal(positions[0], reference[0]), 'Initial sample mismatch: '+stem)
            if common_reference is not None:
                require(np.array_equal(reference, common_reference), 'Methods have different references')
            common_reference = reference
            displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            speed = displacement/DT
            length, duration = float(displacement.sum()), (len(positions)-1)*DT
            peak, mean = float(speed.max()), length/duration
            require(duration == result['complete_time_s'], 'Duration mismatch: '+stem)
            require(abs(length-result['executed_length_mm']) < 1e-9, 'Length mismatch: '+stem)
            require(abs(peak-result['vector_peaks'][0]) < 1e-9, 'Speed mismatch: '+stem)
            cumulative = np.r_[0., np.cumsum(displacement)]
            trajectories[path, method] = dict(reference=reference, positions=positions,
                cumulative=cumulative, speed=np.r_[0., speed], times=np.arange(len(positions))*DT)
            metrics.append(dict(path=path, method=method, seed=seed, sample_count=len(positions), dt_s=DT,
                executed_length_mm=length, full_cycle_time_s=duration,
                peak_planar_speed_mm_s=peak, time_weighted_mean_planar_speed_mm_s=mean,
                speed_deficit_distance_mm=vmax*duration-length,
                terminal_rest_s=result['terminal_rest_s'],
                max_sampled_polyline_deviation_mm=result['max_polyline_deviation_mm'],
                endpoint_error_mm=result['geometry']['endpoint_error_mm'],
                source_result=result_file.relative_to(ROOT).as_posix(), source_trace=trace_file.relative_to(ROOT).as_posix()))
    # All validity checks precede figure/data output.
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.titlesize': 8.2, 'axes.labelsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .65, 'path.simplify': False,
        'lines.solid_capstyle': 'round'})
    fig = plt.figure(figsize=(7.2, 5.15))
    grid = fig.add_gridspec(2, 3, left=.072, right=.985, bottom=.10, top=.88,
                           hspace=.45, wspace=.38, height_ratios=[1.05, 1.])
    axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    zoom_ax = fig.add_subplot(grid[1, 0])
    speed_ax = fig.add_subplot(grid[1, 1:])
    handles = [Line2D([0],[0],color='#676767',ls=(0,(3,2)),lw=.9,label='Reference polyline'),
               Line2D([0],[0],color=COLORS['adaptive'],lw=1.15,label='Adaptive preview'),
               Line2D([0],[0],color=COLORS['SCMO'],ls=(0,(5,1.7)),lw=1.15,label='SCMO')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53,.997), ncol=3,
               frameon=False, columnspacing=1.8, handlelength=2.5)
    by_metric = {(r['path'],r['method']): r for r in metrics}
    all_boxes = []
    for path in PATHS:
        points = np.vstack([trajectories[path, 'adaptive']['reference'],
                            *[trajectories[path, method]['positions'] for method, _ in METHODS]])
        all_boxes.append((points.min(0), points.max(0)))
    full_span = max(float((hi-lo).max()) for lo, hi in all_boxes)*1.10

    def spatial(ax, path):
        ref = trajectories[path, 'adaptive']['reference']
        ax.plot(ref[:,0], ref[:,1], color='#676767', lw=.85, ls=(0,(3,2)), zorder=1)
        for method, _ in reversed(METHODS):
            pos = trajectories[path, method]['positions']
            ax.plot(pos[:,0], pos[:,1], color=COLORS[method], lw=1.05,
                    ls='-' if method == 'adaptive' else (0,(5,1.7)), zorder=3 if method == 'adaptive' else 2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)', labelpad=2)
        ax.tick_params(length=2.5, width=.6, pad=2)

    for j, (ax, path, (lo, hi)) in enumerate(zip(axes, PATHS, all_boxes)):
        spatial(ax, path)
        mid=(lo+hi)/2
        ax.set_xlim(mid[0]-full_span/2,mid[0]+full_span/2)
        ax.set_ylim(mid[1]-full_span/2,mid[1]+full_span/2)
        ax.set_title(f'{"abc"[j]}  {NAMES[path]}',loc='left',fontweight='bold',pad=7)
        a,b=by_metric[path,'adaptive'],by_metric[path,'SCMO']
        ax.text(.02,.01,f'{a["full_cycle_time_s"]:.3f} s',color=COLORS['adaptive'],
                transform=ax.transAxes,ha='left',va='bottom',fontsize=7.3)
        ax.text(.98,.01,f'{b["full_cycle_time_s"]:.3f} s',color=COLORS['SCMO'],
                transform=ax.transAxes,ha='right',va='bottom',fontsize=7.3)
    x0,x1,y0,y1=ZOOM
    axes[1].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#474747',lw=.7,ls=':'))
    axes[1].annotate('d',xy=(x1,y1),xytext=(4,3),textcoords='offset points',fontsize=7.5)
    spatial(zoom_ax,'reserve_keyhole')
    zoom_ax.set_xlim(x0,x1); zoom_ax.set_ylim(y0,y1)
    zoom_ax.set_title('d  Keyhole: neck corner',loc='left',fontweight='bold',pad=7)
    for method,_ in METHODS:
        data=trajectories['reserve_keyhole',method]
        speed_ax.plot(data['cumulative'],data['speed'],lw=.9,color=COLORS[method],
                      ls='-' if method=='adaptive' else (0,(5,1.7)))
    speed_ax.set_title('e  Keyhole: speed',loc='left',fontweight='bold',pad=7)
    speed_ax.set_xlabel('Executed arc length (mm)')
    speed_ax.set_ylabel('Speed (mm/s)',labelpad=3)
    speed_ax.set_xlim(0.,max(trajectories['reserve_keyhole',m]['cumulative'][-1] for m,_ in METHODS)*1.01)
    speed_ax.set_ylim(0.,85.);speed_ax.set_yticks([0,20,40,60,80])
    speed_ax.grid(axis='y',alpha=.16,lw=.6)
    output_base=OUT/'reserved_successful_trajectories'
    outputs=[]
    for extension in ('pdf','svg','png'):
        file=output_base.with_suffix('.'+extension)
        fig.savefig(file,dpi=300,facecolor='white')
        outputs.append(file)
    plt.close(fig)
    data_file=OUT/'reserved_trajectory_source.csv'
    with data_file.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['path','method','seed','sample','time_s','x_mm','y_mm','executed_arc_length_mm','backward_difference_speed_mm_s'])
        for path in PATHS:
            for method,seed in METHODS:
                data=trajectories[path,method]
                for k,(time,pos,arc,speed) in enumerate(zip(data['times'],data['positions'],data['cumulative'],data['speed'])):
                    writer.writerow([path,method,'' if seed is None else seed,k,*[format(v,'.17g') for v in [time,*pos,arc,speed]]])
    outputs.append(data_file)
    comparisons=[]
    for path in PATHS:
        a,b=by_metric[path,'adaptive'],by_metric[path,'SCMO']
        delta_time = b['full_cycle_time_s']-a['full_cycle_time_s']
        distance_term = (b['executed_length_mm']-a['executed_length_mm'])/vmax
        deficit_term = (b['speed_deficit_distance_mm']-a['speed_deficit_distance_mm'])/vmax
        require(abs(delta_time-distance_term-deficit_term)<1e-12, 'Time-accounting identity failed')
        comparisons.append(dict(path=path,
            scmo_minus_adaptive_time_s=delta_time,
            executed_distance_accounting_term_s=distance_term,
            speed_deficit_accounting_term_s=deficit_term,
            time_accounting_residual_s=delta_time-distance_term-deficit_term,
            adaptive_time_reduction_relative_to_scmo_percent=100*(b['full_cycle_time_s']-a['full_cycle_time_s'])/b['full_cycle_time_s'],
            adaptive_executed_length_reduction_relative_to_scmo_percent=100*(b['executed_length_mm']-a['executed_length_mm'])/b['executed_length_mm'],
            adaptive_full_cycle_mean_speed_increase_relative_to_scmo_percent=100*(a['time_weighted_mean_planar_speed_mm_s']-b['time_weighted_mean_planar_speed_mm_s'])/b['time_weighted_mean_planar_speed_mm_s'],
            scope='Accounting identity for predetermined seed 101; neither term establishes an isolated causal mechanism or preview benefit.'))
    caption=('Representative complete command trajectories from the reserved synthetic programs. '
        '(a-c) Reference polylines, the predetermined adaptive-preview seed 101 and the frozen SCMO comparator on the rounded raster, keyhole and three-lobe programs. '
        'The three full-path panels share an equal millimetre scale; annotated times include startup, braking and final rest. '
        '(d) The marked keyhole-neck region at enlarged equal XY scale. '
        '(e) Unsmoothed planar backward-difference speed versus cumulative executed arc length for the keyhole. '
        'Each trace has its own executed-distance coordinate; the abscissa is neither a synchronized time nor a common reference station. '
        'All commands use a 1 ms interval. These specified successful cases and one predetermined learning seed illustrate commanded motion only; '
        'the complete five-program, five-seed outcomes, including failures, remain in the main results tables. No controller feedback or cutting measurements are shown.')
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        figure='reserved_successful_trajectories',source_summary_sha256=source_hashes[summary_file.relative_to(ROOT).as_posix()],
        selection=dict(paths=PATHS,learning_seed=101,baseline='SCMO',selection_basis='Specified representative successful programs; seed predetermined, not selected by speed.'),
        metrics=metrics,comparisons=comparisons,caption=caption,
        metric_definitions=dict(length='sum(norm(diff(positions)))',time='(N-1)*dt; full recorded command interval',
            speed='norm(diff(positions))/dt; initial display sample is the declared zero initial speed',
            mean_speed='length/time, including stationary intervals',progress='cumulative executed planar distance, separately for each trace',
            speed_deficit_distance='D_all=Vmax*T-L=sum((Vmax-speed[k])*dt), including startup, braking and stationary intervals',
            time_accounting='T_scmo-T_adaptive=(L_scmo-L_adaptive)/Vmax+(D_scmo-D_adaptive)/Vmax; an algebraic identity, not causal attribution',
            vmax_mm_s=vmax),
        rendering=dict(backend='Python/matplotlib',figure_inches=[7.2,5.15],png_dpi=300,
            full_path_xy_span_mm=full_span,keyhole_zoom_xy_limits_mm=ZOOM,equal_xy_aspect=True,
            smoothing=False,resampling=False,path_simplification=False,editable_svg_text=True,pdf_fonttype=42),
        source_sha256=source_hashes,
        script_sha256=sha(__file__),contract_sha256=sha(OUT/'reserved_trajectory_figure_contract.md'),
        outputs_sha256={file.relative_to(ROOT).as_posix():sha(file) for file in outputs},
        limitations=['Synthetic reference programs; single predetermined learning seed.',
                     'Descriptive successful examples do not replace complete success statistics.',
                     'Different command lengths and speed profiles do not identify a causal mechanism or isolated preview benefit.',
                     'Commanded motion only; no machine tracking or surface-quality evidence.'])
    for name,digest in source_hashes.items():
        require(sha(ROOT/name)==digest,'Source changed during plotting: '+name)
    receipt_file=OUT/'reserved_trajectory_receipt.json'
    receipt_file.write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output_base),metrics=metrics,comparisons=comparisons),indent=2))


if __name__=='__main__':
    main()
