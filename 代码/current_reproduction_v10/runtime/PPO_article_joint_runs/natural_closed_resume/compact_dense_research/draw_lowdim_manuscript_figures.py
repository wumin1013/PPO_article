"""Render the frozen14input model with current S2 and trained two-action data."""
import os
os.environ['MPLBACKEND'] = 'Agg'
import importlib.util
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluate_historical_dense import ROOT, BASE, write, sha
from fit_retained_basis_repair import read


def main():
    delivery = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision'
    binding_path = delivery / 'paper_result_binding.json'
    binding = read(binding_path)
    for path, digest in binding['source_sha256'].items():
        assert sha(path) == digest
    work = ROOT / 'lowdim_original_manuscript_work'
    work.mkdir(exist_ok=True)
    output = delivery / 'figures_publication'
    output.mkdir(exist_ok=True)
    helper_source = ROOT.parent / 'all34_figure_helpers.py'
    helper_copy = work / 'lowdim_figure_helpers.py'
    text = helper_source.read_text(encoding='utf-8')
    # The user requested removal of the redundant10mm scale arrow in panel4b.
    text = text.replace("    ax.annotate('', xy=(4210, 2230), xytext=(4200, 2230), arrowprops=dict(arrowstyle='<->', lw=.7, shrinkA=0, shrinkB=0))\n", '')
    text = text.replace("    ax.text(4205, 2236, '10 mm', fontsize=7, ha='center')\n", '')
    helper_copy.write_text(text, encoding='utf-8', newline='\n')
    spec = importlib.util.spec_from_file_location('lowdim_figure_helpers', helper_copy)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    sys.path.insert(0, str(ROOT.parent))
    import draw_all34_publication_figures as previous
    # Reuse drawing utilities only. Replace every plotted dataset and method label.
    for module in [common, previous, previous.common]:
        module.NAMES['Fixed'] = 'Two-action'
        module.COLORS['Fixed'] = '#82748C'
        module.NAMES['Joint'] = 'J-NNC'
    ref = np.loadtxt(binding['cases']['original']['reference'], delimiter=',', skiprows=1)
    geometry = read(binding['cases']['original']['geometry'])
    data = {}
    for name, key in [('S2', 'S2'), ('Two-action', 'Fixed'), ('J-NNC', 'Joint')]:
        entry = binding['methods'][name]['original'][0]
        data[key] = previous.load_trace(entry['trace'], ref)
        payload = read(entry['metrics'])
        data[key].update(result=read(entry['result']), metrics=payload.get('metrics', payload))
    contracts = {
        '4': 'Case geometry and full vector motion: unchanged long route,34angles,10mm Cartesian grid,braking and rest.',
        '5': 'Three prescribed local trajectory comparisons with a second magnification;common corridor and equal axes.',
        '6': 'User-prescribed135/90/45columns and six synchronized rows;one frozen model at two integration steps.',
        '7': 'All34speed deficits and cumulative time below99mm/s;no filled cruise carpet.',
        '8': 'A trained two-output ablation:checkpoint selection opportunities and two local dense-case trajectory details.',
        '9': 'Complete times and resolution-paired efficiency gains;one selected model,no invented independent-seed errorbars.'}
    write(output / 'figure_contract.json', dict(binding=str(binding_path), binding_sha256=sha(binding_path),
        backend='Python matplotlib', main_figure_count=9, figure4_continued=True,
        protected_figures='Original1--3unchanged', contracts=contracts,
        layout_source='User-prescribed figure roles and approved original-revision local layouts;no external journal precedent claim.',
        statistics='Two physical integration resolutions of one selected model,not independent training repeats.',
        integrity='Recorded rows and held requests only;extrema-preserving display reduction for full traces;no signal smoothing.'))
    for index in range(1, 4):
        shutil.copy2(common.ORIGINAL / f'Figure_{index}.pdf', output / f'Figure_{index}.pdf')
    common.figure4_case(ref, geometry, output, data)
    common.figure4_execution(data, output)
    previous.figure5(ref, data, geometry, output)
    previous.figure7(data, geometry, output)
    mechanism = ROOT / 'direct14_dictionary02_resume_round73/mechanism_0000/figure_review'
    for ext in ['pdf', 'svg', 'png']:
        shutil.copy2(mechanism / f'mechanism_review.{ext}', output / f'Figure_6.{ext}')
    dense = {}
    dense_ref = np.loadtxt(binding['cases']['dense']['reference'], delimiter=',', skiprows=1)
    dense_geometry = read(binding['cases']['dense']['geometry'])
    for name in ['J-NNC', 'Two-action']:
        dense[name] = previous.load_trace(binding['methods'][name]['dense'][0]['trace'], dense_ref)
    figure8(binding, dense_ref, dense_geometry, dense, output, common)
    figure9(binding, output, common)
    write(output / 'figure_data_binding.json', dict(source_sha256=binding['source_sha256'],
        binding_sha256=sha(binding_path), script_sha256=sha(__file__),
        helper_original_sha256=sha(helper_source), helper_copy_sha256=sha(helper_copy),
        reused_drawing_module_sha256=sha(previous.__file__),
        outputs={path.name: sha(path) for path in output.glob('Figure_*.*')},
        protected_original_sha256={str(i): sha(common.ORIGINAL / f'Figure_{i}.pdf') for i in range(1,4)},
        visual_qa_complete=False, final_manuscript_accepted=False))
    print('FROZEN_MODEL_FIGURES', output, flush=True)


def figure8(binding, ref, geometry, data, output, common):
    fig = plt.figure(figsize=(7.25, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1., 1.15], hspace=.5, wspace=.36)
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(ref[:,0], ref[:,1], color='#526876', lw=.8)
    ax.set(aspect='equal', xlabel='X (mm)', ylabel='Y (mm)')
    common.panel(ax, 'a', 'Dense closed case: 84 corners')
    chosen = ['corner_00', 'corner_27']
    for ident in chosen:
        corner = next(c for c in geometry['sharp_corners'] if c['id'] == ident)
        point = np.array(corner['point_mm'])
        ax.plot(*point, 'o', ms=4, color=common.COLORS['Joint'])
        ax.annotate('C'+ident[-2:], xy=point, xytext=(6,6), textcoords='offset points', fontsize=7)
    ax = fig.add_subplot(grid[0, 1])
    counts = [8,16,24,32,40,48]
    run = ROOT / 'direct14_capacity_two_action_ppo_round58'
    for key, style, label in [('1ms','-','1 ms'),('half_ms','--','0.5 ms')]:
        times = [read(run / f'eval_{i:04d}_dense_{key}.json')['complete_time_s'] for i in counts]
        assert all(t is not None for t in times)
        ax.plot(counts, times, linestyle=style, marker='o', markersize=3,
                color=common.COLORS['Fixed'], label='Two-action, '+label)
    for index, style in [(0,'-'),(1,'--')]:
        ax.axhline(binding['methods']['J-NNC']['dense'][index]['complete_time_s'],
                   color=common.COLORS['Joint'], linestyle=style, lw=1.)
    ax.text(28,14.72,'J-NNC reference',color=common.COLORS['Joint'],fontsize=7.5,ha='center')
    ax.set(xlabel='Two-action PPO update', ylabel='Complete dense time (s)',
           xlim=(5,51),ylim=(14.6,16.15),xticks=[8,24,40,48])
    ax.legend(fontsize=7,loc='upper left')
    common.panel(ax, 'b', 'Two-action checkpoint selection')
    local_rows = {}
    for col, ident in enumerate(chosen):
        ax = fig.add_subplot(grid[1,col])
        corner = next(c for c in geometry['sharp_corners'] if c['id']==ident)
        vertex, rotation, outgoing = common.local_geometry(ref, corner)
        common.corridor(ax,outgoing,(-6,4),(-1.5,8.5))
        for name,key in [('Two-action','Fixed'),('J-NNC','Joint')]:
            xy = common.local_curve(data[name],vertex,rotation,corner,window=8.)
            local_rows[ident+'_'+key] = xy
            ax.plot(*xy.T,color=common.COLORS[key],linestyle=common.STYLES[key],lw=1.25,label=name)
        ax.set(xlabel='Incoming-axis x (mm)',ylabel='Inward-normal y (mm)',xticks=[-5,0,4],yticks=[0,4,8])
        common.panel(ax,chr(99+col),f'C{ident[-2:]}: {corner["interior_angle_deg"]:.0f}° local detail')
        ax.legend(fontsize=7,loc='upper left')
    fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.09)
    common.save(fig,'Figure_8',output)
    np.savez_compressed(output/'Figure_8_local_rows.npz',**local_rows)
    write(output/'Figure_8_scope.json',dict(primary_two_action='Trained corresponding-capacity checkpoint24',
        checkpoint_curve='All six scheduled checks from48additional PPO updates;not independent seeds.',
        dense_local_ids=chosen,local_progress_halfwidth_mm=8.,
        local_corridor_display='Local adjacent line-line corridor;actual complete certification uses the full path.',
        no_fixed_preview_as_primary_ablation=True))


def figure9(binding, output, common):
    fig = plt.figure(figsize=(7.25, 5.45))
    grid = fig.add_gridspec(2,2,height_ratios=[1.,1.1],hspace=.62,wspace=.38)
    methods = ['S2','Two-action','J-NNC']
    colors = [common.COLORS[k] for k in ['S2','Fixed','Joint']]
    for col,case in enumerate(['original','dense']):
        ax = fig.add_subplot(grid[0,col])
        for row,(name,color) in enumerate(zip(methods,colors)):
            values=[v['complete_time_s'] for v in binding['methods'][name][case]]
            ax.plot(values,[row-.08,row+.08],color=color,lw=.75)
            for j,(value,offset) in enumerate(zip(values,[-.08,.08])):
                ax.plot(value,row+offset,marker='o' if j==0 else 's',mfc=color if j==0 else 'white',mec=color,ms=4)
            ax.text(max(values)+(.035 if case=='original' else .16),row,
                    f'{values[0]:.3f} / {values[1]:.4f}',fontsize=7,va='center',color=color)
        ax.set(yticks=range(3),yticklabels=methods,xlabel='Complete time (s)',ylim=(2.5,-.5))
        ax.set_xlim((444.05,445.95) if case=='original' else (14.3,23.5))
        ax.grid(axis='x',color='#E5E9EC',lw=.45)
        common.panel(ax,chr(97+col),'Long path' if case=='original' else 'Dense case')
    ax=fig.add_subplot(grid[1,:])
    rows=[('Long / S2','original','S2'),('Long / two-action','original','Two-action'),
          ('Dense / S2','dense','S2'),('Dense / best two-action','dense','conservative')]
    for i,(label,case,other) in enumerate(rows):
        current=[v['complete_time_s'] for v in binding['methods']['J-NNC'][case]]
        baseline=binding['conservative_dense_two_action_times_s'] if other=='conservative' else [v['complete_time_s'] for v in binding['methods'][other][case]]
        gains=100.*(1.-np.array(current)/baseline)
        for j,offset in enumerate([-.10,.10]):
            ax.plot(gains[j],i+offset,marker='o' if j==0 else 's',mfc=common.COLORS['Joint'] if j==0 else 'white',
                    mec=common.COLORS['Joint'],ms=4.5)
        ax.text(max(gains)+.45,i,f'{gains[0]:.2f}% / {gains[1]:.2f}%',va='center',fontsize=8)
    ax.axvline(0,color='#657681',lw=.7)
    ax.set(xlim=(-.4,33),ylim=(3.55,-.65),yticks=range(4),yticklabels=[r[0] for r in rows],xlabel='Complete-time reduction (%)')
    ax.grid(axis='x',color='#E5E9EC',lw=.45)
    common.panel(ax,'c','Paired resolution checks of the same selected model')
    handles=[Line2D([],[],color='#334D60',marker='o',ls='none',ms=4,label='1 ms'),
             Line2D([],[],color='#334D60',marker='s',mfc='white',ls='none',ms=4,label='0.5 ms')]
    fig.legend(handles=handles,loc='upper center',ncol=2,bbox_to_anchor=(.56,1.))
    fig.subplots_adjust(left=.19,right=.98,top=.89,bottom=.10)
    common.save(fig,'Figure_9',output)
    write(output/'Figure_9_repeat_scope.json',dict(selected_model_count=1,
        verified_independent_full_pipeline_repetitions=0,error_bars=False,
        point_definition='Same model at two numerical integration steps;paired values are not independent seed replicates.'))


if __name__ == '__main__':
    main()
