"""Plot v9 Figure 8 using the original geometry-defined windows and complete final commands."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from generate_development_v3 import local_frame
METHODS=('JNN','S2','S2R','SCMO')
def read(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
LABELS = dict(JNN='J-NNC', S2='S2-Q', S2R='S2-R', SCMO='SCMO')
COLORS = dict(JNN='#28688a', S2='#ba7944', S2R='#577862', SCMO='#806595')
STYLES = dict(JNN='-', S2=(0, (4, 1.5)), S2R=(0, (2, 1.3)), SCMO=(0, (5, 1.3, 1, 1.3)))
CASES = dict(original=('Long mixed path', 'corner_02'), dense=('Dense corner path', 'corner_01'))


def plot_arrays(run, frame):
    with np.load(Path(run['trace'])) as z:
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


def generate(records, case_definitions, OUT):
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, frames, references, local_rows = {}, {}, {}, []
    for case, (_, corner) in CASES.items():
        record = records[f'{case}_JNN_dt1000us']
        reference = np.loadtxt(case_definitions[case]['reference'], delimiter=',', skiprows=1)
        references[case] = reference
        frames[case] = local_frame(reference, read(case_definitions[case]['geometry']), corner)
        for method in METHODS:
            key = f'{case}_{method}_dt1000us'
            arrays[key] = plot_arrays(records[key], frames[case])
            records[key]['local_visits'] = arrays[key]['local_visits']
            for i, xy, progress in zip(arrays[key]['local_indices'], arrays[key]['local_xy_mm'], arrays[key]['local_progress_mm']):
                local_rows.append([case, method, int(i), i*.001, *xy, progress, arrays[key]['speed_mm_s'][i]])
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial', 'DejaVu Sans'],
        'font.size':7.5, 'axes.labelsize':7.5, 'axes.titlesize':8., 'xtick.labelsize':7.,
        'ytick.labelsize':7., 'legend.fontsize':7.5, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.65, 'path.simplify':False,'svg.hashsalt':'jnn-v9'})
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
            style = dict(color=COLORS[method],ls=STYLES[method],lw=1.1,rasterized=True)
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
        fig.savefig(OUT / f'figure8.{extension}',dpi=300,facecolor='white')
    plt.close(fig)
    np.savez_compressed(OUT / 'figure8_source.npz', **{key+'__'+k:v for key,a in arrays.items()
                        for k,v in a.items() if k!='local_visits'})
    with (OUT / 'figure8_local.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['case','method','sample','time_s','x_mm','y_mm','progress_mm','speed_mm_s'])
        writer.writerows(local_rows)
    receipt=dict(source_traces={key:r['trace_sha256'] for key,r in records.items()},
        windows={case:corner for case,(_,corner) in CASES.items()},
        selection='first geometric-window visit; complete-path deficit',
        outputs={p.name:sha(p) for p in OUT.glob('figure8*') if p.is_file() and p.name!='figure8_receipt.json'})
    (OUT/'figure8_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
