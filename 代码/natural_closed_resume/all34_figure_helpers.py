"""Evidence-bound manuscript figures. Plot only recorded motion; do not simulate."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Rectangle
from matplotlib.ticker import MultipleLocator

BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path.insert(0, str(BASE))
from study_io import sha, write_json
from natural_case_metrics import ordered_projection

EVIDENCE = BASE / 'paper_evidence_review/verified_results'
CASE = BASE / 'paper_evidence_review/natural_closed_case'
ORIGINAL = Path(r'D:\Nutstore\1\基于强化学习的运动规划方法\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous\figures')
COLORS = {'Joint': '#17689A', 'S2': '#D27C26', 'Fixed': '#37816D', 'Zero': '#855699', 'Max': '#BC4249'}
STYLES = {'Joint': '-', 'S2': (0, (5, 2)), 'Fixed': (0, (1.3, 1.7)), 'Zero': (0, (5, 2)), 'Max': (0, (4, 1.4))}
NAMES = {'Joint': 'J-NNC', 'S2': 'S2', 'Fixed': 'Fixed', 'Zero': r'$\alpha=0$', 'Max': r'$F=100$ mm/s'}
ORDER = ['S2', 'Fixed', 'Joint']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.titlesize': 9, 'axes.titleweight': 'medium', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': .65, 'xtick.major.width': .6,
    'ytick.major.width': .6, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'legend.frameon': False, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'lines.linewidth': 1.25, 'savefig.facecolor': 'white'})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(fig, name, out):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(out / f'{name}.{ext}', dpi=320 if ext != 'png' else 190, bbox_inches='tight', pad_inches=.055)
    plt.close(fig)


def panel(ax, letter, title=None):
    ax.set_title((f'({letter})  ' + title) if title else f'({letter})', loc='left', pad=9)


def legend_handles(names):
    return [Line2D([], [], color=COLORS[n], ls=STYLES[n], label=NAMES[n]) for n in names]


def extrema_envelope(x, y, width=120):
    # Retain both extrema and endpoints of each display block, with no smoothing.
    starts = np.arange(0, len(y), width)
    ids = []
    for i in starts:
        z = y[i:i + width]
        ids.extend([i, i + np.argmin(z), i + np.argmax(z), i + len(z) - 1])
    ids = np.unique(ids)
    return x[ids], y[ids]


def load_data(binding):
    data = {}
    refs = np.loadtxt(CASE / 'reference.csv', delimiter=',', skiprows=1)
    for name, source in binding['primary_methods'].items():
        for kind in ['trace', 'result', 'metrics']:
            assert sha(Path(source[kind])) == source[kind + '_sha256']
        z = np.load(source['trace'])
        rows = z['rows']
        data[name] = dict(rows=rows, t=(np.arange(len(rows)) + 1) * .001,
            speed=np.linalg.norm(rows[:, 2:4], axis=1),
            actions=z['actions'] if 'actions' in z else None,
            progress=ordered_projection(refs, rows[:, :2]),
            metrics=read(source['metrics']), result=read(source['result']))
    for key, label in [('Zero', 'Zero steering'), ('Max', 'Maximum feed')]:
        source = next(a for a in binding['ablations'] if a['label'] == label)
        result = read(source['result'])
        trace = Path(source['result']).with_name('full_trace.npz')
        assert sha(trace) == result['trace_sha256']
        z = np.load(trace)
        rows = z['rows']
        data[key] = dict(rows=rows, t=(np.arange(len(rows)) + 1) * .001,
            speed=np.linalg.norm(rows[:, 2:4], axis=1), actions=z['actions'],
            progress=ordered_projection(refs, rows[:, :2]), result=result, source=str(trace))
    return refs, data


def merged_lines(geometry):
    runs = []
    current = None
    for p in geometry['primitives']:
        if p['kind'] != 'line':
            current = None
            continue
        a, b = np.array(p['start_mm']), np.array(p['end_mm'])
        direction = (b - a) / np.linalg.norm(b - a)
        if current is not None and np.linalg.norm(a - current['end']) < 1e-6 and np.dot(direction, current['direction']) > 1 - 1e-12:
            current['end'] = b
            current['length_mm'] += p['length_mm']
            current['labels'].append(p['label'])
        else:
            current = dict(start=a, end=b, direction=direction, length_mm=p['length_mm'], labels=[p['label']])
            runs.append(current)
    # A closed path may split the same straight at its arbitrary start point.
    if len(runs) > 1 and geometry['primitives'][0]['kind'] == geometry['primitives'][-1]['kind'] == 'line':
        last, first = runs[-1], runs[0]
        if np.linalg.norm(last['end'] - first['start']) < 1e-6 and np.dot(last['direction'], first['direction']) > 1 - 1e-12:
            first['start'] = last['start']
            first['length_mm'] += last['length_mm']
            first['labels'] = last['labels'] + first['labels']
            runs.pop()
    return runs


def grid10(ax, xlim, ylim, global_view=False):
    # True physical 10 mm Cartesian grid; keep vector lines zoomable in the PDF.
    xs = np.arange(np.ceil(xlim[0] / 10) * 10, xlim[1] + .01, 10)
    ys = np.arange(np.ceil(ylim[0] / 10) * 10, ylim[1] + .01, 10)
    segments = [[(x, ylim[0]), (x, ylim[1])] for x in xs]
    segments += [[(xlim[0], y), (xlim[1], y)] for y in ys]
    ax.add_collection(LineCollection(segments, colors='#AAB7C1' if not global_view else '#DEE5EA',
        linewidths=.32 if not global_view else .06, alpha=.58 if not global_view else .12,
        zorder=-5, rasterized=False))
    ax.set(xlim=xlim, ylim=ylim, aspect='equal')


def dimension(ax, a, b, offset, text, color='#33424C', fontsize=8):
    a, b = np.array(a), np.array(b)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    aa, bb = a + n * offset, b + n * offset
    for p, q in [(a, aa), (b, bb)]:
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=.55)
    ax.annotate('', xy=aa, xytext=bb, arrowprops=dict(arrowstyle='<->', color=color, lw=.7, shrinkA=0, shrinkB=0))
    angle = np.rad2deg(np.arctan2(u[1], u[0]))
    if angle > 90 or angle < -90:
        angle += 180
    mid = (aa + bb) / 2 + n * (.12 * offset)
    ax.text(*mid, text, color=color, fontsize=fontsize, ha='center', va='center', rotation=angle,
            bbox=dict(fc='white', ec='none', pad=1.6))


def figure4_case(refs, geometry, out, data):
    runs = merged_lines(geometry)
    longest, shortest = max(runs, key=lambda p: p['length_mm']), min(runs, key=lambda p: p['length_mm'])
    corners = geometry['sharp_corners']
    amin = min(corners, key=lambda c: c['interior_angle_deg'])
    amax = max(corners, key=lambda c: c['interior_angle_deg'])
    fig = plt.figure(figsize=(7.25, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[5.6, 1.55], width_ratios=[1, 1.55], hspace=.34, wspace=.43)
    ax = fig.add_subplot(gs[0, :])
    grid10(ax, (-500, 13700), (-600, 10200), global_view=True)
    ax.plot(refs[:, 0], refs[:, 1], color='#AAB9C4', lw=1.5, zorder=2)
    ax.plot(data['Joint']['rows'][:, 0], data['Joint']['rows'][:, 1], color='#314D60', lw=1.1, zorder=3)
    ax.set(xlabel='X (mm)', ylabel='Y (mm)', xticks=np.arange(0, 14000, 2000), yticks=np.arange(0, 10000, 2000))
    panel(ax, 'a', 'Closed reference path: 44 420.514 mm; 34 corners')
    ax.text(.98, .98, '10 mm Cartesian grid', transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            bbox=dict(fc='white', ec='none', pad=2))
    for run in [longest, shortest]:
        pp = np.stack([run['start'], run['end']])
        ax.plot(*pp.T, color=COLORS['Joint'], lw=2.2, zorder=4)
    dimension(ax, longest['start'], longest['end'], 520, f"Longest straight\n{longest['length_mm']:.3f} mm", fontsize=7.5)
    ax.annotate('Shortest straight: 80 mm\n(detail in b)', xy=(4160, 2251.47186), xytext=(5400, 3450),
        arrowprops=dict(arrowstyle='->', lw=.7, color='#4C5962'), ha='center', fontsize=8,
        bbox=dict(fc='white', ec='none', pad=2))
    for c, title, xytext in [(amin, r'Minimum $\beta=45.000^\circ$', (8800, 6550)),
                            (amax, r'Maximum $\beta=175.679^\circ$', (3900, -400))]:
        xy = np.array(c['point_mm'])
        ax.plot(*xy, 'o', ms=3.8, mfc='white', mec='#263F50', zorder=6)
        ax.annotate(title + '\n' + c['id'].replace('corner_', 'C'), xy=xy, xytext=xytext,
            arrowprops=dict(arrowstyle='->', color='#4C5962', lw=.7), ha='center', fontsize=8,
            bbox=dict(fc='white', ec='none', pad=2))
        k = c['reference_vertex_index']
        u, v = refs[k-1] - xy, refs[k+1] - xy
        angles = np.rad2deg(np.arctan2([u[1], v[1]], [u[0], v[0]]))
        diff = (angles[1] - angles[0]) % 360
        start, end = (angles[0], angles[0] + diff) if diff <= 180 else (angles[1], angles[1] + 360 - diff)
        ax.add_patch(Arc(xy, 650, 650, theta1=start, theta2=end, color='#495C67', lw=.7))
    for p in geometry['primitives']:
        if p['kind'] != 'arc':
            continue
        center = np.array(p['center_mm'])
        ang = p['start_angle_rad'] + p['sweep_rad'] * .5
        point = center + p['radius_mm'] * np.array([np.cos(ang), np.sin(ang)])
        ax.plot([center[0], point[0]], [center[1], point[1]], color='#687D89', lw=.6, ls='--')
        ax.plot(*center, '+', color='#687D89', ms=4, mew=.7)
        offset = (1300, 300) if p['radius_mm'] == 800 else (-1800, -150)
        ax.annotate(f"R = {p['radius_mm']:.0f} mm\n90° circular arc", xy=point, xytext=center + offset,
            arrowprops=dict(arrowstyle='->', color='#687D89', lw=.65), fontsize=8, ha='center',
            bbox=dict(fc='white', ec='none', pad=2))
    for cid, text, off in [('corner_01', 'C01: 135°', (-1100, -630)), ('corner_02', 'C02: 90°', (1100, 850))]:
        c = next(c for c in corners if c['id'] == cid)
        xy = np.array(c['point_mm'])
        ax.scatter(*xy, s=14, color=COLORS['Joint'], zorder=6)
        ax.annotate(text, xy=xy, xytext=xy + off, fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='-', color=COLORS['Joint'], lw=.65),
            bbox=dict(fc='white', ec='none', pad=1.5))
    ax.plot(*refs[0], 's', color='#26343E', ms=4.5)
    ax.annotate('Start / end', xy=refs[0], xytext=(1550, 9000), ha='center', fontsize=8,
        arrowprops=dict(arrowstyle='->', lw=.7))
    ax.annotate('', xy=(2750, 7800), xytext=(2350, 7800), arrowprops=dict(arrowstyle='->', color='#314D60', lw=1.4))
    ax = fig.add_subplot(gs[1, 0])
    grid10(ax, (4100, 4220), (2220, 2300))
    ax.plot(*refs.T, color='#314D60', lw=1.25)
    dimension(ax, shortest['start'], shortest['end'], 26, '80 mm', fontsize=8)
    ax.set(xlabel='X (mm)', ylabel='Y (mm)', xticks=[4100, 4140, 4180, 4220], yticks=[2220, 2260, 2300])
    ax.ticklabel_format(useOffset=False)
    ax.annotate('', xy=(4210, 2230), xytext=(4200, 2230), arrowprops=dict(arrowstyle='<->', lw=.7, shrinkA=0, shrinkB=0))
    ax.text(4205, 2236, '10 mm', fontsize=7, ha='center')
    panel(ax, 'b', 'Shortest-straight detail')
    ax = fig.add_subplot(gs[1, 1])
    angles = np.array([c['interior_angle_deg'] for c in corners])
    ax.vlines(np.arange(34), 0, angles, color='#CCD5DB', lw=1.1)
    ax.scatter(np.arange(34), angles, s=13, color='#657D8C', zorder=3)
    for cid in geometry['mechanism_columns']:
        i = next(i for i,c in enumerate(corners) if c['id']==cid)
        ax.scatter(i, angles[i], s=26, color=COLORS['Joint'], edgecolors='white', linewidth=.4, zorder=4)
    ax.set(xlim=(-1, 34), ylim=(0, 185), xlabel='Corner index (path order)', ylabel=r'Included angle $\beta$ (°)',
        xticks=range(0, 34, 5), yticks=[0,45,90,135,180])
    ax.grid(axis='y', color='#E3E7EA', lw=.5)
    panel(ax, 'c', r'All 34 angles; $|\delta|=180^\circ-\beta$')
    fig.subplots_adjust(left=.085, right=.985, top=.95, bottom=.065)
    save(fig, 'Figure_4', out)
    def plain(p):
        return {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in p.items()}
    write_json(out/'geometry_annotations.json', dict(longest_straight=plain(longest), shortest_straight=plain(shortest),
        minimum_included_angle=amin, maximum_included_angle=amax,
        definition='Maximal connected, codirectional collinear straight runs, including the closed seam.',
        grid_spacing_mm=10, global_grid_rasterized=False, source_sha256=sha(CASE/'geometry.json')))


def figure4_execution(data, out):
    fig = plt.figure(figsize=(7.25, 6.55))
    gs = fig.add_gridspec(4, 3, height_ratios=[1,1,1,1.1], hspace=.88, wspace=.17)
    limits=[100,2000,62500]
    labels=['Speed\n(mm/s)', 'Acceleration\n(mm/s²)', 'Jerk\n(mm/s³)']
    for col, name in enumerate(ORDER):
        d=data[name]
        for row in range(3):
            ax=fig.add_subplot(gs[row,col])
            yy=np.linalg.norm(d['rows'][:,2+2*row:4+2*row],axis=1)
            xx,yy=extrema_envelope(d['t'],yy)
            ax.plot(xx,yy,color=COLORS[name],lw=.7)
            ax.axhline(limits[row],color='#4B5054',ls='--',lw=.7)
            ax.set(xlim=(0,447),ylim=(-.025*limits[row],1.09*limits[row]),xticks=[0,200,400],yticks=[0,limits[row]])
            if col==0:ax.set_ylabel(labels[row])
            else:ax.tick_params(labelleft=False)
            if row==0:ax.set_title(NAMES[name],color=COLORS[name],pad=10)
            if row==2:ax.set_xlabel('Elapsed time (s)')
    ax=fig.add_subplot(gs[3,:])
    for name in ORDER:
        d=data[name]
        ax.plot(d['t'][-171:]-d['t'][-1],d['speed'][-171:],color=COLORS[name],ls=STYLES[name],label=NAMES[name])
    ax.axvspan(-.02,0,color='#EDF1F4',zorder=-3)
    ax.text(-.01,76,'≥20 ms\nphysical rest',ha='center',fontsize=7.5)
    ax.set(xlabel='Time relative to complete stop (s)',ylabel='Speed\n(mm/s)',xlim=(-.17,.003),ylim=(-3,106),yticks=[0,50,100])
    ax.legend(loc='upper left',ncol=3)
    panel(ax,'g','Complete braking and stationary tail')
    fig.text(.08,.983,'(d–f)  Complete vector V / A / J, separated by method',fontsize=9)
    fig.subplots_adjust(left=.105,right=.985,top=.925,bottom=.085)
    save(fig,'Figure_4_execution',out)


def local_geometry(refs,c):
    k=c['reference_vertex_index']
    u=refs[k]-refs[k-1];u/=np.linalg.norm(u)
    normal=np.array([-u[1],u[0]])*np.sign(c['signed_heading_change_deg'])
    rot=np.stack([u,normal],axis=1)
    delta=np.deg2rad(abs(c['signed_heading_change_deg']))
    return refs[k],rot,np.array([np.cos(delta),np.sin(delta)])


def corridor(ax,outgoing,xlim,ylim):
    x=np.linspace(*xlim,220);y=np.linspace(*ylim,220)
    xx,yy=np.meshgrid(x,y);p=np.stack([xx,yy],axis=-1)
    t=np.minimum(xx,0)
    d1=np.hypot(xx-t,yy)
    t2=np.maximum(p@outgoing,0)
    d2=np.linalg.norm(p-t2[:,:,None]*outgoing,axis=-1)
    distance=np.minimum(d1,d2)
    ax.contourf(xx,yy,distance,levels=[0,1.25],colors=['#EDF1F4'],zorder=-4)
    ax.contour(xx,yy,distance,levels=[1.25],colors=['#AFBCC5'],linewidths=.5,zorder=-3)
    ref=np.stack([[-50,0],[0,0],outgoing*50])
    ax.plot(*ref.T,color='#7C858C',lw=.75,ls=(0,(3,2)),zorder=-1)
    ax.scatter([0],[0],s=10,c='#59636B',marker='+',zorder=5)
    ax.set(xlim=xlim,ylim=ylim,aspect='equal')
    ax.grid(color='#E2E7EB',lw=.4,zorder=-5)


def local_curve(d,vertex,rot,c,window=40):
    mask=(d['progress']>=c['reference_s_mm']-window)&(d['progress']<=c['reference_s_mm']+window)
    return (d['rows'][mask,:2]-vertex)@rot


def figure5(refs,data,geometry,out):
    fig,axes=plt.subplots(2,3,figsize=(7.25,5.75),gridspec_kw={'hspace':.38,'wspace':.24})
    for col,cid in enumerate(geometry['mechanism_columns']):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid)
        vertex,rot,outgoing=local_geometry(refs,c)
        for row,(xlim,ylim) in enumerate([((-10,10),(-2,12)),((-4,2),(-1,5))]):
            ax=axes[row,col];corridor(ax,outgoing,xlim,ylim)
            for name in ORDER:
                xy=local_curve(data[name],vertex,rot,c)
                ax.plot(*xy.T,color=COLORS[name],ls=STYLES[name],lw=1.4,zorder=4 if name=='Joint' else 3)
            if row==0:
                ax.add_patch(Rectangle((-4,-1),6,6,fill=False,ec='#505F69',lw=.6,ls=':'))
                ax.set_xticks([-10,0,10]);ax.set_yticks([0,5,10])
                panel(ax,chr(97+col),f"{c['interior_angle_deg']:.0f}° / C{cid[-2:]}")
            else:
                ax.set_xticks([-4,-2,0,2]);ax.set_yticks([0,2,4]);panel(ax,chr(100+col),'Transition detail')
            if col==0:ax.set_ylabel('Left-normal y (mm)')
            ax.set_xlabel('Incoming-axis x (mm)')
    fig.legend(handles=legend_handles(ORDER)+[Line2D([],[],color='#7C858C',ls='--',lw=.8,label='Reference')],
        loc='upper center',bbox_to_anchor=(.52,1.015),ncol=4)
    fig.text(.5,.017,'Shading: 1.25 mm Euclidean corridor. Equal aspect and common scales within each row.',ha='center',fontsize=7.5)
    fig.subplots_adjust(top=.9,bottom=.115,left=.09,right=.985)
    save(fig,'Figure_5',out)


def figure6(data,geometry,refs,out):
    fig,axes=plt.subplots(6,3,figsize=(7.25,8.0),sharex=True)
    values=[[] for _ in range(6)]
    for col,cid in enumerate(geometry['mechanism_columns']):
        c=next(c for c in geometry['sharp_corners'] if c['id']==cid)
        vertex,rot,_=local_geometry(refs,c)
        axes[0,col].set_title(f"({chr(97+col)})  {c['interior_angle_deg']:.0f}° / C{cid[-2:]}",pad=8)
        for name in ORDER:
            d=data[name];cm=next(m for m in d['metrics']['corners'] if m['id']==cid);tc=cm['crossing_time_s']
            keep=(d['t']>=tc-.35)&(d['t']<=tc+.35)
            v=d['rows'][keep,2:4];a=d['rows'][keep,4:6];speed=d['speed'][keep]
            h=np.rad2deg(np.unwrap(np.arctan2(v@rot[:,1],v@rot[:,0])))
            k=np.sign(c['signed_heading_change_deg'])*(v[:,0]*a[:,1]-v[:,1]*a[:,0])/speed**3
            for row,yy in [(3,h),(4,k),(5,speed)]:
                axes[row,col].plot(d['t'][keep]-tc,yy,color=COLORS[name],ls=STYLES[name],lw=1.05)
                values[row].append(yy)
            if name!='S2':
                aa=d['actions'];ta=np.arange(len(aa))*.02-tc;keep=(ta>=-.38)&(ta<=.38)
                for row,yy in [(0,np.rad2deg(aa[:,0])),(1,aa[:,1]),(2,aa[:,2])]:
                    axes[row,col].step(ta[keep],yy[keep],where='post',color=COLORS[name],ls=STYLES[name],lw=1.05)
                    values[row].append(yy[keep])
        axes[3,col].axhline(abs(c['signed_heading_change_deg']),color='#7F8C95',lw=.7,ls=':')
        axes[5,col].axhline(99,color='#7F8C95',lw=.65,ls=':')
    limits=[(-15,15),(25,104),(0,12),(-8,146),None,(15,104)]
    curvature=np.concatenate(values[4]);limits[4]=(min(-.04,float(curvature.min())*1.1),float(curvature.max())*1.12)
    for row,title in enumerate(['Steering (°)','Requested feed\n(mm/s)','Preview (mm)','Actual heading\n(°)','Signed curvature\n(mm⁻¹)','Actual speed\n(mm/s)']):
        axes[row,0].set_ylabel(title)
        for col in range(3):
            ax=axes[row,col];ax.set(xlim=(-.35,.35),ylim=limits[row],xticks=[-.3,0,.3])
            ax.axvline(0,color='#B0BAC1',lw=.6);ax.grid(axis='y',color='#E7EBEE',lw=.5)
            if col:ax.tick_params(labelleft=False)
    for ax in axes[-1]:ax.set_xlabel('Time from gate crossing (s)')
    fig.legend(handles=legend_handles(ORDER),loc='upper center',bbox_to_anchor=(.55,1),ncol=3)
    fig.subplots_adjust(left=.115,right=.985,top=.938,bottom=.062,hspace=.23,wspace=.17)
    save(fig,'Figure_6',out)


def cruise_intervals(speed):
    transitions=np.diff(np.r_[False,speed>=99,False].astype(int))
    starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)
    return starts*.001,(ends-starts)*.001


def figure7(data,geometry,out):
    fig=plt.figure(figsize=(7.25,5.75))
    gs=fig.add_gridspec(3,1,height_ratios=[2.3,1.5,1.5],hspace=.68)
    ax=fig.add_subplot(gs[0])
    x=np.arange(34)
    for name in ORDER:
        yy=[c['speed_deficit_mm'] for c in data[name]['metrics']['corners']]
        ax.plot(x,yy,color=COLORS[name],ls=STYLES[name],marker={'Joint':'o','S2':'^','Fixed':'s'}[name],
            ms=3.3,mew=.6,mfc='white' if name!='Joint' else COLORS[name],lw=1,label=NAMES[name])
    ax.set(xlim=(-.7,33.7),ylim=(-.5,15.3),xlabel='All 34 corners in path order',ylabel='Speed deficit (mm)',xticks=np.arange(0,34,2))
    ax.set_xticks(np.arange(34),minor=True)
    ax.tick_params(axis='x',labelsize=8)
    ax.grid(axis='y',color='#E2E7EB',lw=.5)
    ax.legend(loc='upper center',ncol=3)
    for cid in geometry['mechanism_columns']:
        idx=int(cid[-2:]);ax.axvspan(idx-.24,idx+.24,color='#EDF1F4',zorder=-3)
    panel(ax,'a','Integrated corner speed loss, registered ±10 mm windows')
    intervals={}
    for i,(start,end) in enumerate([(0,225),(225,450)]):
        ax=fig.add_subplot(gs[1+i]);ax.set(xlim=(start,end),ylim=(-.6,2.6),yticks=[0,1,2],yticklabels=[NAMES[n] for n in ORDER],
            xlabel='Complete elapsed time (s)',xticks=np.arange(start,end+1,45))
        ax.invert_yaxis();ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        for row,name in enumerate(ORDER):
            d=data[name];s,length=cruise_intervals(d['speed']);intervals[name]=dict(start_s=s.tolist(),duration_s=length.tolist())
            ax.plot([max(0,start),min(d['t'][-1],end)],[row,row],color='#E7EBEE',lw=3.3,solid_capstyle='butt',zorder=0)
            pieces=[[(a,row),(a+b,row)] for a,b in zip(s,length) if a<end and a+b>start]
            ax.add_collection(LineCollection(pieces,colors=COLORS[name],linewidths=3.3,capstyle='butt'))
            j=int(np.argmax(length));a,b=s[j],s[j]+length[j]
            if a>=start and b<=end:
                ax.annotate('',xy=(a,row+.22),xytext=(b,row+.22),arrowprops=dict(arrowstyle='|-|',lw=.65,color='#49545C',shrinkA=0,shrinkB=0))
            if i==0:
                total=(d['speed']>=99).sum()*.001;longest=length.max()
                ax.text(end+7,row,f'{total:.3f} s / {longest:.3f} s',fontsize=7.6,va='center',clip_on=False,color=COLORS[name])
        if i==0:
            ax.text(end+7,-.72,'Total / longest cruise',fontsize=7.5,clip_on=False)
        ax.grid(axis='x',color='#D7DEE4',lw=.5)
        panel(ax,chr(98+i),'≥99 mm/s cruise: 0–225 s' if i==0 else '≥99 mm/s cruise: 225–450 s')
    fig.text(.10,.015,'Color: ≥99 mm/s; gray: below threshold. Brackets mark the longest uninterrupted intervals.',fontsize=7.6)
    fig.subplots_adjust(left=.105,right=.74,bottom=.10,top=.94)
    # Keep all-corner panel wide; reserve the right margin only beside the cruise bands.
    first=fig.axes[0];p=first.get_position();first.set_position([p.x0,p.y0,.88,p.height])
    save(fig,'Figure_7',out)
    write_json(out/'Figure_7_cruise_intervals.json',dict(threshold_mm_s=99,sample_interval_s=.001,methods=intervals))


def figure8(refs,data,geometry,out):
    fig=plt.figure(figsize=(7.25,5.9))
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.5],hspace=.5,wspace=.42)
    ax=fig.add_subplot(gs[0,:])
    names=['Joint','Fixed','Zero']
    differences=[1000*(data[n]['result']['complete_time_s']-data['Joint']['result']['complete_time_s']) for n in names]
    ax.barh([0,1,2],differences,height=.5,color=[COLORS[n] for n in names],alpha=.9)
    ax.plot(0,0,'o',color=COLORS['Joint'],ms=5)
    for y,n,v in zip(range(3),names,differences):
        ax.text(v+5,y,f"{v:+.0f} ms  ({data[n]['result']['complete_time_s']:.3f} s)",va='center',fontsize=8)
    ax.set(xlim=(0,205),ylim=(2.55,-.65),yticks=[0,1,2],yticklabels=[NAMES[n] for n in names],xlabel='Additional complete time relative to J-NNC (ms)')
    ax.grid(axis='x',color='#E5E9EC',lw=.5);ax.set_axisbelow(True)
    panel(ax,'a','Preview and steering: complete-task efficiency')
    c=next(c for c in geometry['sharp_corners'] if c['id']=='corner_06')
    vertex,rot,outgoing=local_geometry(refs,c)
    ax=fig.add_subplot(gs[1,0]);corridor(ax,outgoing,(-4,2),(-1,5))
    for name in ['Fixed','Zero','Joint']:
        xy=local_curve(data[name],vertex,rot,c);ax.plot(*xy.T,color=COLORS[name],ls=STYLES[name],label=NAMES[name],lw=1.4)
    panel(ax,'b','45° corner: preview and steering')
    ax.set(xlabel='Incoming-axis x (mm)',ylabel='Left-normal y (mm)',xticks=[-4,-2,0,2],yticks=[0,2,4])
    ax.legend(loc='upper right',fontsize=7.5)
    ax=fig.add_subplot(gs[1,1]);corridor(ax,outgoing,(-4,2),(-1,5))
    for name in ['Max','Joint']:
        xy=local_curve(data[name],vertex,rot,c);ax.plot(*xy.T,color=COLORS[name],ls=STYLES[name],label=NAMES[name],lw=1.4)
    last=(data['Max']['rows'][-1,:2]-vertex)@rot
    ax.plot(*last,marker='x',ms=7,mew=1.3,color=COLORS['Max'])
    ax.annotate('Corridor violation\nat 81.596 s',xy=last,xytext=(-3.65,3.9),fontsize=7.6,color=COLORS['Max'],
        arrowprops=dict(arrowstyle='->',color=COLORS['Max'],lw=.7),bbox=dict(fc='white',ec='none',pad=1.5))
    ax.set(xlabel='Incoming-axis x (mm)',ylabel='Left-normal y (mm)',xticks=[-4,-2,0,2],yticks=[0,2,4])
    panel(ax,'c','45° corner: coordinated feedrate')
    ax.legend(loc='lower left',fontsize=7.4)
    fig.subplots_adjust(top=.945,left=.12,right=.97,bottom=.085)
    save(fig,'Figure_8',out)


def figure9(data,binding,out):
    fig=plt.figure(figsize=(7.25,6.4))
    gs=fig.add_gridspec(3,1,height_ratios=[1.2,1.05,1.25],hspace=.65)
    refined={n:read(p)['complete_time_s'] for n,p in binding['resolution_results'].items()}
    ax=fig.add_subplot(gs[0]);comparators=['S2','Fixed']
    gains={}
    for i,n in enumerate(comparators):
        gains[n]=[1000*(data[n]['result']['complete_time_s']-data['Joint']['result']['complete_time_s']),1000*(refined[n]-refined['Joint'])]
        for j,gain in enumerate(gains[n]):
            ax.barh(i+(j-.5)*.27,gain,height=.23,color=COLORS[n],alpha=1 if j==0 else .45,
                hatch=None if j==0 else '//',edgecolor=COLORS[n],lw=.5)
            ax.text(gain+5,i+(j-.5)*.27,f'{gain:g} ms',va='center',fontsize=8)
    ax.axvline(100,color='#505D66',ls=':',lw=1)
    ax.text(104,.58,'100 ms acceptance margin',fontsize=7.5,ha='left')
    ax.set(xlim=(0,510),ylim=(1.52,-.55),yticks=[0,1],yticklabels=['vs S2','vs Fixed'],xlabel='J-NNC complete-time advantage (ms)')
    ax.legend(handles=[Rectangle((0,0),1,1,fc='#607B8C',label='1 ms execution'),
        Rectangle((0,0),1,1,fc='#C6D1D8',ec='#607B8C',hatch='//',label='0.5 ms execution')],loc='lower right',ncol=2,fontsize=7.5)
    panel(ax,'a','Time gains persist at both execution resolutions')
    ax.grid(axis='x',color='#E5E9EC',lw=.5);ax.set_axisbelow(True)
    ax=fig.add_subplot(gs[1]);decomp={}
    for i,n in enumerate(comparators):
        length=lambda name:data[name]['speed'].sum()*.001
        geometric=(length(n)-length('Joint'))/100*1000
        total=gains[n][0];velocity=total-geometric;decomp[n]=dict(shorter_path_ms=geometric,reduced_speed_deficit_ms=velocity)
        ax.barh(i,geometric,height=.45,color='#899EAC',label='Shorter executed path' if i==0 else None)
        ax.barh(i,velocity,left=geometric,height=.45,color=COLORS['Joint'],label='Reduced speed deficit' if i==0 else None)
        ax.text(geometric+velocity/2,i,f'{velocity:.1f} ms',ha='center',va='center',color='white',fontsize=8)
        if geometric>15:ax.text(geometric/2,i,f'{geometric:.1f}',ha='center',va='center',fontsize=7.5,color='white')
        else:ax.annotate(f'{geometric:.1f} ms',xy=(geometric/2,i),xytext=(40,i-.38),fontsize=7.5,
            arrowprops=dict(arrowstyle='-',lw=.6),ha='center')
        ax.text(total+5,i,f'{total:.0f} ms',va='center',fontsize=8)
    ax.set(xlim=(0,510),ylim=(1.52,-.65),yticks=[0,1],yticklabels=['vs S2','vs Fixed'],xlabel='Source of the 1 ms time advantage (ms)')
    ax.legend(loc='lower right',ncol=2,fontsize=7.5)
    panel(ax,'b',r'Exact decomposition: $T=L_{\rm exec}/100+E_v$')
    ax.grid(axis='x',color='#E5E9EC',lw=.5);ax.set_axisbelow(True)
    ax=fig.add_subplot(gs[2])
    for i,n in enumerate(['Joint','Fixed']):
        for j,record in enumerate(binding['conditional_full_repeats'][n]):
            r=read(record['result']);x=j+(-.09 if n=='Joint' else .09)
            if r['success']:
                y=1000*(data['S2']['result']['complete_time_s']-r['complete_time_s'])
                ax.plot(x,y,'o' if n=='Joint' else 's',color=COLORS[n],ms=5)
                ax.annotate(f'{r["complete_time_s"]:.3f} s',(x,y),xytext=(0,8 if n=='Joint' else -15),textcoords='offset points',ha='center',fontsize=7.5,color=COLORS[n])
            else:
                ax.text(x,62,'Incomplete',color=COLORS[n],ha='center',fontsize=7.5)
                ax.text(x,25,f'prefix {r["elapsed_prefix_or_complete_s"]:.3f} s',color=COLORS[n],ha='center',fontsize=7.2)
    fixed_gain=gains['Fixed'][0]
    strong_fixed=1000*(data['S2']['result']['complete_time_s']-data['Fixed']['result']['complete_time_s'])
    ax.axhline(strong_fixed,color=COLORS['Fixed'],ls=':',lw=.9)
    ax.text(2.47,strong_fixed,'Strong Fixed\n444.882 s',ha='left',va='center',color=COLORS['Fixed'],fontsize=7.4)
    ax.axhline(0,color=COLORS['S2'],ls='--',lw=.9)
    ax.text(2.47,0,'S2\n445.185 s',ha='left',va='center',color=COLORS['S2'],fontsize=7.4)
    ax.set(xlim=(-.4,2.43),ylim=(-22,505),xticks=[0,1,2],xticklabels=['921221','921222','921223'],
        xlabel='Conditional parameter-search seed (shared inherited model)',ylabel='Gain over S2 (ms)',yticks=[0,200,400])
    ax.legend(handles=legend_handles(['Joint','Fixed']),loc='center right',fontsize=7.5)
    panel(ax,'c','Three conditional searches; individual complete outcomes')
    fig.subplots_adjust(left=.135,right=.855,top=.955,bottom=.085)
    save(fig,'Figure_9',out)
    write_json(out/'Figure_9_efficiency.json',dict(gains_ms=gains,decomposition_1ms=decomp,
        conditional_replication_scope='Three searches share one inherited model; zero independent complete PPO replications.'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('output');ap.add_argument('--only',nargs='*',type=int,default=[4,5,6,7,8,9]);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    binding_path=EVIDENCE/'paper_result_binding.json';binding=read(binding_path)
    assert binding['research_acceptance']['passed']
    refs,data=load_data(binding);geometry=read(CASE/'geometry.json')
    for i in range(1,4):shutil.copy2(ORIGINAL/f'Figure_{i}.pdf',out/f'Figure_{i}.pdf')
    for number in args.only:
        if number==4:figure4_case(refs,geometry,out,data);figure4_execution(data,out)
        if number==5:figure5(refs,data,geometry,out)
        if number==6:figure6(data,geometry,refs,out)
        if number==7:figure7(data,geometry,out)
        if number==8:figure8(refs,data,geometry,out)
        if number==9:figure9(data,binding,out)
        print('Generated Figure',number,flush=True)
    contract=dict(binding=str(binding_path),binding_sha256=sha(binding_path),backend='Python/matplotlib',
        original_figures_1_3_unchanged=True,raw_motion_smoothed=False,main_figure_count=9,
        dense_background_grid_only_rasterized=False,global_signals='Separate methods; extrema-preserving display reduction.',
        trajectory_comparison='Prescribed local windows; equal aspect; shared scales; no global overplot.',
        figure4='Annotated source geometry and maximal straight runs, 10 mm grid, all34 angles; continued V/A/J and rest.',
        figure5='Three prescribed corners, overview and magnified transition, unsmoothed original positions.',
        figure6='135/90/45 degrees, requested steering/feed/preview then actual heading/curvature/speed.',
        figure7='All34 speed deficits; exact threshold intervals on separate method rows in two time halves; no heatmap.',
        figure8='Complete effects and recorded local intervention trajectories; failed prefix receives no complete time.',
        figure9='Resolution-checked gains, exact path/speed decomposition, three shared-model conditional searches.',
        intervention_traces={n:dict(path=data[n]['source'],sha256=sha(Path(data[n]['source']))) for n in ['Zero','Max']},
        visual_precedent='User-approved original manuscript and measured motion plots; no claimed external figure precedent.')
    write_json(out/'figure_contract.json',contract)
    write_json(out/'receipt.json',dict(files_sha256={p.name:sha(p) for p in out.glob('Figure_*.*')},
        source_binding_sha256=sha(binding_path),script_sha256=sha(Path(__file__)),visual_qa_complete=False))


if __name__=='__main__':main()
