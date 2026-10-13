"""Build the frozen S2 layer for the retained five-panel comparison."""
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from shapely.geometry import LineString, box
from study_io import BASE, ROOT, register, receipt, write_json, sha


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    original = ROOT/'论文项目/revision_20260913_preview_efficiency_v2/recursive_kcm_roundoff_continuation_v1/figures_v1'
    draw_path = original/'two_step_display_v1/draw.py'
    cache = original/'two_step_cache_adapter_v1/cache_v1/mixed_lengths_large'
    convert_path = ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources = [__file__, BASE/'s2_figure_layer_contract.md', draw_path, convert_path,
        BASE/'s2_frozen_command_new_geometry_v1/receipt.json',
        BASE/'s2_frozen_command_new_geometry_v1/results.json']
    sources.extend(cache/name for name in ['cache_receipt.json','cache_record.json',
                                         'source_binding.json','TS_source_data.npz'])
    stage = register('s2_figure_layer_v2',sources,
        dict(plotting_cache_reads=1, figures=1, svg_pdf_conversions=1,
             model_calls=0, physics_ticks=0, new_derivatives=0, full_audits=0),
        'Prepare the S2 layer from saved commands with the current Euclidean boundary; final joint overlay remains pending.')
    old = load_module('saved_s2_cache_reader',draw_path)
    source = old.bind_cache('mixed_lengths_large','TS')
    data = old.load_cache_arrays(source)
    geo = json.loads((BASE/'s2_frozen_command_new_geometry_v1/results.json').read_text(encoding='utf-8'))['results'][0]
    assert geo['success'] and geo['command_sha256']==source['binding']['source_trace_sha256']
    assert geo['complete_time_s']==source['cache']['complete_time_s']
    ref = data['reference_positions_mm']
    positions = data['positions_mm']
    region = LineString(ref).buffer(1.25,resolution=128)
    center = np.array(source['binding']['definition']['preview_detail_centers_mm'][0])
    zoom = np.r_[center[0]-10,center[0]+10,center[1]-10,center[1]+10]
    margin = .025*np.ptp(ref,axis=0).max()
    bounds = np.r_[ref[:,0].min()-margin,ref[:,0].max()+margin,
                   ref[:,1].min()-margin,ref[:,1].max()+margin]
    mpl.rcParams.update({'font.family':'Arial','font.size':8.5,'axes.titlesize':9,
        'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42,
        'path.simplify':False,'agg.path.chunksize':0})
    fig = plt.figure(figsize=(180/25.4,175/25.4))
    grid = fig.add_gridspec(3,2,height_ratios=[1.6,1,1],left=.11,right=.98,
                          top=.90,bottom=.08,hspace=.63,wspace=.40)
    axes = [fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),
            fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1]),fig.add_subplot(grid[2,:])]
    for i,(ax,lim) in enumerate(zip(axes[:2],[bounds,zoom])):
        clipped = region.intersection(box(lim[0],lim[2],lim[1],lim[3]))
        parts = list(clipped.geoms) if hasattr(clipped,'geoms') else [clipped]
        for part in parts:
            if part.is_empty or not hasattr(part,'exterior'): continue
            x,y = part.exterior.xy
            ax.fill(x,y,color='#edf3f8',zorder=0)
        boundary = region.boundary
        rings = list(boundary.geoms) if hasattr(boundary,'geoms') else [boundary]
        for ring in rings:
            xy = np.asarray(ring.coords)
            ax.plot(xy[:,0],xy[:,1],color='#829bab',ls='--',lw=.6,zorder=1)
        ax.plot(ref[:,0],ref[:,1],color='#474747',ls=':',lw=.7,zorder=2)
        if i==0:
            ax.plot(positions[:,0],positions[:,1],color='#d55e00',lw=.7,zorder=3)
            ax.add_patch(Rectangle((zoom[0],zoom[2]),20,20,fill=False,lw=.7))
            ax.annotate('B',center,xytext=(10,14),textcoords='offset points',
                        arrowprops=dict(arrowstyle='-',lw=.6))
            ax.plot(*ref[0],'^',ms=3,color='#262626')
            ax.plot(*positions[-1],'o',ms=3,mfc='white',mec='#d55e00')
        else:
            low, high = np.minimum(positions[:-1],positions[1:]), np.maximum(positions[:-1],positions[1:])
            indices = np.flatnonzero((high[:,0]>=lim[0])&(low[:,0]<=lim[1])&
                                    (high[:,1]>=lim[2])&(low[:,1]<=lim[3]))
            segments = np.stack([positions[indices],positions[indices+1]],axis=1)
            ax.add_collection(LineCollection(segments,colors=['#d55e00'],linewidths=1,zorder=3))
        ax.set(xlim=lim[:2],ylim=lim[2:],xlabel='X (mm)',ylabel='Y (mm)',aspect='equal')
        ax.xaxis.set_major_locator(MaxNLocator(3 if i==0 else 4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    checks=[]
    for ax,key,limit,label in zip(axes[2:],
        ['speed_mm_s','acceleration_norm_mm_s2','jerk_norm_mm_s3'],
        [100.,2000.,62500.],
        [r'$\|\mathbf{v}\|/V_{\max}$',r'$\|\mathbf{a}\|/A_{\max}$',r'$\|\mathbf{j}\|/J_{\max}$']):
        t,y = data['timestamp_s'],data[key]/limit
        line, = ax.plot(t,y,color='#d55e00',lw=.42)
        assert np.array_equal(line.get_ydata(),y,equal_nan=True)
        k = int(np.nanargmax(y))
        ax.plot(t[k],y[k],'D',ms=3,mfc='white',mec='#d55e00',mew=.65)
        ax.axhline(1,color='#38815d',ls='--',lw=.6)
        ax.set(xlim=(0,float(t[-1])*1.006),ylim=(-.025,max(1.,float(y[k]))*1.09),
               xlabel='Time (s)',ylabel=label)
        ax.grid(color='#e3e6e9',lw=.35,ls=':')
        checks.append(dict(channel=key,samples=len(t),peak=float(y[k]),
                           first_nan=bool(np.isnan(y[0])),end_time_s=float(t[-1])))
    for ax,letter,title in zip(axes,'abcde',['Full trajectory','Corner B','Velocity','Acceleration','Jerk']):
        ax.set_title(f'({letter}) {title}',loc='left',pad=5)
        ax.tick_params(length=2.5,width=.6)
    fig.legend(handles=[Line2D([],[],color='#d55e00',label='Two-step'),
                        Line2D([],[],color='#474747',ls=':',label='Reference'),
                        Line2D([],[],color='#829bab',ls='--',label='Tolerance boundary')],
               loc='upper center',bbox_to_anchor=(.54,.99),ncol=3,frameon=False,fontsize=8)
    svg = stage/'S2_layer.svg'
    fig.savefig(svg)
    plt.close(fig)
    converter = load_module('s2_layer_converter',convert_path)
    external = Path('D:/PPO_article_joint_runs/s2_figure_layer_v2')
    converter.convert(svg,external)
    shutil.copy2(external/'S2_layer.pdf',stage/'S2_layer.pdf')
    write_json(stage/'source_binding.json',dict(source=source['metadata'],
        command_sha256=geo['command_sha256'],complete_time_s=geo['complete_time_s'],
        tolerance_mm=1.25,boundary='Euclidean neighbourhood of the complete finite reference',
        full_bounds_mm=bounds.tolist(),local_bounds_mm=zoom.tolist(),checks=checks,
        actual_svg_to_pdf=True,svg_sha256=sha(svg),pdf_sha256=sha(stage/'S2_layer.pdf'),
        preview=str(external/'preview.png'),final_joint_overlay_pending=True))
    caption = ('Conventional two-step motion on the mixed-length path. (a) Complete trajectory and '
        '(b) the prescribed corner window B within the 1.25 mm contour-tolerance band. '
        '(c–e) Planar velocity, acceleration and jerk norms, normalized by their common limits. '
        'The curves include final braking and rest; complete traversal takes 323.367 s. '
        'Diamonds indicate the maxima of the saved 1 ms samples. The first unavailable jerk sample is left blank.')
    (stage/'caption.txt').write_text(caption+'\n',encoding='utf-8')
    receipt(stage,dict(status='S2_layer_exported_pending_visual_QA',plotting_cache_reads=1,
        figures=1,models=0,physics_ticks=0,new_derivatives=0,full_audits=0,
        final_joint_overlay_pending=True,goal_complete=False))
    print(external/'preview.png')


if __name__=='__main__':
    main()
