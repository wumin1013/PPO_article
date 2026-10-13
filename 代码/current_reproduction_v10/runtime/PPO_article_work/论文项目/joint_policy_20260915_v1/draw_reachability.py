"""Saved-command illustration of the feasible inward-cutting behavior."""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely.geometry import LineString
from study_io import BASE,register,receipt

if __name__=='__main__':
    paths = [BASE/'preview_range_probe_v2'/f'angle155_speed35_D{distance}.npz' for distance in [3,8]]
    stage = register('inward_cut_illustration_v2_layout_fix',[__file__,*paths,BASE/'preview_range_probe_v2/receipt.json'],
        dict(figures=1,source_traces=2,physics_ticks=0,models=0),
        'Actual 155-degree constant-intent feasibility example; not an RL result. Original XY/velocity figure grammar retained.')
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes = plt.subplots(1,2,figsize=(180/25.4,86/25.4))
    fig.subplots_adjust(left=.08,right=.84,bottom=.16,top=.82,wspace=.26)
    records=json.loads((BASE/'preview_range_probe_v2/results.json').read_text(encoding='utf-8'))['runs']
    for ax,path,D in zip(axes,paths,[3,8]):
        record=next(r for r in records if r['angle_deg']==155 and r['speed']==35 and r['D']==D)
        T,gap=record['completion_time_s'],record['minimum_vertex_distance_mm']
        data = np.load(path)
        ref,rows = data['reference'],data['rows']
        boundary = np.asarray(LineString(ref).buffer(1.25,resolution=256).exterior.coords)
        ax.fill(boundary[:,0],boundary[:,1],color='#edf1f5')
        ax.plot(boundary[:,0],boundary[:,1],'--',color='#788da0',lw=.7)
        ax.plot(ref[:,0],ref[:,1],'-.',color='#4b4b4b',lw=.8)
        xy = np.vstack([ref[0],rows[:,:2]])
        collection = LineCollection(np.stack([xy[:-1],xy[1:]],axis=1),cmap='turbo',norm=Normalize(0,100),linewidth=1.6)
        collection.set_array(np.linalg.norm(rows[:,2:4],axis=1))
        ax.add_collection(collection)
        ax.scatter([0],[0],s=10,color='#4b4b4b',zorder=3)
        ax.set(xlim=(-11,2),ylim=(-2,5),aspect='equal',xlabel='X (mm)',ylabel='Y (mm)',
            title=f'D = {D} mm; complete time = {T:.3f} s\nMinimum distance to vertex: {gap:.3f} mm')
        ax.grid(ls=':',alpha=.2)
    colorbar = fig.add_axes([.87,.23,.014,.50])
    fig.colorbar(collection,cax=colorbar,label='Velocity (mm/s)')
    fig.suptitle('155° corner, 35 mm/s speed intent: both complete and feasible',fontsize=10)
    fig.savefig(stage/'inward_cut.png',dpi=240)
    fig.savefig(stage/'inward_cut.svg')
    receipt(stage,dict(status='saved_command_figure_generated_pending_visual_QA',figures=1,
        models=0,physics_ticks=0,learned_policy_result=False))
