"""Read saved pilot commands only; no environments or model execution."""
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from shapely.geometry import LineString
from study_io import BASE, ROOT, register, receipt, write_json, sha
from geometry_spec import Corridor

GROUPS = [('Initial','joint_cost_training_v1','initial_validation'),
          ('Joint policy','joint_pilot_continuation_v1','final_validation'),
          ('Fixed D = 3 mm','fixed3_joint_pilot_v1','final_validation'),
          ('D only','only_d_joint_pilot_v1','final_validation')]


def vertex_distance(xy):
    vectors = np.diff(xy,axis=0)
    fraction = np.clip(np.sum(-xy[:-1]*vectors,axis=1)/np.maximum(np.sum(vectors*vectors,axis=1),1e-30),0.,1.)
    return float(np.linalg.norm(xy[:-1]+fraction[:,None]*vectors,axis=1).min())


if __name__=='__main__':
    sources = [__file__,BASE/'pilot_figure_contract.md',BASE/'geometry_spec.py',BASE/'study_io.py']
    for _,run,tag in GROUPS:
        if not (BASE/run/'receipt.json').exists(): raise ValueError('Pilot has not finished: '+run)
        sources += [BASE/run/'receipt.json',BASE/run/tag/'summary.json']
        sources += [BASE/run/tag/(str(i)+'.npz') for i in range(4)]
    stage = register('pilot_analysis_v1',sources,dict(trace_files=16,figures=2,environment_steps=0,model_calls=0),
        'Compare all three pilot arms from saved data; no performance reruns.')
    results = []
    caches = {}
    for name,run,tag in GROUPS:
        summary = json.loads((BASE/run/tag/'summary.json').read_text(encoding='utf-8'))
        for index,result in enumerate(summary['results']):
            data = np.load(BASE/run/tag/(str(index)+'.npz'))
            reference,rows = data['reference'],data['rows']
            xy = np.vstack([reference[0],rows[:,:2]])
            corridor = Corridor(reference)
            contour = max(corridor.point_distance(point) for point in xy)
            speed = np.linalg.norm(rows[:,2:4],axis=1)
            local = np.linalg.norm(rows[:,:2],axis=1)<5.
            results.append(dict(method=name,path=index,angle_deg=[45.,90.,130.,155.][index],
                success=result['success'],time_s=result['completion_time_s'],
                minimum_vertex_distance_mm=vertex_distance(xy),peak_contour_mm=float(contour),
                corner_min_speed_mm_s=float(speed[local].min()) if local.any() else None,
                source=str((BASE/run/tag/(str(index)+'.npz')).relative_to(ROOT))))
            caches[name,index] = (reference,rows,result)
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.65,'legend.frameon':False})
    for index,angle in [(2,130),(3,155)]:
        fig,axes = plt.subplots(2,2,figsize=(180/25.4,145/25.4))
        fig.subplots_adjust(left=.09,right=.87,bottom=.1,top=.92,wspace=.30,hspace=.33)
        for panel,(ax,(name,_,_)) in enumerate(zip(axes.ravel(),GROUPS)):
            ref,rows,result = caches[name,index]
            band = LineString(ref).buffer(1.25,resolution=256)
            boundary = np.asarray(band.exterior.coords)
            ax.fill(boundary[:,0],boundary[:,1],color='#edf1f5',zorder=0)
            ax.plot(boundary[:,0],boundary[:,1],color='#788da0',ls='--',lw=.7)
            ax.plot(ref[:,0],ref[:,1],color='#4b4b4b',ls='-.',lw=.8)
            xy = np.vstack([ref[0],rows[:,:2]])
            segments = np.stack([xy[:-1],xy[1:]],axis=1)
            collection = LineCollection(segments,cmap='turbo',norm=Normalize(0,100),linewidth=1.6)
            collection.set_array(np.linalg.norm(rows[:,2:4],axis=1))
            ax.add_collection(collection)
            ax.scatter([0],[0],color='#4b4b4b',s=9,zorder=3)
            ax.set(xlim=(-7,2),ylim=(-1.8,6.2),xlabel='X (mm)',ylabel='Y (mm)',aspect='equal')
            ax.grid(alpha=.2,ls=':',lw=.5)
            label=f'{result["completion_time_s"]:.3f} s' if result['success'] else 'failed'
            ax.set_title(f'({chr(97+panel)}) {name}, {label}')
        cax = fig.add_axes([.90,.18,.022,.63])
        fig.colorbar(collection,cax=cax,label='Velocity (mm/s)')
        fig.suptitle(f'{angle}° corner: equal-budget development pilot',fontsize=10)
        stem = stage/f'corner_{angle}'
        fig.savefig(str(stem)+'.svg')
        fig.savefig(str(stem)+'.png',dpi=240)
        plt.close(fig)
    write_json(stage/'results.json',dict(rows=results,training_decisions_per_arm=10000,
        seeds=1,development_validation_paths=4,comparison_with_S2=False,
        unseen_generalization=False,trajectory_data_replayed=False))
    text = ['# 单种子先导结果','',
        '三个学习组各累计10,000个宏决策训练步。三动作组的前4,096步作为成本标定，其后5,904步原状态续接，未重跑已完成验证；另外两组从相同初始化来源训练。以下均为四条短开发轨迹，不是投稿级泛化证据。','',
        '| 方法 | 完整成功 | 完整用时总和 / s |','|---|---:|---:|']
    for name,_,_ in GROUPS:
        items = [r for r in results if r['method']==name]
        success = sum(r['success'] for r in items)
        total = f'{sum(r["time_s"] for r in items):.3f}' if success==4 else '不可全量排名'
        text.append(f'| {name} | {success}/4 | {total} |')
    text += ['', '曲线使用已保存命令，未调用环境或模型。离尖顶距离、轮廓距离和局部速度见results.json。尚需多个角连续出现的课程、长路径观测范围核查、合理固定D集合、独立测试与多种子。论文尚未进入最终定稿。']
    (stage/'先导结果.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    receipt(stage,dict(status='pilot_analysis_and_svg_generation_completed_pending_visual_QA',
        figures=2,models=0,physics_ticks=0,source_bindings=len(sources),
        all_inputs_completed=True,goal_complete=False))
    print('\n'.join(text),flush=True)
