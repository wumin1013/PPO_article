"""Read saved validation commands only; never alter frozen evaluation labels."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

import revision_plot as common
ROOT=common.ROOT
sys.path.insert(0,str(ROOT/'PPO_project'))
from src.utils.revision_metrics import _reference_witnesses, ordered_projection

BASE=ROOT/'论文项目/revision_20260908'
OUT=BASE/'preview_direct_gain/ordered_witness_diagnostic'
REFERENCE=ROOT/'PPO_project/data/preview_v2/paths/dense_corners_validation.csv'
INPUTS={
    'fixed_mid_pilot':BASE/'preview_direct_gain/pilot/fixed_mid_seed101/evaluation_validation_nominal/dense_corners_validation',
    'constant_action_diagnostic':BASE/'preview_v2/guidance_gain_probe/blend_0.015_D2.3_dense_corners_validation',
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_on_command(witness,positions,timestamps,minimum_time=0.):
    starts=positions[:-1].copy()
    vectors=np.diff(positions,axis=0)
    start_t=timestamps[:-1].copy()
    valid=timestamps[1:]>=minimum_time
    fractions=np.clip((minimum_time-start_t)/np.maximum(np.diff(timestamps),1e-20),0,1)
    starts+=fractions[:,None]*vectors
    start_t+=fractions*np.diff(timestamps)
    vectors=positions[1:]-starts
    lengths2=np.sum(vectors*vectors,axis=1)
    projection=np.clip(np.sum((witness-starts)*vectors,axis=1)/np.maximum(lengths2,1e-24),0,1)
    points=starts+projection[:,None]*vectors
    distances=np.linalg.norm(witness-points,axis=1)
    distances[~valid]=np.inf
    index=int(np.argmin(distances))
    return dict(distance_mm=float(distances[index]),command_segment_end_index=index+1,
        timestamp_s=float(start_t[index]+projection[index]*(timestamps[index+1]-start_t[index])),
        point_mm=points[index].tolist())


def ordered_match_times(witnesses,positions,timestamps,radius):
    """Expose matching times using the frozen segment-interval rule for diagnostics."""
    frontier=0
    matches=[]
    for k in range(len(positions)):
        a=positions[max(0,k-1)];vector=positions[k]-a
        length2=float(vector@vector);earliest=0.
        while frontier<len(witnesses):
            relative=witnesses[frontier]-a
            if length2<=1e-24:
                if np.linalg.norm(relative)>radius+1e-12: break
            else:
                center=float(relative@vector/length2)
                perpendicular2=float(relative@relative-center*center*length2)
                discriminant=radius*radius-max(0.,perpendicular2)
                if discriminant < -1e-12: break
                width=np.sqrt(max(0.,discriminant)/length2)
                lower,upper=max(0.,center-width),min(1.,center+width)
                earliest=max(earliest,lower)
                if earliest>upper+1e-12: break
            matches.append(dict(witness_index=frontier,command_segment_end_index=k,
                timestamp_s=float(timestamps[max(0,k-1)]+earliest*(timestamps[k]-timestamps[max(0,k-1)])),
                command_point_mm=(a+earliest*vector).tolist()))
            frontier+=1
    return matches


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    common.style()
    reference=np.loadtxt(REFERENCE,delimiter=',',skiprows=1)
    witness,arcs=_reference_witnesses(reference,.25)
    sources={REFERENCE.relative_to(ROOT).as_posix():sha(REFERENCE)}
    results={};traces={};csv_rows=[]
    for name,stem in INPUTS.items():
        for suffix in ['.npz','.json']:
            path=Path(str(stem)+suffix);sources[path.relative_to(ROOT).as_posix()]=sha(path)
        raw=json.loads(Path(str(stem)+'.json').read_text(encoding='utf-8'))
        z=np.load(Path(str(stem)+'.npz'))
        columns=dict(zip(z['columns'].tolist(),z['data'].T))
        xy=np.c_[columns['x_mm'],columns['y_mm']];t=columns['timestamp_s'];traces[name]=(xy,t)
        strict=ordered_projection(reference,xy,tolerance_mm=1.25)
        assert np.isclose(strict['ordered_progress'][-1],raw['ordered_progress'],atol=1e-12)
        assert np.isclose(strict['contour_error_mm'].max(),raw['contour_error_max_mm'],atol=1e-12)
        matches=ordered_match_times(witness,xy,t,1.125)
        assert len(matches)==strict['covered_witness_count']
        assert np.all(np.diff([m['timestamp_s'] for m in matches])>=-1e-12)
        first=len(matches)
        nearest=[nearest_on_command(w,xy,t) for w in witness]
        for k,item in enumerate(nearest):
            csv_rows.append(dict(trace=name,witness_index=k,arc_mm=float(arcs[k]),x_mm=float(witness[k,0]),y_mm=float(witness[k,1]),
                minimum_command_distance_mm=item['distance_mm'],nearest_time_s=item['timestamp_s'],
                strict_covered=k<first,strict_match_time_s=matches[k]['timestamp_s'] if k<first else None))
        sensitivity=[]
        for spacing in [.25,.1,.05,.01]:
            diagnostic=ordered_projection(reference,xy,tolerance_mm=1.25,witness_spacing_mm=spacing)
            sensitivity.append(dict(scope='Diagnostic sensitivity only; original result and failure label unchanged',
                spacing_mm=spacing,matching_radius_mm=diagnostic['witness_match_radius_mm'],
                witness_count=diagnostic['witness_count'],coverage_complete=diagnostic['ordered_coverage_complete'],
                final_ordered_progress=float(diagnostic['ordered_progress'][-1])))
        first_uncovered=None
        if first<len(witness):
            previous_time=matches[-1]['timestamp_s'] if matches else 0.
            first_uncovered=dict(index=first,one_based_number=first+1,arc_mm=float(arcs[first]),point_mm=witness[first].tolist(),
                previous_accepted_witness=matches[-1] if matches else None,nearest_any_time=nearest[first],
                nearest_after_previous_match=nearest_on_command(witness[first],xy,t,previous_time),
                within_physical_tolerance=nearest[first]['distance_mm']<=1.25,
                outside_frozen_matching_radius=nearest[first]['distance_mm']>1.125)
        results[name]=dict(original_success=raw['success'],original_failure_reasons=raw['failure_reasons'],
            frozen_progress=float(strict['ordered_progress'][-1]),frozen_witness_count=len(witness),covered_witness_count=first,
            contour_error_max_mm=float(strict['contour_error_mm'].max()),terminal_time_s=float(t[-1]),
            first_uncovered=first_uncovered,sensitivity=sensitivity,
            frozen_witness_max_nearest_command_distance_mm=max(item['distance_mm'] for item in nearest),
            strict_ordered_matches=matches)
    missed=results['fixed_mid_pilot']['first_uncovered']
    point=np.array(missed['point_mm'])
    teacher_xy,teacher_t=traces['constant_action_diagnostic']
    results['same_failed_witness_constant_action_distance']=nearest_on_command(point,teacher_xy,teacher_t)
    sources['PPO_project/src/utils/revision_metrics.py']=sha(ROOT/'PPO_project/src/utils/revision_metrics.py')
    sources['PPO_project/src/utils/ordered_coverage.py']=sha(ROOT/'PPO_project/src/utils/ordered_coverage.py')
    payload=dict(scope='Saved dense validation only, no policy execution, no training, no test read, no metric or label mutation',
        command='python PPO_project/tools/preview_ordered_witness_diagnosis.py',generator_sha256=sha(Path(__file__)),source_hashes=sources,
        one_sided_tolerance_mm=1.25,original_spacing_mm=.25,original_matching_radius_mm=1.125,
        radius_definition='tolerance minus half of declared witness spacing; not a universal 0.9 tolerance rule',
        interpretation='This fixed-mid failure is sensitive to the conservative 0.125 mm witness margin. Its first blocked witness is physically within 1.25 mm in proper command order, and denser ordered witness certificates complete. The retained original label remains failed; this evidence does not establish a true branch skip or continuous Frechet equivalence.',
        results=results)
    (OUT/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (OUT/'witness_distances.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    fig,axes=plt.subplots(2,1,figsize=(7.2,4.8),gridspec_kw={'height_ratios':[1,2.5]},constrained_layout=True)
    for ax in axes:
        ax.plot(reference[:,0],reference[:,1],color='#bbbbbb',lw=2,label='Reference')
        for name,color,label in [('constant_action_diagnostic','#1f77b4','Constant action'),('fixed_mid_pilot','#d78332','Fixed-mid pilot')]:
            xy,_=traces[name];ax.plot(xy[:,0],xy[:,1],color=color,lw=1,label=label)
        ax.scatter(*point,color='#9a4a43',marker='x',s=35,zorder=5,label='First blocked witness')
        ax.set(xlabel='x (mm)',ylabel='y (mm)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15,ls=':')
    axes[0].legend(fontsize=6,loc='lower center',bbox_to_anchor=(.5,1.06),ncol=4)
    for radius,color,style in [(1.125,'#9a4a43','--'),(1.25,'#777777',':')]:
        axes[1].add_patch(Circle(point,radius,fill=False,color=color,ls=style,lw=.8))
    near=missed['nearest_any_time'];near_xy=np.array(near['point_mm'])
    axes[1].plot([point[0],near_xy[0]],[point[1],near_xy[1]],color='#9a4a43',lw=.8)
    axes[1].scatter(*near_xy,color='#d78332',s=18,zorder=6)
    axes[1].set(xlim=(point[0]-1.45,point[0]+1.45),ylim=(point[1]-1.45,point[1]+1.45),title='1.14642 mm: between both radii')
    axes[1].text(1.04,.80,'Dashed: frozen radius 1.125 mm\nDotted: physical tolerance 1.25 mm',
        fontsize=7,transform=axes[1].transAxes,va='top')
    for letter,ax in zip('ab',axes):ax.text(-.12,1.04,letter,transform=ax.transAxes,weight='bold')
    common.save(fig,'ordered_witness_diagnostic')
    previous=missed['previous_accepted_witness'];teacher=results['same_failed_witness_constant_action_distance']
    lines=['# 顺序覆盖保守余量诊断','','只读已保存的 dense validation 指令；未执行新策略、未读取测试集，也未修改任何失败标签或冻结评价代码。','',
        f"固定中前瞻 pilot 原判定失败；原顺序进度 {results['fixed_mid_pilot']['frozen_progress']:.9f}，全局最大轮廓误差 {results['fixed_mid_pilot']['contour_error_max_mm']:.9f} mm。",
        f"首个未覆盖见证点为零基索引 {missed['index']}（第 {missed['one_based_number']} 个），参考弧长 {missed['arc_mm']:.9f} mm，坐标 {missed['point_mm']}。",
        f"前一个见证点在 {previous['timestamp_s']:.9f} s 已按序匹配。未覆盖点到完整输出折线的最小距离为 {near['distance_mm']:.9f} mm，时刻 {near['timestamp_s']:.9f} s；最小值位于前一个见证点匹配之后。",
        f"常动作对照到相同见证点的最小距离为 {teacher['distance_mm']:.9f} mm。",'',
        '1.125 mm 来自允差 1.25 mm 减去见证点间隔 0.25 mm 的一半；本例恰好等于 0.9 倍允差，并非固定比例定义。该余量为相邻见证点之间的覆盖提供保守裕量，不能将其自动等同于物理轮廓允差。','',
        '| 诊断见证点间隔/mm | 匹配半径/mm | fixed-mid顺序覆盖 | 常动作顺序覆盖 |','|---:|---:|---|---|']
    for fixed,constant in zip(results['fixed_mid_pilot']['sensitivity'],results['constant_action_diagnostic']['sensitivity']):
        lines.append(f"| {fixed['spacing_mm']:g} | {fixed['matching_radius_mm']:g} | {fixed['coverage_complete']} | {constant['coverage_complete']} |")
    lines += ['', '保持同一 1.25 mm 物理允差，增加见证点密度并相应减小保守扣除量后，保存轨迹可按相同顺序匹配规则完整覆盖。该案例支持“固定离散证书余量导致保守失败”的解释，尚无证据将其写为真实跨分支跳段。它也不等价于精确连续 Fréchet 求解或机床约束保证。',
        '', '冻结结果和训练奖励保持原样；若后续改变覆盖离散精度，应先冻结新的共同评价定义，并对所有方法共同重评、必要时重训，不能只替失败方法更改统计口径。', '',
        '原始 JSON/NPZ、参考路径及评价源码哈希在 summary.json；每个原见证点的最小折线距离和严格匹配时刻在 witness_distances.csv；矢量图由本脚本自动生成。', '']
    (OUT/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(missed,ensure_ascii=False,indent=2))
    print(OUT/'summary.json')


if __name__=='__main__':main()
