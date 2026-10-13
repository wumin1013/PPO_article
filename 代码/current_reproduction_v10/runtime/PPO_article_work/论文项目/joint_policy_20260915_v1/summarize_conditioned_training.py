"""Compare completed development evaluations using their saved commands."""
import json
import numpy as np
from study_io import BASE, register, receipt, write_json, sha, record_progress

GROUPS = [('v2 Joint','joint_mixed_curriculum_v2'),
          ('v2 Fixed 3','fixed3_mixed_recovery_v2'),
          ('v2 Fixed 8','fixed8_mixed_recovery_v2'),
          ('v3 Joint','joint_conditioned_training_v3'),
          ('v3 Fixed 3','fixed3_conditioned_training_v3'),
          ('v3 Fixed 4.6','fixed4p6_conditioned_training_v3')]


def distance_to_commands(point, xy):
    d = np.diff(xy,axis=0)
    fraction = np.clip(np.sum((point-xy[:-1])*d,axis=1)/
        np.maximum(np.sum(d*d,axis=1),1e-30),0.,1.)
    nearest = xy[:-1]+fraction[:,None]*d
    return float(np.linalg.norm(nearest-point,axis=1).min())


def diagnostics(reference, rows, actions):
    xy = np.vstack([reference[0],rows[:,:2]])
    speed = np.linalg.norm(rows[:,2:4],axis=1)
    tangents = np.diff(reference,axis=0)
    tangents /= np.linalg.norm(tangents,axis=1)[:,None]
    turns = np.rad2deg(np.arccos(np.clip(np.sum(tangents[:-1]*tangents[1:],axis=1),-1.,1.)))
    sharp = np.flatnonzero(turns>=30.)+1
    corner_rows = []
    for vertex in sharp:
        point = reference[vertex]
        local = np.linalg.norm(rows[:,:2]-point,axis=1)<10.
        corner_rows.append(dict(vertex_index=int(vertex),turn_deg=float(turns[vertex-1]),
            minimum_vertex_distance_mm=distance_to_commands(point,xy),
            speed_min_within_10mm=float(speed[local].min()) if local.any() else None))
    far = np.linalg.norm(rows[:,:2]-reference[0],axis=1)>15.
    far &= np.linalg.norm(rows[:,:2]-reference[-1],axis=1)>15.
    for point in reference[sharp]:
        far &= np.linalg.norm(rows[:,:2]-point,axis=1)>15.
    return dict(command_length_mm=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
        speed_quantiles_mm_s=np.quantile(speed,[.1,.5,.9]).tolist(),
        interior_away_from_sharp_corners_median_speed=float(np.median(speed[far])) if far.any() else None,
        feed_intent_quantiles_mm_s=np.quantile(actions[:,1],[.1,.5,.9]).tolist(),
        preview_quantiles_mm=np.quantile(actions[:,2],[.1,.5,.9]).tolist(),
        corners=corner_rows)


if __name__=='__main__':
    sources=[__file__,BASE/'study_io.py',BASE/'s2_line_development_v2/receipt.json',
        BASE/'s2_line_development_v2/results.json']
    for _,name in GROUPS:
        directory=BASE/name
        if not (directory/'receipt.json').exists():
            raise ValueError('Pending stage: '+name)
        sources += [directory/'receipt.json',directory/'initial_validation/summary.json',
                    directory/'final_validation/summary.json']
    stage=register('conditioned_training_analysis_v3',sources,
        dict(groups=6,phases_per_group=2,traces=144,models=0,physics_ticks=0),
        'Read completed initial/final development evidence and saved actions; no reruns.')
    records=[]
    for label,name in GROUPS:
        phases=[]
        for phase in ['initial','final']:
            directory=BASE/name/(phase+'_validation')
            summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            cases=[]
            for r in summary['results']:
                path=directory/(str(r['path'])+'.npz')
                with np.load(path) as data:
                    diag=diagnostics(data['reference'],data['rows'],data['actions'])
                cases.append(dict(r,diagnostics=diag,trace_sha256=sha(path)))
            full=all(r['success'] for r in cases)
            line_full=all(r['success'] for r in cases[:8])
            phases.append(dict(phase=phase,successes=sum(r['success'] for r in cases),
                full_time_sum_s=sum(r['completion_time_s'] for r in cases) if full else None,
                common_line_time_sum_s=sum(r['completion_time_s'] for r in cases[:8]) if line_full else None,
                results=cases))
        # A partial success set has no total complete time ranking.
        best=max(phases,key=lambda p:(p['successes'],-p['full_time_sum_s'] if p['full_time_sum_s'] is not None else float('-inf')))
        records.append(dict(method=label,stage=name,phases=phases,selected_phase=best['phase'],
            selection='development success count, then total complete time only when all twelve succeed'))
    s2=json.loads((BASE/'s2_line_development_v2/results.json').read_text(encoding='utf-8'))
    total=sum(r['completion_time_s'] for r in s2['results'])
    write_json(stage/'results.json',dict(methods=records,s2_line_total_s=total,
        model_calls=0,physics_ticks=0,independent_test=False))
    lines=['# 开发阶段全量结果','',
        '全部采用保存命令和完成收据，未重跑评价。开发路径参与过选择，因此不能当作独立测试。', '',
        '| 方法 | 阶段 | 完整成功 | 十二条总时间 / s | 共同八条直线路径 / s |',
        '|---|---|---:|---:|---:|']
    for group in records:
        for p in group['phases']:
            times=[f'{p[k]:.3f}' if p[k] is not None else '不可全量排名' for k in ['full_time_sum_s','common_line_time_sum_s']]
            lines.append(f'| {group["method"]} | {p["phase"]} | {p["successes"]}/12 | {times[0]} | {times[1]} |')
    lines += ['',f'常规S2在共同八条路径全部成功，总时间{total:.3f} s。固定方法使用其较好的已评估检查点，不能只引用退化的final。',
        '', 'results.json保留各路径速度分位数、前瞻分位数和各尖角最近距离；失败轨迹的形状仅描述已执行前缀。远离尖顶与提高效率要同时在允差内、按序、满足V/A/J和停稳的完整轨迹上判断。',
        '', '本阶段模型调用和物理步均为0；原训练和评价成本仍以各来源收据为准。']
    (stage/'开发结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_development_evidence_analysis',groups=6,
        traces=144,models=0,physics_ticks=0,goal_complete=False))
    record_progress('conditioned_training_analysis_v3',[stage/'receipt.json'],
        'v2/v3全量开发结果和保存轨迹诊断已汇总于conditioned_training_analysis_v3，按成功数及完整时间选择已评估检查点，固定退化和失败全部保留。下一步依据速度输出探测决定共同学习初始化或课程；不预写胜过S2及合理固定。封存新测试、多种子、长路径与原稿风格的20–25页论文/逐图报告/全PDF验收仍待完成。')
    print('\n'.join(lines))
