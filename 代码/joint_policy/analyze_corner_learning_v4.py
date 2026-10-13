"""Read fixed 30k traces and derive a restricted circular-corner reference."""
import json
import math
import numpy as np
from study_io import BASE,register,receipt,write_json
from summarize_conditioned_training import distance_to_commands


if __name__=='__main__':
    groups=[('v3 fixed4.6 final','fixed4p6_conditioned_training_v3','final_validation'),
        ('v4 joint 30k','joint_curriculum_training_v4','validation_30000'),
        ('v4 fixed3 30k','fixed3_curriculum_training_v4','validation_30000'),
        ('v4 fixed4.6 30k','fixed4p6_curriculum_training_v4','validation_30000')]
    sources=[__file__,BASE/'summarize_conditioned_training.py']
    for _,name,phase in groups:
        sources += [BASE/name/phase/'summary.json',BASE/name/phase/'3.npz']
    stage=register('corner_learning_diagnosis_v4',sources,
        dict(saved_traces=4,summary_files=4,analytic_turns=4,models=0,physics_ticks=0),
        'Explain acute-corner failure and the permitted inward cut using saved commands and a restricted tangent-circle calculation; no new controller runs.')
    circle_rows=[]
    for degrees in [50.,95.,125.,150.]:
        half=math.radians(degrees)/2
        radius=1.25/(1-math.cos(half))
        tangent_offset=radius*math.tan(half)
        nearest_vertex=1.25/math.cos(half)
        speed=min(100.,math.sqrt(2000.*radius),(62500.*radius*radius)**(1/3))
        assert abs(radius*(1-math.cos(half))-1.25)<1e-12
        assert abs(radius*(1/math.cos(half)-1)-nearest_vertex)<1e-12
        circle_rows.append(dict(turn_deg=degrees,radius_mm=radius,tangent_offset_mm=tangent_offset,
            nearest_vertex_mm=nearest_vertex,constant_speed_circle_vaj_ceiling_mm_s=speed,
            assumes_tangent_circle_and_sufficient_adjacent_lengths=True,
            curvature_transition_or_full_time_certificate=False))
    observed=[]
    for label,name,phase in groups:
        directory=BASE/name/phase
        results=json.loads((directory/'summary.json').read_text(encoding='utf-8'))['results']
        with np.load(directory/'3.npz') as data:
            ref,rows,actions=data['reference'],data['rows'],data['actions']
        point=ref[1]
        distances=np.linalg.norm(rows[:,:2]-point,axis=1)
        nearby=distances<=10.
        action_index=np.minimum(np.arange(len(rows))//20,len(actions)-1)
        held=actions[action_index]
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        incoming=rows[:,0]<=point[0]
        inward=np.flatnonzero((rows[:,1]>.1)&incoming)
        start=rows[inward[0],:2] if len(inward) else None
        observed.append(dict(label=label,success=results[3]['success'],reason=results[3]['reason'],
            complete_time_s=results[3]['completion_time_s'],prefix_elapsed_s=results[3]['elapsed_s'],
            full_set_successes=sum(r['success'] for r in results),
            prefix_nearest_vertex_mm=distance_to_commands(point,np.vstack([ref[0],rows[:,:2]])),
            speed_near_corner_quantiles=np.quantile(speed[nearby],[.1,.5,.9]).tolist(),
            intended_feed_near_corner_quantiles=np.quantile(held[nearby,1],[.1,.5,.9]).tolist(),
            preview_near_corner_quantiles=np.quantile(held[nearby,2],[.1,.5,.9]).tolist(),
            steering_degrees_near_corner_quantiles=np.rad2deg(np.quantile(held[nearby,0],[.1,.5,.9])).tolist(),
            incoming_distance_at_y0p1_mm=float(point[0]-start[0]) if start is not None else None,
            partial_failure_shape_is_not_completed_transition=not results[3]['success']))
    write_json(stage/'results.json',dict(tangent_circle_reference=circle_rows,observations=observed,
        controller_rule_added=False,full_path_optimality_claimed=False))
    text=['# 急弯形状与学习诊断','',
        '“在允差内远离尖顶”已经由新的欧氏允差和有向门任务允许。它不是要求轨迹绕到外侧，也不要求经过原尖顶。当前训练是否掌握这种形状，要同时核对完整可行性、局部运动与总时间。','',
        '对孤立拐角，若仅考虑同时与两侧直线相切的圆弧，转向角为θ、允差为ε，则R≤ε/(1−cos(θ/2))，切入距离为R tan(θ/2)，圆弧距尖顶最近距离为R[sec(θ/2)−1]。在相邻线段足够长时，圆弧向内扩展可以同时减小曲率、缩短这类轨迹的长度。该结论只针对这一圆弧族；连接处还需要曲率变化过程，不能把圆弧匀速上限当成完整jerk可行轨迹。','',
        '| 转向角 | 满允差圆弧半径/mm | 切入距离/mm | 最近尖顶距离/mm | 匀速圆弧V/A/J上限/(mm/s) |',
        '|---:|---:|---:|---:|---:|']
    for r in circle_rows:text.append(f'| {r["turn_deg"]:.0f}° | {r["radius_mm"]:.3f} | {r["tangent_offset_mm"]:.3f} | {r["nearest_vertex_mm"]:.3f} | {r["constant_speed_circle_vaj_ceiling_mm_s"]:.3f} |')
    text += ['', '150°单角已有命令如下。失败行的距离与速度只描述已执行前缀，不代表完成了一次更优过弯。', '',
        '| 检查点 | 十二条成功数 | 150°成功 | 近角执行速度中位数 | 近角速度意图中位数 | 近角D中位数 | 距尖顶最近距离 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in observed:text.append(f'| {r["label"]} | {r["full_set_successes"]}/12 | {r["success"]} | {r["speed_near_corner_quantiles"][1]:.3f} | {r["intended_feed_near_corner_quantiles"][1]:.3f} | {r["preview_near_corner_quantiles"][1]:.3f} | {r["prefix_nearest_vertex_mm"]:.3f} |')
    text += ['',
        '这些证据把目标进一步落实为学习问题：策略需要在急弯之前共同调整速度、转向和前瞻，不能仅提高全程速度。圆弧计算用于解释尺度，没有作为逐角D表、速度上限表或执行规则加入控制器。当前四份保存轨迹只能定位现象，不能单独证明某个奖励或超参数是唯一原因。', '',
        '下一步继续读取已经登记的训练检查点；如需后继，优先比较共同的失败进度信用改进，并保留较强固定模型。固定D也要保留转向和速度学习权限；不能用固定D失败时间或成功子集时间制造完整优势。']
    (stage/'急弯学习诊断.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    receipt(stage,dict(status='saved_acute_corner_diagnosis_and_circle_reference_completed',
        saved_traces=4,analytic_turns=4,models=0,physics_ticks=0,controller_rule_added=False))
    print('\n'.join(text))
