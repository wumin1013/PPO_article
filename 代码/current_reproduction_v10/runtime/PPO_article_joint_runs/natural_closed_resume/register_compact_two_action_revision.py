"""Record the user's amended scope without changing a model or old evidence."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT = Path(r'D:\PPO_article_joint_runs\natural_closed_resume')
R = OUT/'all34_dominance_research'
now = datetime.now(timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%S%fZ')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def item(p):
    return {'path':str(p),'sha256':sha(p)}

def write(p, v):
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

feasibility_path = R/'five_percent_complete_time_feasibility.json'
f = read(feasibility_path)
doc = BASE/'paper_evidence_review/紧凑状态与双动作消融修订分析.md'
assert not doc.exists()
doc.write_text(f'''# 紧凑状态与双动作消融修订分析

记录时间：{now.isoformat()}。本文件记录用户新要求与已完成的静态分析，不是新模型通过验收的声明。旧模型、旧论文和旧验收均作为上轮历史保留。

## 平滑性与出弯外摆

入弯与出弯必须具有可执行的速度、加速度和jerk，但轨迹不必始终在出射参考线内侧。允许带内的适度外展可能扩大有效转弯半径、降低曲率峰值及变化速度，从而减少降速。代价可能是增加行程、出现曲率反转及回正损失。几何外摆与动态不平滑不是同一件事。

因此撤回上一轮提出的一律禁止外摆、强制外摆≤0.01mm的拟定验收。新研究按真实曲率、向量V/A/J、横向误差与完整时间评估。保留外摆幅值、回正距离和局部形状报告，比较必须覆盖同一入弯开始至出弯恢复区间；不能只看顶点附近速度或裁掉不利恢复段。是否当前外摆产生收益，尚需受控候选比较，图像本身不构成因果证据。

## 完整时间5%的可达性

S2完整时间为{f['s2_complete_time_s']:.6f}s，缩短5%要求不超过{f['time_for_5percent_reduction_s']:.6f}s。

静态下界使用当前参考轨迹和实际geometry_spec.py中的有序有限门定义。对选定门i，其可通过位置包含于以该参考顶点为中心、半径epsilon/cos(turn_i/2)的球内。依次通过这些门的距离至少为相邻中心距离减去两端半径；由三角不等式逐段相加，再除以100mm/s得到下界。

只选40个强制门、用弦代替圆弧，并忽略其他门、加速度、jerk、最终行程及制动停稳，仍有：

- 路程下界：{f['gate_radius_distance_lower_bound_mm']:.6f}mm。
- 完整时间下界：{f['complete_time_lower_bound_s']:.6f}s。
- 即使进一步把每个选定门放宽为半径10mm球，时间下界仍为{f['relaxed_10mm_gate_ball_lower_bound_s']:.6f}s。

两个下界均高于5%目标。因此，在保留本案例、既有S2、100mm/s限速及完整有序覆盖时，任何策略都无法满足完整时间快5%的要求。下界不能当作已实现的最优时间；由较紧下界给出的最大可能收益上限约{f['optimistic_maximum_possible_reduction_percent']:.4f}%，实际可达上限还可能更低。

原始可复算证据：[{feasibility_path.name}](<{feasibility_path.as_posix()}>）。本次物理执行tick、策略调用和训练更新均为0。不得把局部过弯收益、亏损降幅或计算耗时改善替代完整时间5%；不得削弱S2来满足比例。

## 真正降维

用户已要求实际模型降维。当前72维模型另含474个因果记忆标量，不能将其保留在隐含包装层中却对外宣称小状态模型。新候选应使用一个明确的输入接口与精简控制器；必要的滤波状态或记忆若保留，须单独明确计数。

先复查已有compact_observation_candidate.py、event_observation_candidate.py与22维spatial22试验。它们属于历史候选或早期短任务试验，不是当前44.42m任务的降维成功证据。

建议从以下16个物理量的无神经循环记忆候选开始，再逐组移除和重新训练；16维是研究起点，不是预先宣称的最小维数：

| 状态组 | 维数 | 作用 |
|---|---:|---|
| 横向误差、航向误差sin/cos | 3 | 位置与方向跟踪 |
| 局部平面速度、加速度 | 4 | 描述执行状态及受jerk约束的运动惯性 |
| 下一几何事件距离、转角、当前参考曲率 | 3 | 描述转弯需求与圆弧 |
| 两个前方参考点的局部相对位置 | 4 | 描述局部形状及邻近事件 |
| 剩余路径距离、终段状态 | 2 | 正确进入完整制动与结束阶段 |

设计时须检验几何编码在事件切换处的连续性，不机械复用可能造成跳变的历史编码。公共环境可保留有序覆盖追踪器与物理约束执行器，但策略不得暗中查询未申报的输入、路径编号、全局时刻或原72维教师输出。

最小性表述应为“所比较候选中通过完整验收的最小状态”。先使用分组干预定位冗余，再通过真实重训练/明确标注的拟合检验；特征低方差或置零结果不能直接证明其可删除。默认三个独立种子，两动作和三动作采用对应种子、相当模型容量、相同训练/选择机会。每轮训练或仿真≤30分钟，开始前登记证据缺口、采用和停止条件。

## 去掉前瞻动作的双动作对照

新消融只有转向与进给两个输出，独立训练。保留与主方法相同的必要前方几何观测、奖励、公共KCM、限速、任务与成功条件，避免同时删掉前方信息而混淆动作维数与观测信息的作用。

双动作控制器直接形成方向与速度请求，不保留第三个可学习前瞻距离或隐藏自适应前瞻控制器。实现时需明确无前瞻动作下的方向解码：以当前局部参考方向为基准，由转向动作形成航向请求，不能仍调用基于固定距离目标点的LOS解码后只改名称。该解码差异应在方法中如实说明。

历史强Fixed保留为历史证据，不再替代用户指定的双动作重训练消融。双动作结果未产生前不能声称5%优势，其失败前缀没有完整时间。

## 正文与附录

延续用户上一轮修改要求：KCM算法/伪代码移附录；取消正文中独立PPO小节，正文只保留必要的学习来源和机制联系；网络/训练/优化详细参数移附录。图4b删除额外10mm文字标注并保留实际10mm网格。前三幅原图不改。新版模型通过相应验收后，实验图统一绑定新版数据，不混用旧模型形状。

## 当前有效状态

新要求未完成，实际降维与双动作重训练均未开始。完整时间5%在原有边界下已被保守下界排除，等待用户确定该冲突的处理方式；降维和对照接口设计可独立推进。未另换路径、未降低基线、未启动不可能目标的盲目训练。

记录和检查各自集中，复用已有输入候选与覆盖定义（KISS/DRY）；不增加无关实验和隐含状态（YAGNI）；观测、动作解码、物理执行和验收职责分离（SOLID）。所有历史均保留，未切换分支或提交推送。
''',encoding='utf-8')
history = R/'state_history'/stamp
history.mkdir(parents=True,exist_ok=False)
states = [BASE/'当前执行状态.json',OUT/'live_research_summary.json',R/'active_research_checkpoint.json']
for p in states:
    shutil.copy2(p,history/p.name)
    s = read(p)
    previous = {'state':item(history/p.name),'acceptance':s.get('authoritative_acceptance'),
                'pdf':s.get('final_pdf'),'delivery_index':s.get('delivery_index'),'paper_binding':s.get('current_paper_binding')}
    s.setdefault('historical_records',{})['before_compact_two_action_request'] = previous
    s.update({'updated_utc':now.isoformat(),'goal_complete':False,'method_frozen':False,
              'research_acceptance_passed':False,'final_manuscript_accepted':False,
              'current':'新要求分析中：实际降维、独立重训练双动作消融；外摆按动态与效率评价。原任务对S2完整时间快5%已被有序覆盖下界排除，需明确目标冲突处理。',
              'current_request_document':item(doc),'current_time_feasibility':item(feasibility_path),
              'current_request_status':'scope_conflict_identified; compact policy and two-action design pending',
              'authoritative_acceptance':None,'final_pdf':None,'delivery_index':None,
              'historical_final_pdf':previous['pdf'],'historical_delivery_index':previous['delivery_index'],
              'complete_times_scope':'Historical selected72-dimensional controller and its original comparators, not the requested compact/new-two-action experiment.',
              'new_compact_policy_verified':False,'new_two_action_controller_trained':False,
              'requested_primary_comparators':['S2','independently retrained steering/feed two-action controller without preview action'],
              'requested_complete_time_reduction_fraction':0.05,
              'blanket_no_outward_swing_requirement':False,
              'active_sessions':{},'active_validation_sessions':{},'running_candidate':None,
              'latest_revision_history':str(history),'next_steps':[
                  'Resolve complete-time5percent conflict without silently changing task, metric, or baseline.',
                  'Audit and rebuild a compact explicit observation/controller interface; report actual dimension and memory.',
                  'Define independent two-action retraining with matched information and fair selection opportunities.',
                  'Apply requested main-text/appendix and figure annotation revisions after model evidence is consistent.']})
    write(p,s)
print(json.dumps({'document':item(doc),'state_history':str(history),'new_requirements_complete':False,
                  'simulations_started':0,'models_modified':0},ensure_ascii=False,indent=2))
