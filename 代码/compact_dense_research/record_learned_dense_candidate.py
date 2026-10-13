"""Persist the new learned candidate separately from its diagnostic teacher."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pilot=ROOT/'historical_fitted_residual_ppo_seed922302_pilot'
    fine=ROOT/'historical_fitted_residual_ppo_seed922302_half_ms'
    result=read(pilot/'eval_0016.json');refined=read(fine/'result.json')
    assert result['success'] and refined['success']
    s2=read(ROOT/'dense_s2_initial/result.json');s2_fine=read(ROOT/'dense_s2_half_ms/result.json')
    m=read(pilot/'all84_metrics_0016.json');b=read(ROOT/'dense_s2_initial/all84_metrics.json')
    training=read(pilot/'result.json')
    now=datetime.now(timezone.utc)
    next_steps=[
        'Re-evaluate this same frozen checkpoint on the original full long case; clustered-turn corrections must preserve all34 results and cruise. This is candidate checking, not method freeze.',
        'Construct a genuinely two-output steering/feed counterpart with the same inherited-control and execution scope, declared finite distance selection and learning budget; do not treat the unrelated14-input baseline as a controlled ablation.',
        'Check learned action/path mechanism, independent repetition scope, allowed/no-swing comparison and actual observation reduction before acceptance.',
        'Keep predictive-search teacher, offline fitting and subsequent PPO costs separate. Only after whole-goal acceptance edit final experiments on the unique18-page source.']
    binding=dict(updated_utc=now.isoformat(),checkpoint=str(pilot/'checkpoint_0016.pt'),
        checkpoint_sha256=sha(pilot/'checkpoint_0016.pt'),nominal_result=result,refined_result=refined,
        nominal_result_sha256=sha(pilot/'eval_0016.json'),refined_result_sha256=sha(fine/'result.json'),
        time_reduction_percent_1ms=100*(1-result['complete_time_s']/s2['complete_time_s']),
        time_reduction_percent_half_ms=100*(1-refined['complete_time_s']/s2_fine['complete_time_s']),
        resolution_time_change_s=refined['complete_time_s']-result['complete_time_s'],
        all84_deficit_no_worse_than_s2=sum(x['speed_deficit_mm']<=y['speed_deficit_mm']+1e-8 for x,y in zip(m['corners'],b['corners'])),
        path_length_time_contribution_s=(b['executed_length_mm']-m['executed_length_mm'])/100.,
        speed_deficit_time_contribution_s=(b['speed_deficit_mm']-m['speed_deficit_mm'])/100.,
        training=training,online_prediction_search=False,goal_complete=False,method_frozen=False,
        inherited_observation_dimension=72,residual_observation_dimension=17,
        independent_full_method_training=False,next_steps=next_steps)
    evidence=ROOT/'learned_dense_candidate_20260922.json'
    assert not evidence.exists()
    evidence.write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=ROOT/'learned_dense_candidate_20260922.md'
    report.write_text(f'''# 新增学得修正候选：仍未完成整体验收

本轮从已核验的历史最佳控制器出发，先用有限预测搜索找到完整可行轨迹，再作明确记录的离线拟合和16次真实PPO更新。最终选定第16次预定评估检查点；执行时不使用在线预测搜索。

| 同一密集案例 | 1ms | 0.5ms |
|---|---:|---:|
| 历史策略＋学得三动作修正 | 15.362s | 15.380s |
| 未改动S2 | 19.957s | 19.9565s |
| 完整时间降低 | {binding['time_reduction_percent_1ms']:.3f}% | {binding['time_reduction_percent_half_ms']:.3f}% |

两次均完整通过84个有序门、1.25mm解析欧氏允差、向量V/A/J和至少20ms物理停稳；时间细化差18ms。最大解析距离分别1.217011mm、1.178050mm。0.5ms通过的停稳审核跨度明确为20ms、41个采样点。该结果尚不等于整个研究已经验收。

1ms净节约4.595s；执行路径长度贡献-0.015613s，速度亏损贡献+4.610613s，说明收益主要来自减少低速损失。新案例68/84角速度亏损不高于S2，≥99mm/s时间0.230s；不能宣称新84角全部最佳或所有巡航指标最佳。原长路径与34角仍须用这个同一新增模型重新完整验证。此前稀疏记忆版本的444.302s及完全相同轨迹属于单独完成的结构核验，不能默默当成本模型的新长路径结果。

训练类别和选择路径分别保留：预测搜索为模型诊断，离线拟合检查点200/600/1600均未完整成功（62/6/16门）；选其中200轮检查点作为已披露的微调起点。随后16次PPO更新共16,384个新决策、57个训练回合，采样中7次成功；固定模型预定评估第4/8/12次仍失败，第16次首次完整成功。训练回合成功不代替固定模型评估。

17维仅是新增修正网络输入，整个方法仍继承72维历史控制器与历史记忆；不是整体17维或独立从零训练。该单个条件训练种子也不构成三个独立完整方法重复。预测诊断、拟合、PPO和步长检查各有独立预算、源码快照、结果及收据，均低于每轮30分钟。

后续：同模型原长路径完整验收；同一继承控制范围下真正两输出的公平消融；观察/记忆精简、重复性与外摆比较。原有14维两动作25.334s结果继续保留，但不将其冒充同结构、同历史预算的受控消融。固定前瞻不扩大为无限优化任务。

项目、分支、原18页稿和前三图保持不变。当前学得候选尚未冻结，未进行最终论文写作验收。所有精确路径和哈希见同名JSON；先前预测诊断图也明确绑定先前预测方案，未换成新PPO模型数据。
''',encoding='utf-8')
    states=[BASE/'当前执行状态.json',OUT/'live_research_summary.json',OUT/'all34_dominance_research/active_research_checkpoint.json',ROOT/'current_progress.json']
    backup=ROOT/'state_history'/now.strftime('%Y%m%dT%H%M%S%fZ');backup.mkdir(parents=True)
    backups=[]
    for path in states:
        raw=path.read_bytes();(backup/path.name).write_bytes(raw);backups.append(dict(original=str(path),sha256=hashlib.sha256(raw).hexdigest()))
        value=json.loads(raw)
        assert not value.get('active_stages')
        value.update(updated_utc=now.isoformat(),current='新学得三动作修正在密集案例1ms/0.5ms均完整成功15.362/15.380s，相对S2降低23.025%/22.932%，执行无在线预测搜索。仍待同模型原长路径、真正两动作公平消融、重复性和降维验收，未冻结。',
            latest_learned_dense_candidate=str(evidence),latest_learned_dense_report=str(report),
            goal_complete=False,method_frozen=False,research_acceptance_passed=False,final_manuscript_accepted=False,final_pdf=None,
            next_steps=next_steps,verified_active_exec_session=None,verified_active_exec_sessions={},
            latest_state_backup_receipt=str(backup/'receipt.json'))
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (backup/'receipt.json').write_text(json.dumps(dict(backups=backups,evidence=str(evidence),goal_complete=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(report),goal_complete=False,active_stages=[]),ensure_ascii=False))


if __name__=='__main__':main()
