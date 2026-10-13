"""Archive this bounded round's result without replacing a successful incumbent."""
import json
import shutil
from datetime import datetime, timezone
from evaluate_historical_dense import ROOT,sha,write

STAGE=ROOT/'direct14_multicase_ppo_seed922302_round1'


def main():
    result=json.loads((STAGE/'result.json').read_text(encoding='utf-8'))
    receipt=json.loads((STAGE/'receipt.json').read_text(encoding='utf-8'))
    summary=json.loads((STAGE/'evidence_summary.json').read_text(encoding='utf-8'))
    cruise=json.loads((STAGE/'straight_cruise_diagnostic.json').read_text(encoding='utf-8'))
    critic=json.loads((STAGE/'critic_diagnostic.json').read_text(encoding='utf-8'))
    assert receipt['exit_code']==0 and receipt['sources_unchanged'] and receipt['wall_s']<=1800.
    assert result['ppo_updates_this_round']==16 and result['new_decisions']==32768
    rows=['| 模型 | 原长路径完整时间 | 密集案例完整时间 |',
          '|---|---:|---:|','| 14维离线拟合初始化 | 445.072 s | 15.254 s |']
    checks=[]
    for record in result['scheduled_evaluations']:
        r=record['results'];values=[]
        for case in ['original','dense']:
            d=r[case];values.append(f"{d['complete_time_s']:.3f} s" if d['success'] else '完整失败')
            checks.append(dict(update=record['update'],case=case,success=d['success'],time_s=d['complete_time_s'],
                gates=f"{d['geometry']['gates_passed']}/{d['geometry']['gate_count']}",
                vector_kinematic_pass=d['audit']['kinematic_pass'],physical_rest_pass=d['audit']['rest_pass'],
                analytic_corridor_pass=d['analytic']['analytic_corridor_pass'],
                maximum_distance_mm=d['analytic']['maximum_evaluated_distance_mm']))
        rows.append(f"| PPO第{record['update']}次更新 | {values[0]} | {values[1]} |")
    report=dict(updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False,method_frozen=False,
        complete_checks=checks,round_wall_s=receipt['wall_s'],adopted=result['adopted'],
        retained_compact_checkpoint=result['retained_checkpoint'],retained_compact_checkpoint_sha256=sha(result['retained_checkpoint']),
        initialization_is_offline_fit=True,retained_compact_ppo_updates=0 if not result['adopted'] else None,
        tested_ppo_updates=16,new_decisions=32768,training_seed=922302,independent_repetitions=1,
        dense_retained_advantage_percent={'vs_S2_1ms':100*(1-15.254/19.957),'vs_two_action_1ms':100*(1-15.254/15.863),
            'vs_S2_half_ms':100*(1-15.271/19.9565),'vs_two_action_half_ms':100*(1-15.271/15.902)},
        two_action_scope='Existing strong historical two-output reference; matched14input two-output retraining is still required if compact is selected.',
        goal_gaps=['Dense gain versus credible two-action reference is below5percent',
            'Direct14 original34corner and strong-baseline time requirements remain unfulfilled',
            'No measured PPO improvement of compact model','Matched compact ablation and independent repetitions remain incomplete'],
        next_research='Do not extend the same setup blindly. Diagnose value-target scale and ensure full long-case training coverage; a new bounded round requires explicit preregistration. Retain actor initialization.',
        implementation_scope='Reused common physical runtime and PPO objective; added independent rollout-boundary support and separate bounded orchestration/diagnostics; no policy input expansion.',
        manuscript_edited=False,original_figures_edited=False,
        evidence_sha256={str(STAGE/name):sha(STAGE/name) for name in ['registration.json','receipt.json','result.json',
            'evidence_summary.json','straight_cruise_diagnostic.json','critic_diagnostic.json','numerical_checks.json']})
    write(STAGE/'verified_round_report.json',report)
    lines=['# 14维直接策略：双案例PPO微调结果','',
        f"本轮完成16次真实PPO更新、32768个新决策样本，训练与完整仿真墙钟共{receipt['wall_s']/60:.2f}分钟。",'',*rows,'',
        '四次完整检查均通过有序覆盖、1.25 mm解析允差、平面向量V/A/J约束和物理停稳。' if all(x['success'] for x in checks) else '完整检查结果见绑定JSON；失败前缀不计作完整时间。','',
        '**本轮未采用新检查点，保留14维离线拟合初始化，历史最好复合主方法保持原状。**' if not result['adopted'] else '本轮候选采用状态见绑定JSON；研究总体尚未验收。','',
        '两次密集案例检查均退步，因此未触发额外0.5 ms测试。原始成功初始化的既有0.5 ms结果为15.271 s。','',
        '原长路径第16次更新多出的1.759 s中，约1.748 s来自34个角点窗口之外的速度亏损。直线内部平均速度由99.878降为99.480 mm/s，请求进给同步降低；全程≥99 mm/s巡航时间减少6.282 s。这支持巡航退步的解释。','',
        '本轮原长路径新采样覆盖到32.49 m，约全长73.1%，尚未在训练中完成一次原长路径回合；完整评估始终覆盖全任务。密集案例完成21个训练回合。价值网络从零初始化，两案例回报尺度和拟合程度差异明显；这是下一轮应检验的训练问题，尚不能单独视为因果证明。','',
        '**总体目标仍未达到。** 保留的14维模型在密集案例相对S2约快23.5%，相对现有强两动作参考约快3.84%–3.97%，仍低于5%。该两动作参考含历史控制骨干，不能冒充与14维策略严格配对的消融。原34角标准也尚未满足。','',
        '不建议直接给当前配置追加长时间训练。应先使价值估计尺度与完整最短时间目标匹配，并让新增长路径采样覆盖终段，再预登记有限验证。本轮不继续扩大计算预算。','',
        '工程处理遵循KISS/YAGNI：保持14维和既有执行链；DRY/SOLID：复用统一评估及PPO更新，将分段回报、运行编排和只读诊断分开。分段隔离、终止与bootstrap、采样概率、KL、来源散列均已核验。','',
        '论文正文、原前三图及最终PDF均未修改。方法未冻结，研究状态保持未完成。','',
        '数据绑定：registration.json、result.json、receipt.json、evidence_summary.json、straight_cruise_diagnostic.json、critic_diagnostic.json和verified_round_report.json。','']
    (STAGE/'结果说明.md').write_text('\n'.join(lines),encoding='utf-8')
    status_path=ROOT/'compact14_verified_status_20260922.json'
    backup=ROOT/'state_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(exist_ok=False,parents=True);saved=backup/status_path.name;shutil.copy2(status_path,saved)
    status=json.loads(status_path.read_text(encoding='utf-8'))
    status.update(updated_utc=report['updated_utc'],goal_complete=False,method_frozen=False,
        latest_ppo_round_report=str(STAGE/'verified_round_report.json'),latest_ppo_round_adopted=result['adopted'],
        next_step=report['next_research'])
    write(status_path,status)
    write(backup/'receipt.json',dict(original=str(status_path),backup=str(saved),sha256=sha(saved),
        reason='Append completed direct14 PPO evidence; keep original compact checkpoint unchanged.'))
    print(json.dumps(dict(report=str(STAGE/'verified_round_report.json'),wall_minutes=receipt['wall_s']/60,
        adopted=result['adopted'],checks=checks),ensure_ascii=False))


if __name__=='__main__':main()
