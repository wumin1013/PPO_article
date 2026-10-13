"""Bind the completed structural and dense-dynamics evidence, without freezing."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    stages=['historical_sparse_memory_full_closed_loop','historical_dense_predictive_action_diagnostic',
        'historical_dense_predictive_action_half_ms','dense_s2_half_ms']
    binding={}
    for name in stages:
        folder=ROOT/name
        receipt=read(folder/'receipt.json')
        assert receipt['exit_code']==0
        result_path=folder/('full_result.json' if name==stages[0] else 'result.json')
        binding[name]=dict(directory=str(folder),result=read(result_path),receipt=receipt,
            result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    nominal=read(ROOT/stages[1]/'result.json');refined=read(ROOT/stages[2]/'result.json')
    s2=read(ROOT/'dense_s2_initial/result.json');s2_refined=read(ROOT/stages[3]/'result.json')
    gains=dict(nominal_saved_s=s2['complete_time_s']-nominal['complete_time_s'],
        nominal_reduction_percent=100*(1-nominal['complete_time_s']/s2['complete_time_s']),
        refined_saved_s=s2_refined['complete_time_s']-refined['complete_time_s'],
        refined_reduction_percent=100*(1-refined['complete_time_s']/s2_refined['complete_time_s']),
        diagnostic_resolution_change_s=refined['complete_time_s']-nominal['complete_time_s'])
    now=datetime.now(timezone.utc)
    summary=dict(updated_utc=now.isoformat(),bindings=binding,gains=gains,
        goal_complete=False,method_frozen=False,
        interpretation='Historical learned controller plus an online finite-horizon action-search diagnostic; zero new PPO updates. Not final independent repeated two-action ablation evidence.',
        outstanding=['Fair independently trained two-output comparison with declared shared execution aids and total budgets.',
            'Learned three-action mechanism versus forecast-rule contribution.',
            'Independent repetitions,dimensional-reduction scope and allow/no-swing comparison.',
            'Unified adopted controller checks on both cases before freeze and final paper editing.'])
    target=ROOT/'historical_dynamics_breakthrough_20260922.json'
    assert not target.exists()
    target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=ROOT/'historical_dynamics_breakthrough_20260922.md'
    body=f'''# 原方法精简与密集案例动力学诊断记录

记录时间：{now.isoformat()}。沿用PPO_article与当前分支；未冻结，未完成，论文正文及前三图未改。

原长路径新闭环完整成功444.302s，完整NPZ轨迹哈希与历史相同；全部34角、全部22,210次动作、原始XY/V/A/J、≥99mm/s巡航均一致。因果记忆474→154，前驱特征缓存237→154；72维观测保留。这验证的是结构精简，不是观测维度最小性。

| 密集案例 | 1ms完整时间 | 0.5ms完整时间 |
|---|---:|---:|
| 历史策略＋有限预测修正诊断 | 15.285s | 15.3045s |
| 未改动S2规划 | 19.957s | 19.9565s |

两种分辨率均通过完整84门、解析欧氏允差、原始XY向量差分V/A/J和至少20ms物理停稳。时间分别减少{gains['nominal_reduction_percent']:.6f}%、{gains['refined_reduction_percent']:.6f}%；主方法诊断分辨率变化19.5ms，远小于约4.65s比较收益。这里没有据此冻结方法。

预测诊断保留历史模型、圆弧误触发修复及3mm前瞻下限，在名义动作的120ms恒动作预测接近不可行时，评价预先固定的11种局部动作变化，并从可行预测中选最小修正。真实动作仍保持20ms，执行与预测均使用共同动力学。它是在线模型预测参数选择，不是新增PPO训练，也不是递归几何安全证明。1ms运行759次决策中88次被修正：前瞻单独50次，进给单独36次，转向单独1次，进给/前瞻共同1次。预测共198,845个物理步，实轨迹15,285步；整轮墙钟113.486s，不据此宣称20ms实时求解。

1ms完整时间差4.672s，其中执行长度项为-0.071738s、速度亏损项为+4.743738s：轨迹略长，主要收益是减少低速损失。新84角中71角速度亏损不高于S2；≥99mm/s时间为1.014s，S2为3.422s。不能把完整时间优势扩写成每个新角或每个巡航指标均最优。新案例局部出弯外摆上限1.109897mm（已登记角窗内的指标），允差另经完整解析几何审核。

固定前瞻保持合理、有限的公平机会，不无限追求其最优。既有两动作25.334s结果保留，但其执行辅助、观测和历史训练预算与本次预测诊断不同，尚不足以作为最终受控消融。主线仍需核验学习策略与预测修正的贡献、真正两动作对照、重复性和降维范围。不会将条件搜索、离线拟合或诊断计成PPO训练。

所有数据路径、哈希、源码快照和预算收据见同名JSON。KISS/YAGNI体现为删除严格零输出系数对应的状态，未添加无关功能；DRY/SOLID体现为复用原执行与审核函数，通过独立包装实现诊断，原历史源码未改。

下一步：用成功轨迹检验能否学习这些局部修正，明确记录离线拟合和PPO各自作用；随后按有限预算完成真实两动作比较。达到整体验收后再写原18页稿的最终实验。
'''
    report.write_text(body,encoding='utf-8')
    states=[BASE/'当前执行状态.json',OUT/'live_research_summary.json',
        OUT/'all34_dominance_research/active_research_checkpoint.json',ROOT/'current_progress.json']
    backup=ROOT/'state_history'/now.strftime('%Y%m%dT%H%M%S%fZ');backup.mkdir(parents=True)
    receipts=[]
    for path in states:
        raw=path.read_bytes();(backup/path.name).write_bytes(raw)
        receipts.append(dict(original=str(path),sha256=hashlib.sha256(raw).hexdigest()))
        state=json.loads(raw)
        state.update(updated_utc=now.isoformat(),current='原长路径记忆精简完整复核完全一致；密集案例有限预测诊断在1ms/0.5ms均成功，相比S2减少23.41%/23.31%。尚非最终PPO/双动作消融与重复性结论，未冻结。',
            latest_dynamics_evidence=str(target),latest_dynamics_report=str(report),
            goal_complete=False,method_frozen=False,research_acceptance_passed=False,
            final_manuscript_accepted=False,final_pdf=None,latest_state_backup_receipt=str(backup/'receipt.json'),
            next_steps=summary['outstanding'])
        path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (backup/'receipt.json').write_text(json.dumps(dict(backups=receipts,evidence=str(target)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(report),gains=gains,goal_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
