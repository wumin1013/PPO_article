"""Capture the live handoff without restarting workers or changing goal status."""
import json
from datetime import datetime, timezone
from pathlib import Path
from study_io import BASE, ROOT, register, receipt, write_json, record_progress


def main():
    handoff = ROOT/'论文项目/新对话续接指令_20260915_联合学习目标与v7进行中.md'
    evidence = [Path(__file__), handoff, BASE/'handoff_goal_observation_20260915_v6.json',
        BASE/'ongoing_research_snapshot_v5/receipt.json', BASE/'v7_checkpoint_60000_analysis_v1/receipt.json',
        BASE/'v7_60000_corner_figure_v1/visual_qa.json', BASE/'学习信号与后续选择_20260915.md',
        BASE/'credit_baseline_fulltext_v1/receipt.json', BASE/'credit_baseline_fulltext_v1/sources.json',
        BASE/'credit_baseline_fulltext_v1/原文核查与项目含义.md',
        BASE/'paired_statistics_validation_v1/receipt.json', BASE/'paired_statistics_validation_v1/results.json']
    stage = register('ongoing_research_snapshot_v6', evidence,
        dict(models=0, physics_ticks=0, training_decisions=0),
        'Save user-requested new-conversation handoff, actual paused app-goal observation, live workers, completed evidence and added state/reward complexity research question.')
    names = ['joint_independent_refinement_v7', 'joint_gsde_refinement_v7',
             'fixed3_independent_refinement_v7', 'fixed3_gsde_refinement_v7']
    rows = []
    for name in names:
        path = Path('D:/PPO_article_joint_runs/rolling_states')/name/'resume_pointer.json'
        rows.append(dict(stage=name, observed_utc=datetime.now(timezone.utc).isoformat(),
            pointer=json.loads(path.read_text(encoding='utf-8')),
            failure_exists=(BASE/name/'failure.json').exists(),
            receipt_exists=(BASE/name/'receipt.json').exists(),
            next120k_summary_exists=(BASE/name/'validation_120000/summary.json').exists()))
    write_json(stage/'results.json', dict(v7_at_observation=rows, app_goal_last_observed='paused',
        worker_processes_observed_present_before_capture=[93076,74748,33836,71936],
        saved_goal_observation='handoff_goal_observation_20260915_v6.json',
        first60k_analysis_complete=True, first60k_figure_qa_pass=True,
        new_user_question='Investigate whether state-space and reward complexity obstruct useful learning; controlled simplification studies after existing v7, no premature causal claim.',
        tucker_main_text_reviewed_sections=['2','3','4','5','7'], tucker_supplement_read=False,
        paired_statistics_synthetic_checks_passed=8, real_final_statistics_available=False,
        sealed_test_opened=False, training_restarted=False, goal_complete=False))
    lines = ['# 目标续接：最新交接与复杂度研究', '',
        '完整新交接入口：论文项目/新对话续接指令_20260915_联合学习目标与v7进行中.md。以其研究目标、最新授权、源文件与读取顺序续接。', '',
        '本次应用目标实际观察为paused；四组后台训练worker在观察时仍存在，不能把两种状态混为一谈，也不能冒称目标已完成。本脚本没有控制进程或改变应用目标。', '',
        '| 阶段 | 当前保存决策 | phase | failure | final receipt |', '|---|---:|---|---|---|']
    for row in rows:
        p = row['pointer']
        lines.append(f"| {row['stage']} | {p['decision']} | {p['phase']} | {row['failure_exists']} | {row['receipt_exists']} |")
    lines += ['', '四组各360k/18000秒，下一共同120k评价；首60k分析与四角图QA均已完成，不重跑。新原文核查与8项合成统计检查也已完成，尚未运行真实最终统计。', '',
        '用户新增重点：状态空间、奖励是否过于复杂，导致学习无效。先区分冗余/尺度/可观测性与奖励分量/失败尺度/信用传播；作为待验证假设写入研究报告。后续有界消融分别只简化状态、只简化奖励塑形，保留必要动态信息与成功/失败语义，固定方法同等条件。不在当前v7中途改设置，不以增加部件或无边界训练代替判断。', '',
        '英文前半稿11页、本次未改。原稿用语风格、20–25页最终稿、逐图报告与全PDF逐页验收继续要求；独立完整种子、最终长路径/消融/封存测试、Fig.4–9仍缺。']
    (stage/'状态与下一步.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    receipt(stage, dict(status='completed_user_requested_handoff', models=0, physics_ticks=0,
        training_decisions=0, goal_complete=False, app_goal_observed='paused'))
    record_progress('handoff_snapshot_v6', [stage/'receipt.json', stage/'results.json',
        stage/'状态与下一步.md', *evidence],
        '新交接入口：新对话续接指令_20260915_联合学习目标与v7进行中.md；快照ongoing_research_snapshot_v6。应用目标实际paused但四组worker观察时仍运行，先核状态，不重复启动。v7共同60k/四角图QA及8项统计合成检查已完成，下一只分析新120k，原各360k/18000秒设置不变。用户新增状态/奖励复杂度假设，后续分别验证精简，不预断原因。最终稳定优势、独立完整种子、长路径/消融/封存测试与20–25页全PDF验收仍未完成。')
    print(handoff)


if __name__ == '__main__':
    main()
