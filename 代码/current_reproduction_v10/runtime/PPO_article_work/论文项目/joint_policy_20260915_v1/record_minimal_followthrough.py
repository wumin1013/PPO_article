"""Append completed diagnostics and 300k analysis through the existing writer."""
from datetime import datetime, timezone
import ast
import json
from pathlib import Path
from study_io import BASE, register, receipt, write_json, record_progress


def main():
    documents = [BASE/name for name in (
        'minimal_pilot_code_review_v1.md', 'minimal_factorial_design_v1.md',
        'train_minimal_factors.py', 'audit_v7_complete_curve.py',
        'front_and_design_proof_v6/visual_qa.json',
        'spatial_learning_contract_v1/receipt.json',
        'realized_value_credit_diagnostic_v1/failure.json',
        'realized_value_credit_diagnostic_v2/failure.json',
        'realized_value_credit_diagnostic_v3/receipt.json',
        'v7_checkpoint_300000_analysis_v1/receipt.json',
        'v7_checkpoint_300000_analysis_v1/results.json',
        '逐图保留与修改报告_进行中.md')]
    assert all(p.exists() for p in documents)
    for name in ('train_minimal_factors.py','audit_v7_complete_curve.py'):
        ast.parse((BASE/name).read_text(encoding='utf-8'))
    stage=register('minimal_followthrough_observation_v1',[Path(__file__),*documents],
        dict(model_calls=0,physics_ticks=0,training_updates=0),
        'Record already completed spatial contract, realized-return diagnostic, 300k saved analysis and 12-page visual review.')
    names=('joint_independent_refinement_v7','joint_gsde_refinement_v7',
           'fixed3_independent_refinement_v7','fixed3_gsde_refinement_v7')
    workers=[dict(stage=name,pointer=json.loads((Path('D:/PPO_article_joint_runs/rolling_states')/
        name/'resume_pointer.json').read_text(encoding='utf-8')),
        receipt_exists=(BASE/name/'receipt.json').exists()) for name in names]
    write_json(stage/'results.json',dict(observed_utc=datetime.now(timezone.utc).isoformat(),workers=workers,
        physical_diagnostics_cumulative=dict(macros=24,ticks=438,training_updates=0),
        realized_credit_deserializations_including_failed_attempts=9,
        front_manuscript_pages=12,all_pages_visually_reviewed=True,
        pilot_code_review_only=True,new_single_factor_training_started=False,
        sealed_test_opened=False,app_goal_current_task=None,goal_complete=False))
    receipt(stage,dict(status='completed_evidence_observation',model_calls=0,physics_ticks=0,training_updates=0))
    target=record_progress(stage.name,[stage/'receipt.json',stage/'results.json',*documents],
        'v7共同300k四组全部12/12：联合20.364/20.340s，固定18.976/18.832s，联合优势仍未建立；原设置与上限继续，最终保存终点359999。空间22维合同检查及保存实际回报残差诊断已完成，事件16维负结果保留；状态/奖励诊断累计24宏步438tick，0训练更新。英文前半稿v6共12页已全页目视通过，逐图记录已更新。六组各60k单因素先导入口已复核但尚未启动，等待v7最终分析及全曲线审核；共享base、只改状态/只去势差、固定D同等机会。30条封存测试未打开；最终方法、独立完整种子、长路径、匹配消融及20–25页论文全部验收仍未完成。')
    print(target)


if __name__=='__main__':
    main()
