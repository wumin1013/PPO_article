"""Record inspected first-checkpoint figures and new saved-data evidence."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from study_io import BASE, register, receipt, write_json, record_progress, sha


def main():
    figure = BASE/'v7_60000_corner_figure_v1'
    qa = figure/'visual_qa.json'
    assert not qa.exists()
    preview = Path(json.loads((figure/'caption.json').read_text(encoding='utf-8'))['preview'])
    write_json(qa, dict(status='single_pdf_page_visually_inspected_pass', pages=1,
        pdf_sha256=sha(figure/'v7_60000_corners.pdf'), svg_sha256=sha(figure/'v7_60000_corners.svg'),
        rendered_preview_path=str(preview), rendered_preview_sha256=sha(preview),
        actual_xy_equal_aspect=True, all_four_corners_shown=True,
        tolerance_geometry='Euclidean buffer of the same finite reference, 1.25 mm',
        labels_clipped=False, overlapping_annotations=False,
        intermediate_development_figure=True, final_manuscript_figure=False))
    names = ['joint_independent_refinement_v7', 'joint_gsde_refinement_v7',
             'fixed3_independent_refinement_v7', 'fixed3_gsde_refinement_v7']
    source_files = [Path(__file__), qa, BASE/'v7_checkpoint_60000_analysis_v1/results.json',
        BASE/'v7_checkpoint_60000_analysis_v1/receipt.json', BASE/'saved_preview_tail_mass_v1/receipt.json',
        BASE/'saved_action_credit_v1/receipt.json', BASE/'saved_action_credit_v1/results.json',
        BASE/'credit_outcome_decomposition_v1/receipt.json', BASE/'credit_outcome_decomposition_v1/results.json',
        BASE/'学习信号与后续选择_20260915.md', BASE/'learning_design_v7.md']
    source_files += [BASE/name/'registration.json' for name in names]
    stage = register('ongoing_research_snapshot_v5', source_files,
        dict(model_calls=0, physics_ticks=0, training_decisions=0),
        'Record the complete v7 first paired checkpoint, inspected corner figure, and saved exploration/credit diagnostics.')
    observed = []
    for name in names:
        pointer_path = Path('D:/PPO_article_joint_runs/rolling_states')/name/'resume_pointer.json'
        observed.append(dict(stage=name, observation_utc=datetime.now(timezone.utc).isoformat(),
            pointer=json.loads(pointer_path.read_text(encoding='utf-8')),
            failure_exists=(BASE/name/'failure.json').exists(), receipt_exists=(BASE/name/'receipt.json').exists()))
    write_json(stage/'results.json', dict(v7_pointers=observed,
        first_checkpoint_complete=True, first_checkpoint_times_s=[21.140, 20.306, 20.018, 19.908],
        first_checkpoint_success_counts=[12, 12, 12, 12],
        saved_failure_credit=dict(failure_rows=50, all_rows=1327, centered_square_share=.9486),
        new_training_started_by_this_snapshot=False, sealed_test_opened=False,
        goal_complete=False))
    content = '''# 目标续接：v7首个检查点与学习信号

v7四组共同60000决策评价全部完成且均12/12成功。联合独立21.140 s、联合gSDE20.306 s、固定独立20.018 s、固定gSDE19.908 s；同八条分别10.680/10.316/10.205/10.232 s，常规S2复用9.493 s。尚未建立联合优势。150°最近尖顶分别0.947/0.992/0.991/1.087 mm，明显深切目标仍未达到。

新四角图v7_60000_corner_figure_v1已实际SVG转PDF，整页预览目视通过，QA绑定原PDF、SVG与预览哈希。四角50/95/125/150度与四组全部显示，真实XY等比例及同一1.25 mm欧氏允差域；这是中间开发图，不冻结为最终Fig.9。

已完成三项保存数据诊断，不重跑：saved_preview_tail_mass_v1（两次模型文件反序列化、零前向）；saved_action_credit_v1（四份59999步原批、四次整批actor前向，零critic/优化/物理）；credit_outcome_decomposition_v1（两份原批、零模型/优化/物理）。联合独立批中50行失败占全批优势中心平方和94.86%，不是梯度损失比例。详见学习信号与后续选择_20260915.md。没有据此中途修改v7。

四组仍按learning_design_v7.md的各360000新决策/18000秒运行，下一共同评价120000。先核当前pointer、failure/receipt和实际进程，不按历史PID表重启。D盘保存完整训练状态；用joint_compact_v5读取。首60000分析入口analyze_v7_checkpoint.py --decisions 60000已经执行，禁止重复；后续只对新完成的共同120000等节点调用相应参数。

七值固定搜索、四组原60k探索、v6长mixed、动作响应、转向动态、时间分区和学习率诊断全部复用。当前较强固定已更新到v7中间19.908 s；较强联合中间20.306 s。完整最终选择仍需保留全曲线和旧结果。

7页研究更新PDF、15页原研究报告、11页英文前半稿均保留，本次未更改英文主稿。独立完整种子、最终三类长路径、匹配无KCM、封存30条测试、Fig.4–9与20–25页全文逐页验收仍未完成。封存路径未打开，目标active。

工程原则：只读诊断与训练分开，复用既有评价/画图/统一writer；未添加逐角执行规则；原文件修改前备份，所有新增文本UTF-8无BOM。
'''
    (stage/'状态与下一步.md').write_text(content, encoding='utf-8')
    report = BASE/'逐图保留与修改报告_进行中.md'
    backup = BASE/'before'/'v7_first_checkpoint_record'
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(report, backup/report.name)
    with report.open('a', encoding='utf-8') as stream:
        stream.write('\n## v7共同60000决策四角图\n\n四组共同60000决策评价全部12/12成功；四角图v7_60000_corner_figure_v1按已确认版式绘制并完成SVG转PDF与整页目视检查，源CSV保留。150°最近尖顶0.947/0.992/0.991/1.087 mm，仍未达到明显深切目标。全部角与方法均展示；中间开发图不冒充最终Fig.9。最新学习信号与原批失败尺度分解见ongoing_research_snapshot_v5及学习信号与后续选择_20260915.md。原图和英文主稿未改，最终20–25页与全PDF验收仍待完成。\n')
    receipt(stage, dict(status='completed_first_v7_checkpoint_record', model_calls=0,
        physics_ticks=0, training_decisions=0, goal_complete=False))
    record_progress('v7_first_checkpoint_snapshot_v5', [stage/'receipt.json', stage/'results.json',
        stage/'状态与下一步.md', qa, report, *source_files],
        'v7四组共同60k评价全部12/12：联合21.140/20.306 s，固定20.018/19.908 s，仍未建立优势；四角新图全页QA通过。保存前瞻尾概率、动作信用与失败尺度分解已完成，不重跑。四组按各360k/18000秒继续，下一只分析共同120k新评价，不中途改参。最新续接ongoing_research_snapshot_v5；独立完整种子、最终长路径/消融/封存测试、Fig.4–9及20–25页英文全文验收仍缺。')
    print(stage/'状态与下一步.md')


if __name__ == '__main__':
    main()
