"""Bind inspected research documents without claiming completion of the goal."""
import json
from pathlib import Path
from study_io import BASE, register, receipt, write_json, record_progress, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    english = BASE/'front_long_results_proof_v13'
    chinese = BASE/'minimal_research_report_pdf_v10'
    en = read(english/'visual_qa.json')
    assert en['accepted'] and en['pages'] == 23 and en['all_pages_covered']
    cn = read(chinese/'build_manifest.json')
    assert len(cn['rendered_pages']) == 21
    assert not (chinese/'visual_qa.json').exists()
    log = (chinese/'report.log').read_text(encoding='utf-8')
    assert not any(t in log for t in ('Overfull', 'Underfull', 'Missing character', 'undefined references'))
    warnings = [line for line in log.splitlines() if ' Warning:' in line]
    checks = [dict(page=i, path=p, sha256=sha(p),
        inspection='Direct full-page visual inspection in this turn')
        for i, p in enumerate(cn['rendered_pages'], 1)]
    write_json(chinese/'visual_qa.json', dict(accepted=True, pages=21, all_pages_covered=True,
        pdf_sha256=sha(cn['pdf']), page_checks=checks, warnings=warnings,
        note='All 21 pages directly inspected; tables, equations, source figures and captions are readable; no clipping or overlap.',
        final_submission=False, goal_complete=False))
    stages = ('preview_exploration_60000_analysis_v1', 'preview_exploration_corners_v1',
        'v7_cruise_request_diagnosis_v1', 'v7_long_time_regions_v1')
    sources = [Path(__file__), english/'visual_qa.json', chinese/'visual_qa.json',
        BASE/'preview_exploration_corners_v1/visual_qa.json',
        BASE/'逐图保留与修改报告_进行中.md', BASE/'README.md',
        BASE/'manuscript_joint_v1/README.md', BASE/'manuscript_joint_v1/design_analysis_status.md']
    sources += [BASE/name/'receipt.json' for name in stages]
    stage = register('preview_and_long_report_delivery_v1', sources,
        dict(model_calls=0, physics_ticks=0, training_decisions=0),
        'Record the new completed exploration and cruise evidence with accepted research PDFs and the per-figure report. The final research objective is still unmet.')
    result = dict(chinese_pages=21, english_pages=23,
        english_pdf_sha256=en['pdf_sha256'], chinese_pdf_sha256=sha(cn['pdf']),
        chinese_package_warnings=warnings, all_pdf_pages_covered=True,
        selected_wide_preview_decisions=0, selected_wide_time_s=20.598,
        wide_endpoint_s=20.932, matched_narrow_endpoint_s=21.140, matched_fixed_endpoint_s=20.018,
        new_training_stage_complete=True, no_automatic_extension=True,
        next_research_question='Why successful training and rare failures lead to low feed requests; inspect saved batches before choosing a single new factor.',
        final_submission=False, sealed_test_opened=False, goal_complete=False)
    write_json(stage/'results.json', result)
    receipt(stage, dict(status='research_reports_and_new_figure_visually_accepted',
        chinese_pages=21, english_pages=23, model_calls=0, physics_ticks=0,
        training_decisions=0, final_submission=False, goal_complete=False))
    record_progress(stage.name, [stage/'receipt.json', stage/'results.json',
        english/'visual_qa.json', chinese/'visual_qa.json'],
        '已延续最小充分观测/简洁奖励目标，27维不是下限。v7四组、六组简化、λ1两组及D std .40单组均已完成，禁止重启。宽探索60k终点20.932 s，窄探索21.140 s，固定20.018 s；宽探索按原0/60k规则选回初值20.598 s。长mixed联合/固定/S2为376.113/347.717/323.367 s；新保存命令分析表明巡航缺口已出现在actor请求中。21页中文v10、23页英文v13及新D4均完成全页覆盖验收，逐图记录已更新。下一步只读保存训练批分析成功/失败对进给更新的影响及几何/任务长度差异，之后再选择一个有界单因素；无自动扩训练。最终方法与合理固定选距、独立完整种子、三类最终长路径/匹配消融、封存30条和最终Fig.4–9及投稿全文仍未完成，不以页数达标标记总目标完成。')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
