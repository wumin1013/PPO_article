"""Bind actually inspected PDF pages and preserve rejected layout versions."""
import json
from pathlib import Path
from study_io import BASE, register, receipt, write_json, record_progress, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    english = BASE / 'front_credit_results_proof_v12'
    chinese = BASE / 'minimal_research_report_pdf_v9'
    old_english = read(BASE / 'front_minimal_results_proof_v11/visual_qa.json')
    assert old_english['accepted']
    en = read(english / 'build_results.json')
    cn = read(chinese / 'build_manifest.json')
    assert en['pages'] == 20 and en['warnings'] == [] and len(cn['rendered_pages']) == 17
    outputs = []
    for directory, paths, pdf, prefix in (
            (english, en['rendered_pages'], english / 'front_sections.pdf', 16),
            (chinese, cn['rendered_pages'], Path(cn['pdf']), 5)):
        assert not (directory / 'visual_qa.json').exists()
        pages = []
        for i, path in enumerate(paths, 1):
            digest = sha(path)
            if i <= prefix:
                previous = (old_english['page_checks'][i-1]['sha256'] if directory == english else
                    sha(Path('D:/PPO_article_joint_runs/minimal_research_report_pdf_v8') / f'page-{i:02d}.png'))
                assert digest == previous, (directory.name, i)
                inspected = ('Identical to accepted v11 page' if directory == english else
                    'Identical to v8 page directly inspected in this turn')
            else:
                inspected = 'Direct full-page visual inspection in this turn'
            pages.append(dict(page=i, path=str(path), sha256=digest, inspection=inspected))
        warnings = []
        if directory == chinese:
            log = (directory / 'report.log').read_text(encoding='utf-8')
            assert not any(token in log for token in ('Overfull', 'Underfull', 'Missing character', 'undefined references'))
            warnings = [line for line in log.splitlines() if ' Warning:' in line]
            assert len(warnings) == 3
        write_json(directory / 'visual_qa.json', dict(accepted=True, pages=len(paths), all_pages_covered=True,
            pdf_sha256=sha(pdf), page_checks=pages, warnings=warnings,
            note='Readable tables, equations, unchanged source figures and labels; no clipping, overlap or unresolved references.',
            final_submission=False, goal_complete=False))
        outputs.append(directory / 'visual_qa.json')
    for version, pages, checked, reason in (
            ('v7', 18, list(range(1,19)), 'Small 180k and event tables split across pages 2/3 and 5/6'),
            ('v8', 17, list(range(1,9)), 'Reward-component table split after its first data row across pages 6/7')):
        folder = BASE / ('minimal_research_report_pdf_' + version)
        manifest = read(folder / 'build_manifest.json')
        assert not (folder / 'visual_qa.json').exists()
        write_json(folder / 'visual_qa.json', dict(accepted=False, pages=pages, directly_inspected_pages=checked,
            reason=reason, pdf_sha256=manifest['pdf_sha256'], original_retained=True,
            correction='Short tables retained as unbroken tabular blocks in v9; large inventories remain multipage.'))
        outputs.append(folder / 'visual_qa.json')
    stage = register('credit_report_delivery_v1', [Path(__file__), *outputs,
        BASE / 'credit_horizon_complete_audit_v1/receipt.json',
        BASE / 'saved_actor_gradient_decomposition_v1/receipt.json',
        BASE / 'v7_selected_lock3_v1/receipt.json'],
        dict(model_calls=0, physics_ticks=0, training_decisions=0),
        'Record full-page review of current Chinese and English research documents, retaining previous layout rejections.')
    receipt(stage, dict(status='current_reports_visually_reviewed', chinese_pages=17, english_pages=20,
        final_submission=False, goal_complete=False))
    record_progress(stage.name, [stage / 'receipt.json', *outputs],
        'λ1两组60k及成本、真实网络梯度和所选v7锁3的24条新评价均完成；17页中文报告v9、20页英文v12完成全页覆盖验收。锁D仅差22/40ms，不能代表强固定对照优势。v7_joint_long_mixed_v1与v7_fixed_long_mixed_v1按已选最强模型各补一次公开长mixed评价；查真实receipt和pointer，不重启。最终Fig.4–9、独立种子、合理选距、三类最终长路径/消融/封存测试仍缺，总目标未完成。')
    print('Chinese 17 pages and English 20 pages accepted; overall goal remains incomplete.')


if __name__ == '__main__':
    main()
