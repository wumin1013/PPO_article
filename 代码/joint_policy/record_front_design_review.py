"""Record the already performed visual review without rebuilding the PDF."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from study_io import BASE, write_json, sha


def main():
    stage = BASE / 'front_and_design_proof_v6'
    target = stage / 'visual_qa.json'
    if target.exists():
        raise FileExistsError('This visual review is already recorded')
    build = json.loads((stage / 'build_results.json').read_text(encoding='utf-8'))
    pdf = stage / 'front_sections.pdf'
    assert build['pdf_sha256'] == sha(pdf)
    assert build['pages'] == 12 and build['warnings'] == []
    notes = [
        'Introduction: readable text, margins and page number; no clipping.',
        'Figure 1: both planning procedures, three action labels and caption readable.',
        'Figure 2: tolerance, ordered section and schematic inward transition readable.',
        'Problem constraints and original 72-component observation equation readable.',
        'Figure 3 and ten-range observation definition readable; actual 72D identity retained.',
        'Action, guidance and planar KCM equations fit within margins.',
        'Acceleration release, reward and full-time ordering equations readable.',
        'PPO, stopping criteria and corrected experimental-setting scope readable.',
        'Training/development geometry, strong fixed-preview comparison and success measure readable.',
        'Time/shape measures and compact candidate description readable.',
        'Value-shift equations and references readable; no unresolved citations.',
        'Remaining references readable; deliberate whitespace at end of partial manuscript.',
    ]
    write_json(target, dict(
        status='passed_full_page_visual_review',
        reviewed_utc=datetime.now(timezone.utc).isoformat(),
        pdf_sha256=sha(pdf),
        pages=[dict(page=i+1, path=p, sha256=sha(p), result=notes[i])
               for i, p in enumerate(build['rendered_pages'])],
        review_method='All 12 existing rendered pages inspected individually with view_image',
        clipping=False, overlap=False, missing_glyphs=False, undefined_references=False,
        original_figures_retained=True, final_manuscript=False, goal_complete=False))
    report = BASE / '逐图保留与修改报告_进行中.md'
    backup = BASE / 'before' / 'front_design_v6_visual_review'
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(report, backup / report.name)
    addition = '''
## 最小信息诊断与12页英文稿验收

英文前半稿与学习设计分析已接入`front_and_design_proof_v6/front_sections.pdf`，12页全部实际逐页目视通过；PDF哈希与各页PNG绑定见同目录`visual_qa.json`。正文、前三图、公式和引用无裁切、重叠、缺字或未解析引用。第8页明确十个固定采样范围属于原72维表示，第10页单独说明空间22维候选，未把候选当作已采用最终方法；Fig.3继续保留真实72维身份。

10页中文研究报告v2已全页验收，内容冻结至共同240k和事件16维负结果。之后的空间22维接口检查和保存回报残差诊断属于新增证据，尚未纳入该冻结PDF，不将旧PDF冒称最新全部结果。空间合同检查共9宏步、138tick，与前述诊断合计24宏步、438tick，均无训练更新。

v7共同240k四组仍12/12成功，联合20.184/20.467 s、固定19.132/18.932 s；这是保存开发轨迹分析，未新增模型评价或改画已完成四角图。较强固定、联合退化节点和全部旧图保留。最终Fig.4–9、独立完整种子、三类最终长路径、匹配消融及20–25页全文仍未完成。
'''
    report.write_text(report.read_text(encoding='utf-8') + addition, encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
