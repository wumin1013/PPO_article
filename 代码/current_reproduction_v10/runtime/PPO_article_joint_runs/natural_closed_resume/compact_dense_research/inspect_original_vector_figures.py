"""Inspect editable PDF objects before altering the approved original figures."""
import sys,json
from pathlib import Path
sys.path.append('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import pypdfium2 as pdfium
ROOT=Path(__file__).resolve().parent
WORK=ROOT/'lowdim_original_manuscript_work/academic_figure_revision'
WORK.mkdir(exist_ok=True)
SRC=Path('D:/PPO_article_work/论文项目/joint_policy_20260915_v1/paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision/latex_source_en/figures_lowdim_revision')
for i in [1,2,3]:
    doc=pdfium.PdfDocument(str(SRC/f'Figure_{i}.pdf')); page=doc[0];tp=page.get_textpage()
    rows=[]
    for j,o in enumerate(page.get_objects(textpage=tp)):
        rows.append(dict(index=j,level=o.level,type=o.type,bounds=o.get_bounds(),text=o.extract() if o.type==1 else None))
    (WORK/f'original_{i}_objects.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    page.render(scale=2).to_pil().save(WORK/f'original_{i}.png')
    print('FIG',i)
    for r in rows:
        if r['type']==1:print(r['index'],[round(x,1) for x in r['bounds']],repr(r['text']))
