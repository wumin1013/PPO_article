"""Resolve the committed manuscript dependencies and render every baseline page."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from PIL import Image, ImageDraw

ROOT=Path(r'D:\PPO_article_work')
AUDIT=Path(r'D:\PPO_article_recovery\main_restore_20260921_101841')
SOURCE=AUDIT/'pdf_source_commit/论文项目'
QA=AUDIT/'main_pdf_pages'
QA.mkdir(exist_ok=True)
receipt=json.loads((AUDIT/'restore_receipt.json').read_text(encoding='utf-8'))
tree=json.loads((AUDIT/'pdf_commit_tree.json').read_text(encoding='utf-8'))
lookup={r['path']:r for r in tree}
tex=(SOURCE/'main.tex').read_text(encoding='utf-8-sig')
dependencies=[]
for kind, name in re.findall(r'\\(includegraphics|input|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}',tex):
    names=[name]
    if kind=='includegraphics': names+=[f'figures/{name}']
    if kind=='bibliography': names=[name+'.bib']
    candidates=['论文项目/'+n for n in names]
    matched=next((p for p in candidates if p in lookup),None)
    if matched:
        row=lookup[matched]
        data=subprocess.check_output(['git','cat-file','blob',row['blob']],cwd=ROOT)
        dest=SOURCE.parent/matched
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(data)
        dependencies.append(dict(kind=kind,request=name,path=matched,blob=row['blob'],sha256=hashlib.sha256(data).hexdigest(),lfs_pointer=data.startswith(b'version https://git-lfs.github.com/spec/v1')))
    else:
        dependencies.append(dict(kind=kind,request=name,missing=True,candidates=candidates))
data_sources=[r for r in tree if r['path'].startswith(('论文项目/generated/','PPO_project/results/','论文项目/scripts/'))]
(AUDIT/'main_dependency_audit.json').write_text(json.dumps(dict(dependencies=dependencies,tracked_data_source_candidates=data_sources),ensure_ascii=False,indent=2),encoding='utf-8')
subprocess.run(['pdftotext','-layout',str(SOURCE/'main.pdf'),str(QA/'full_text.txt')],check=True)
pages=(QA/'full_text.txt').read_text(encoding='utf-8').split('\f')
if not pages[-1].strip():pages.pop()
for i,t in enumerate(pages,1):(QA/f'page_{i:02d}.txt').write_text(t,encoding='utf-8')
subprocess.run(['pdftoppm','-r','100','-png',str(SOURCE/'main.pdf'),str(QA/'page')],check=True)
images=sorted(QA.glob('page-*.png'))
for i in range(0,len(images),2):
    ims=[]
    for p in images[i:i+2]:
        im=Image.open(p).convert('RGB'); im.thumbnail((830,1175));ims.append(im)
    out=Image.new('RGB',(sum(im.width for im in ims),max(im.height for im in ims)+30),'#ddd')
    x=0
    for j,im in enumerate(ims):
        out.paste(im,(x,30));ImageDraw.Draw(out).text((x+10,8),f'Main baseline page {i+j+1}',fill='black');x+=im.width
    out.save(QA/f'contact_{i+1:02d}_{i+len(ims):02d}.png')
print(json.dumps(dict(page_count=len(pages),dependencies=dependencies,tracked_source_candidates=len(data_sources),qa=str(QA)),ensure_ascii=False,indent=2))

