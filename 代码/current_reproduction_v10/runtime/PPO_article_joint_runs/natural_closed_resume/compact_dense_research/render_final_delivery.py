"""Render final manuscript artifacts without changing their sources."""
from pathlib import Path
import json
import subprocess
import shutil
import sys
from PIL import Image, ImageOps, ImageDraw
import re

ROOT = Path(r'D:\PPO_article_joint_runs\natural_closed_resume\compact_dense_research')
WORK = ROOT/'lowdim_original_manuscript_work'
NEW = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1\paper_evidence_review\verified_results\lowdim_original_revision\selected923202_revision')
POP = Path(r'C:\Users\wumin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin')

def main():
    notes = WORK/'build_selected923202_final_notes_ascii_v6'
    notes.mkdir(exist_ok=True)
    shutil.copy2(NEW/'修改说明.tex', notes/'revision_notes.tex')
    cmd = [r'D:\texlive\2025\bin\windows\xelatex.exe', '-interaction=nonstopmode', '-halt-on-error', 'revision_notes.tex']
    for i in range(2):
        p = subprocess.run(cmd, cwd=notes, capture_output=True)
        (notes/f'compile_pass{i+1}.txt').write_bytes(p.stdout+p.stderr)
        if p.returncode:
            raise RuntimeError(f'Notes compile failed: {p.returncode}')
    documents = {
        'en': WORK/'build_selected923202_final_en_v4/manuscript.pdf',
        'zh': WORK/'build_selected923202_final_zh_v6/manuscript_zh.pdf',
        'notes': notes/'revision_notes.pdf',
    }
    inventory = json.loads((WORK/'final_render_inventory.json').read_text(encoding='utf-8')) if (WORK/'final_render_inventory.json').exists() else {}
    for key, pdf in documents.items():
        if len(sys.argv)>1 and key not in sys.argv[1:]:
            continue
        directory = WORK/f'final_visual_v6_{key}'
        directory.mkdir(exist_ok=True)
        p = subprocess.run([str(POP/'pdftoppm.exe'), '-r', '105', '-png', str(pdf), str(directory/'page')], capture_output=True)
        (directory/'render_output.txt').write_bytes(p.stdout+p.stderr)
        assert p.returncode == 0
        p = subprocess.run([shutil.which('pdftotext'), '-layout', str(pdf), str(directory/'fulltext.txt')], capture_output=True)
        (directory/'text_output.txt').write_bytes(p.stdout+p.stderr)
        assert p.returncode == 0
        pages = sorted(directory.glob('page-*.png'))
        info=subprocess.check_output([str(POP/'pdfinfo.exe'),str(pdf)]).decode('utf-8',errors='replace')
        assert len(pages) == int(re.search(r'Pages:\s+(\d+)',info).group(1))
        contacts=[]
        for first in range(0,len(pages),4):
            canvas=Image.new('RGB',(1600,2320),'#e8e8e8')
            d=ImageDraw.Draw(canvas)
            for j, page in enumerate(pages[first:first+4]):
                im=Image.open(page).convert('RGB')
                im.thumbnail((780,1105))
                x=(j%2)*800+(800-im.width)//2
                y=(j//2)*1160+35
                canvas.paste(im,(x,y))
                d.text(((j%2)*800+20,(j//2)*1160+10),f'{key} page {first+j+1}',fill='black')
            out=directory/f'contact-{first//4+1:02d}.png'
            canvas.save(out)
            contacts.append(str(out))
        inventory[key]=dict(pdf=str(pdf),pages=len(pages),renders=[str(p) for p in pages],contacts=contacts)
        print(key,len(pages),'pages',flush=True)
    (WORK/'final_render_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
