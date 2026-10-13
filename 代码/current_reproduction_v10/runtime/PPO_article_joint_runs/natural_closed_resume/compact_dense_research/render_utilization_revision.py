"""Render both manuscript PDFs for page-by-page layout inspection."""
import subprocess, json, re, sys, shutil
from pathlib import Path
from PIL import Image, ImageDraw
from revise_nnc_utilization_figures import ROOT, OUT, write, sha

WORK=ROOT/'lowdim_original_manuscript_work/figure_revision_20260924'
POP=Path('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')

def main():
    inv={}
    final='--final' in sys.argv
    for lang,build,name in [('en','build_en_v1','manuscript'),('zh','build_zh_v2','manuscript_zh')]:
        if final:build='build_'+lang+'_final'
        pdf=WORK/build/(name+'.pdf'); directory=WORK/('visual_'+lang+('_final' if final else ''))
        directory.mkdir(exist_ok=True)
        info=subprocess.check_output([str(POP/'pdfinfo.exe'),str(pdf)]).decode('utf-8',errors='replace')
        count=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        if len(list(directory.glob('page-*.png')))!=count:
            r=subprocess.run([str(POP/'pdftoppm.exe'),'-r','105','-png',str(pdf),str(directory/'page')],capture_output=True)
            assert r.returncode==0
            (directory/'render_output.txt').write_bytes(r.stdout+r.stderr)
        r=subprocess.run([shutil.which('pdftotext'),'-layout',str(pdf),str(directory/'fulltext.txt')],capture_output=True)
        assert r.returncode==0
        pages=sorted(directory.glob('page-*.png'));assert len(pages)==count
        contacts=[]
        for start in range(0,count,4):
            canvas=Image.new('RGB',(1600,2320),'#e8e8e8');draw=ImageDraw.Draw(canvas)
            for j,p in enumerate(pages[start:start+4]):
                im=Image.open(p).convert('RGB');im.thumbnail((780,1105))
                x=(j%2)*800+(800-im.width)//2;y=(j//2)*1160+35
                canvas.paste(im,(x,y));draw.text(((j%2)*800+20,(j//2)*1160+10),f'{lang} page {start+j+1}',fill='black')
            dest=directory/f'contact-{start//4+1:02d}.png';canvas.save(dest);contacts.append(str(dest))
        log=pdf.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        issues=re.findall(r'^.*(?:Overfull|undefined references|undefined citations|Missing character:).*$',log,re.M|re.I)
        inv[lang]=dict(pdf=str(pdf),sha256=sha(pdf),pages=count,renders=[str(p) for p in pages],contacts=contacts,compile_issues=issues)
        print(lang,count,'pages',issues,flush=True)
    write(OUT/('render_inventory_final.json' if final else 'render_inventory.json'),inv)

if __name__=='__main__':main()
