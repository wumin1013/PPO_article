"""Compile and render the bilingual editorial revision in the nonsynced work area."""
import sys,subprocess,json,re
from PIL import Image,ImageDraw
from pathlib import Path
from revise_original_vectors_academic import OUT,WORK,write,sha
COMPILE=Path('C:/Users/wumin/.codex/plugins/cache/openai-bundled/latex/0.2.7/scripts/compile_latex.py')
POP=Path('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
def main():
    inv={};suffix=sys.argv[1] if len(sys.argv)>1 else 'v1'
    for lang,name in [('en','manuscript'),('zh','manuscript_zh')]:
        if len(sys.argv)>2 and lang!=sys.argv[2]:continue
        src=OUT/f'latex_source_{lang}';bld=WORK/f'build_{lang}_{suffix}';bld.mkdir(exist_ok=True)
        cmd=[sys.executable,str(COMPILE),str(src/(name+'.tex')),'--compiler','texlive','--engine','xelatex','--output-directory',str(bld),'--json']
        r=subprocess.run(cmd,capture_output=True)
        (WORK/f'compile_{lang}_{suffix}.txt').write_bytes(r.stdout+r.stderr)
        print('COMPILE',lang,'exit',r.returncode,flush=True)
        if r.returncode:
            print((r.stdout+r.stderr).decode('utf-8',errors='replace')[-3000:]);raise RuntimeError('compile failed')
        pdf=bld/(name+'.pdf');vis=WORK/f'visual_{lang}_{suffix}';vis.mkdir(exist_ok=True)
        info=subprocess.check_output([str(POP/'pdfinfo.exe'),str(pdf)]).decode('utf-8',errors='replace');pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        r=subprocess.run([str(POP/'pdftoppm.exe'),'-r','105','-png',str(pdf),str(vis/'page')],capture_output=True);assert r.returncode==0
        subprocess.run(['D:/texlive/2025/bin/windows/pdftotext.exe','-layout',str(pdf),str(vis/'fulltext.txt')],capture_output=True,check=True)
        renders=sorted(vis.glob('page-*.png'));assert len(renders)==pages;contacts=[]
        for start in range(0,pages,4):
            canvas=Image.new('RGB',(1600,2320),'#eeeeee');draw=ImageDraw.Draw(canvas)
            for j,p in enumerate(renders[start:start+4]):
                im=Image.open(p).convert('RGB');im.thumbnail((780,1105));x=j%2*800+(800-im.width)//2;y=j//2*1160+35
                canvas.paste(im,(x,y));draw.text((j%2*800+20,j//2*1160+10),f'{lang} p{start+j+1}',fill='black')
            p=vis/f'contact-{start//4+1:02d}.png';canvas.save(p);contacts.append(str(p))
        log=pdf.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
        issues=re.findall(r'^.*(?:Overfull|undefined references|undefined citations|Missing character:).*$',log,re.M|re.I)
        inv[lang]=dict(pdf=str(pdf),sha256=sha(pdf),pages=pages,renders=[str(p) for p in renders],contacts=contacts,compile_issues=issues)
        print('RENDERED',lang,pages,'pages;',issues,flush=True)
    write(OUT/f'render_inventory_{suffix}.json',inv)
if __name__=='__main__':main()
