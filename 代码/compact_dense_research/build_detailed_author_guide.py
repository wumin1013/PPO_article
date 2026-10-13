"""Typeset the author-only guide and verify manuscript bytes remain unchanged."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, re, subprocess
from PIL import Image, ImageDraw

ROOT=Path(r'D:\PPO_article_joint_runs\natural_closed_resume\compact_dense_research')
WORK=ROOT/'lowdim_original_manuscript_work/detailed_guide_work'
NEW=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1\paper_evidence_review\verified_results\lowdim_original_revision\selected923202_revision')
SRC=Path(r'D:\PPO_article_work\论文项目\submission_package_EAAI\06_latex_source_files\latex_source_anonymous')
POP=Path(r'C:\Users\wumin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin')
NAME='图件修改建议与实验外修改说明_详细版'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def tex_escape(s):
    math=[]
    def save(m):math.append(m.group(0));return f'MATHPLACEHOLDER{len(math)-1}END'
    s=re.sub(r'\$[^$]+\$',save,s)
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    s=s.replace('ε',r'$\varepsilon$').replace('≥',r'$\geq$').replace('→',r'$\rightarrow$').replace('²',r'$^2$')
    for i,m in enumerate(math):s=s.replace(f'MATHPLACEHOLDER{i}END',m)
    return s

def main():
    manifest=read(NEW/'delivery_manifest.json')
    for k in ['pdf','chinese_pdf']:assert sha(manifest[k])==manifest[k+'_sha256']
    en=(SRC/'manuscript.tex').read_text(encoding='utf-8')
    zh=(NEW/'latex_source_zh/manuscript_zh.tex').read_text(encoding='utf-8')
    patterns={'en':r'\bexploratory\s+(?:process|history)|\bfailure\s+repair|round[- ]by[- ]round|\bevidence\s+directory|\btraining\s+logs?\b', 'zh':r'探索过程|失败修复|逐轮日志|证据目录'}
    findings={}
    for lang,s in [('en',en),('zh',zh)]:
        mainpart,appendix=s.split(r'\appendix',1)
        findings[lang]={'main_body_matches':re.findall(patterns[lang],mainpart,re.I),'appendix_matches':re.findall(patterns[lang],appendix,re.I)}
        assert not any(findings[lang].values())
    build=WORK/'build'
    build.mkdir(parents=True,exist_ok=True)
    text=(NEW/(NAME+'.md')).read_text(encoding='utf-8')
    lines=text.splitlines();out=[];i=0;first=True
    while i<len(lines):
        line=lines[i]
        if line.startswith('# '):
            if not first:out.append(r'\clearpage')
            first=False
            out.append(r'\section*{'+tex_escape(line[2:])+'}')
        elif line.startswith('## '):out.append(r'\subsection*{'+tex_escape(line[3:])+'}')
        elif line.startswith('!['):
            key=re.search(r'\(([^)]+)\)',line).group(1)
            p=SRC/'figures_lowdim_revision'/('Figure_1.pdf' if key.endswith('1') else 'Figure_3.pdf')
            out.append(r'\begin{center}\includegraphics[width=.94\linewidth,height=75mm,keepaspectratio]{\detokenize{'+p.as_posix()+r'}}\end{center}')
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                if re.fullmatch(r'\|[-:|\s]+\|',lines[i].strip()):
                    i+=1
                    continue
                cells=[s.strip() for s in re.split(r'\s+\|\s+',lines[i].strip().strip('|').strip())]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells):rows.append(cells)
                i+=1
            i-=1
            cols=len(rows[0]);widths=[.31,.65] if cols==2 else [.19,.39,.36]
            spec='@{}'+' '.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\linewidth}' for w in widths)+'@{}'
            out.append(r'{\small\begin{longtable}{'+spec+'}')
            out.append(r'\toprule'+'\n'+' & '.join(tex_escape(c) for c in rows[0])+r'\\\midrule\endhead')
            for row in rows[1:]:out.append(' & '.join(tex_escape(c) for c in row)+r'\\[3pt]')
            out.append(r'\bottomrule\end{longtable}}')
        elif line.strip():out.append(tex_escape(line)+'\n')
        else:out.append('')
        i+=1
    preamble=r'''\documentclass[11pt,a4paper]{article}
\usepackage[top=18mm,bottom=18mm,left=20mm,right=20mm]{geometry}
\usepackage{fontspec,xeCJK,amsmath,amssymb,graphicx,booktabs,longtable,array,hyperref,fancyhdr}
\setmainfont{TeX Gyre Termes}
\setCJKmainfont[AutoFakeBold=2.0,ItalicFont=KaiTi]{SimSun}
\hypersetup{hidelinks,pdfauthor={},pdftitle={图件修改建议与实验外修改说明（详细版）}}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.09}
\setlength{\headheight}{14pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small PPO\_article：作者修订清单}
\fancyhead[R]{\small 非论文正文或附录}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{.3pt}
\begin{document}
'''
    tex=preamble+'\n'.join(out)+'\n'+r'\end{document}'+'\n'
    (NEW/(NAME+'.tex')).write_text(tex,encoding='utf-8')
    (build/'detailed_guide.tex').write_text(tex,encoding='utf-8')
    for passnum in range(2):
        p=subprocess.run([r'D:\texlive\2025\bin\windows\xelatex.exe','-interaction=nonstopmode','-halt-on-error','detailed_guide.tex'],cwd=build,capture_output=True)
        (build/f'compile_{passnum+1}.txt').write_bytes(p.stdout+p.stderr)
        assert p.returncode==0,p.returncode
    pdf=build/'detailed_guide.pdf'
    render=WORK/'render';render.mkdir(exist_ok=True)
    p=subprocess.run([str(POP/'pdftoppm.exe'),'-r','115','-png',str(pdf),str(render/'page')],capture_output=True)
    (render/'renderer.txt').write_bytes(p.stdout+p.stderr);assert p.returncode==0
    info=subprocess.check_output([str(POP/'pdfinfo.exe'),str(pdf)]).decode('utf-8',errors='replace')
    count=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    pages=[render/f'page-{i:02d}.png' for i in range(1,count+1)] if count>=10 else [render/f'page-{i}.png' for i in range(1,count+1)]
    for first in range(0,count,4):
        canvas=Image.new('RGB',(1600,2320),'#e8e8e8');d=ImageDraw.Draw(canvas)
        for j,page in enumerate(pages[first:first+4]):
            im=Image.open(page).convert('RGB');im.thumbnail((780,1105))
            canvas.paste(im,((j%2)*800+(800-im.width)//2,(j//2)*1160+35))
            d.text(((j%2)*800+20,(j//2)*1160+10),f'Guide page {first+j+1}',fill='black')
        canvas.save(render/f'contact-{first//4+1:02d}.png')
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),manuscripts_unchanged=True,english_pdf=manifest['pdf'],english_pdf_sha256=sha(manifest['pdf']),chinese_pdf=manifest['chinese_pdf'],chinese_pdf_sha256=sha(manifest['chinese_pdf']),english_source_sha256=sha(SRC/'manuscript.tex'),chinese_source_sha256=sha(NEW/'latex_source_zh/manuscript_zh.tex'),search_patterns=patterns,findings=findings,conclusion='The quoted storage sentence and exploratory logs are absent from both manuscript bodies and appendices. This turn changes only the author guide.',detailed_guide_build_pdf=str(pdf),detailed_guide_pages=count,renders=[str(p) for p in pages],protected_figures_unchanged={str(k):sha(SRC/f'figures_lowdim_revision/Figure_{k}.pdf') for k in [1,2,3]})
    write(NEW/'author_guide_scope_audit.json',report)
    print('AUTHOR_GUIDE',count,'pages; manuscript bodies and appendices contain no quoted storage/process text; PDFs unchanged.')

if __name__=='__main__':main()
