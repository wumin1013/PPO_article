"""Compile the current English front sections and checked method figures."""
import json
import re
import shutil
import subprocess
from pathlib import Path
from study_io import BASE,register,receipt,write_json,sha


if __name__=='__main__':
    manuscript=BASE/'manuscript_joint_v1'
    figures=[BASE/'method_figure1_v4/Figure_1.pdf',
        BASE/'method_figures_v2/Figure_2.pdf',BASE/'method_figures_v2/Figure_3.pdf']
    sources=[__file__,BASE/'method_figure_contract_v1.md']+figures
    sources += [manuscript/name for name in ['introduction.tex','problem_formulation.tex','methodology.tex','references.bib']]
    sources += [BASE/'method_figure1_v4/receipt.json',BASE/'method_figures_v2/receipt.json',
        BASE/'citation_claim_corrections_v1.md',BASE.parent/'research_20260915_joint_policy_v1/literature_motion.md',
        BASE.parent/'research_20260915_joint_policy_v1/motion_sources/toppra_crossref.json']
    stage=register('front_sections_proof_v3',sources,
        dict(xelatex_runs=3,bibtex_runs=1,max_rendered_pages=14,models=0,physics_ticks=0),
        'Build the current introduction, formulation, method and three visually checked method figures; this is not the full 20–25-page submission.')
    work=Path('D:/PPO_article_joint_runs/front_sections_proof_v3')
    work.mkdir(parents=True,exist_ok=False)
    for name in ['introduction.tex','problem_formulation.tex','methodology.tex','references.bib']:
        shutil.copy2(manuscript/name,work/name)
    for figure in figures:shutil.copy2(figure,work/figure.name)
    main=r'''% !TeX program = xelatex
\documentclass[preprint,12pt]{elsarticle}
\usepackage{amssymb,amsmath}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{fontspec,graphicx,caption}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdfauthor={},pdftitle={J-NNC: Introduction, Formulation and Method}}
\captionsetup[figure]{name=Fig.}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}
\begin{document}
\begin{center}
{\large J-NNC: Introduction, Formulation and Method}\\[4pt]
{\small Working manuscript sections}
\end{center}
\input{introduction}
\input{problem_formulation}
\input{methodology}
\bibliographystyle{elsarticle-num}
\bibliography{references}
\end{document}
'''
    (work/'front_sections.tex').write_text(main,encoding='utf-8')
    (stage/'front_sections.tex').write_text(main,encoding='utf-8')
    executables=Path('D:/texlive/2025/bin/windows')
    commands=[['xelatex','-interaction=nonstopmode','-halt-on-error','front_sections.tex'],
        ['bibtex','front_sections'],['xelatex','-interaction=nonstopmode','-halt-on-error','front_sections.tex'],
        ['xelatex','-interaction=nonstopmode','-halt-on-error','front_sections.tex']]
    for index,command in enumerate(commands):
        with (work/f'compile_{index}.stdout').open('wb') as out,(work/f'compile_{index}.stderr').open('wb') as err:
            result=subprocess.run([str(executables/(command[0]+'.exe')),*command[1:]],cwd=work,
                stdout=out,stderr=err,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:
            write_json(stage/'failure.json',dict(command=command,returncode=result.returncode,work=str(work)))
            raise RuntimeError('Compilation failed; binary console files retained')
    pdf=work/'front_sections.pdf'
    info=subprocess.run([str(executables/'pdfinfo.exe'),str(pdf)],capture_output=True,check=True).stdout.decode('utf-8',errors='replace')
    (work/'pdfinfo.txt').write_text(info,encoding='utf-8')
    count=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    assert count<=14
    log=(work/'front_sections.log').read_text(encoding='utf-8',errors='replace')
    warnings=[line for line in log.splitlines() if any(token in line for token in
        ['Overfull','Underfull','LaTeX Warning','Package natbib Warning','undefined references'])]
    subprocess.run([str(executables/'pdftoppm.exe'),'-scale-to','1350','-png',str(pdf),str(work/'page')],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    shutil.copy2(pdf,stage/'front_sections.pdf')
    shutil.copy2(work/'front_sections.log',stage/'front_sections.log')
    write_json(stage/'build_results.json',dict(pages=count,pdf_sha256=sha(pdf),warnings=warnings,
        rendered_directory=str(work),visual_QA_pending=True,whole_submission=False,
        included_figures=[dict(name=p.name,source=str(p),sha256=sha(p)) for p in figures]))
    receipt(stage,dict(status='front_sections_compiled_rendered_pending_full_visual_QA',pages=count,
        xelatex_runs=3,bibtex_runs=1,models=0,physics_ticks=0,goal_complete=False))
    print((stage/'build_results.json').read_text(encoding='utf-8'))
