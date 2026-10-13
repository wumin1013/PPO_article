"""Preserve the checked front sections and add evidence-based design analysis."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='front_and_design_proof_v6')
    parser.add_argument('--with-curve',action='store_true')
    parser.add_argument('--with-minimal',action='store_true')
    args=parser.parse_args()
    if args.with_minimal and not args.with_curve:
        raise ValueError('Include the strong development reference with the simplification results')
    if not re.fullmatch(r'front_[a-z0-9_]+',args.stage):
        raise ValueError('Use a local front-section stage name')
    manuscript=BASE/'manuscript_joint_v1'
    names=('introduction.tex','problem_formulation.tex','methodology.tex','experimental_setup.tex',
        'design_analysis.tex','references.bib','design_analysis_references.bib')
    figures=(BASE/'method_figure1_v4/Figure_1.pdf',BASE/'method_figures_v2/Figure_2.pdf',BASE/'method_figures_v2/Figure_3.pdf')
    if args.with_curve:
        names+=('development_results.tex',)
        figures+=(BASE/'v7_complete_curve_figure_v2/v7_complete_curve.pdf',)
    if args.with_minimal:
        names+=('minimal_results.tex',)
        figures+=(BASE/'minimal_factor_corner_figures_v2/joint_corners.pdf',
                  BASE/'minimal_factor_corner_figures_v2/fixed_corners.pdf',
                  BASE/'preview_exploration_corners_v1/preview_corners.pdf')
    previous=BASE/'front_and_settings_proof_v4/front_sections.tex'
    sources=[Path(__file__),previous,*figures]+[manuscript/name for name in names]
    sources += [BASE/name/'receipt.json' for name in ('minimal_information_credit_audit_v1',
        'event_horizon_information_probe_v2','spatial_learning_contract_v1')]
    sources += [BASE/'minimal_state_sources_v1/curated_reading_manifest.json']
    if args.with_curve:
        sources += [BASE/'v7_complete_curve_audit_v1/results.json',
            BASE/'v7_complete_curve_figure_v2/visual_qa.json']
    if args.with_minimal:
        sources += [BASE/name/'results.json' for name in ('minimal_factorial_complete_audit_v1',
            'minimal_factors_60000_analysis_v1','minimal_first_batch_credit_v1','minimal_late_credit_audit_v1')]
        sources += [BASE/'minimal_factor_corner_figures_v2/visual_qa.json',
                    BASE/'minimal_factor_corner_figures_v2/source_data.csv']
        for name in ('credit_horizon_30000_analysis_v1','credit_horizon_60000_analysis_v1',
                     'credit_horizon_complete_audit_v1','saved_actor_gradient_decomposition_v1',
                     'v7_selected_lock3_v1','v7_long_time_regions_v1','v7_selected_exploration_coverage_v1',
                     'preview_exploration_60000_analysis_v1','v7_cruise_request_diagnosis_v1'):
            sources += [BASE/name/'receipt.json',BASE/name/'results.json']
        sources += [BASE/'preview_exploration_design_v1.md',
                    BASE/'preview_exploration_corners_v1/source_data.csv',
                    BASE/'preview_exploration_corners_v1/visual_qa.json']
        sources += [BASE/'credit_baseline_fulltext_v1/原文核查与项目含义.md',
            BASE.parent/'research_20260915_joint_policy_v1/sources/reward_shaping.txt']
    stage=register(args.stage,sources,
        dict(xelatex_runs=3,bibtex_runs=1,pdf_render_calls=1,model_calls=0,physics_ticks=0),
        'Add sourced observation/reward analysis to the current working front sections; no final performance or submission-completion claim.')
    work=Path('D:/PPO_article_joint_runs')/stage.name
    work.mkdir(exist_ok=False)
    for name in names:
        shutil.copy2(manuscript/name,work/name)
    for path in figures:
        shutil.copy2(path,work/path.name)
    main=previous.read_text(encoding='utf-8').replace('Introduction, Method and Experimental Settings',
        'Method, Experimental Settings and Learning Design')
    main=main.replace(r'\input{experimental_setup}',r'\input{experimental_setup}'+'\n'+r'\input{design_analysis}')
    if args.with_curve:
        main=main.replace(r'\begin{document}',r'\usepackage{placeins}'+'\n'+r'\begin{document}')
        main=main.replace(r'\bibliographystyle',r'\input{development_results}'+'\n'+r'\FloatBarrier'+'\n'+r'\bibliographystyle')
    if args.with_minimal:
        main=main.replace(r'\begin{document}',r'\usepackage{booktabs}'+'\n'+r'\begin{document}')
        main=main.replace(r'\bibliographystyle',r'\input{minimal_results}'+'\n'+r'\FloatBarrier'+'\n'+r'\bibliographystyle')
    main=main.replace(r'\bibliography{references}',r'\bibliography{references,design_analysis_references}')
    (work/'front_sections.tex').write_text(main,encoding='utf-8')
    (stage/'front_sections.tex').write_text(main,encoding='utf-8')
    texbin=Path('D:/texlive/2025/bin/windows')
    for i,(tool,args) in enumerate((('xelatex',['-interaction=nonstopmode','-halt-on-error','front_sections.tex']),
        ('bibtex',['front_sections']),('xelatex',['-interaction=nonstopmode','-halt-on-error','front_sections.tex']),
        ('xelatex',['-interaction=nonstopmode','-halt-on-error','front_sections.tex']))):
        out=subprocess.run([str(texbin/(tool+'.exe')),*args],cwd=work,capture_output=True,timeout=60)
        (stage/f'compile_{i}.txt').write_bytes(out.stdout+out.stderr)
        if out.returncode:
            raise RuntimeError('Compile failed; outputs retained')
    info=subprocess.run([str(texbin/'pdfinfo.exe'),'front_sections.pdf'],cwd=work,capture_output=True,check=True).stdout.decode('utf-8',errors='replace')
    count=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    subprocess.run([str(texbin/'pdftoppm.exe'),'-scale-to','1350','-png','front_sections.pdf','page'],
        cwd=work,capture_output=True,check=True,timeout=60)
    log=(work/'front_sections.log').read_text(encoding='utf-8',errors='replace')
    warnings=[line for line in log.splitlines() if any(token in line for token in
        ('Overfull','Underfull','LaTeX Warning','Package natbib Warning','undefined references','Missing character'))]
    for name in ('front_sections.pdf','front_sections.log','front_sections.bbl'):
        shutil.copy2(work/name,stage/name)
    write_json(stage/'build_results.json',dict(pages=count,pdf_sha256=sha(stage/'front_sections.pdf'),
        warnings=warnings,rendered_pages=[str(p) for p in sorted(work.glob('page-*.png'))],
        visual_qa_pending=True,whole_submission=False,goal_complete=False))
    receipt(stage,dict(status='compiled_rendered_pending_visual_review',pages=count,model_calls=0,physics_ticks=0))
    print(dict(pages=count,warnings=warnings))


if __name__=='__main__':
    main()
