"""Build the current evidence report; retain multipage tables and every version."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
from study_io import BASE, ROOT, register, receipt, sha, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',required=True)
    args=parser.parse_args()
    source=BASE/'最小充分状态与简洁奖励研究报告_进行中.md'
    header=ROOT/'论文项目/research_20260915_joint_policy_v1/report_header.tex'
    stages=('minimal_information_credit_audit_v1','compact_information_counterexamples_v1',
        'event_observation_contract_v1','event_horizon_information_probe_v2',
        'v7_checkpoint_120000_analysis_v1','v7_checkpoint_180000_analysis_v1','v7_checkpoint_240000_analysis_v1',
        'v7_checkpoint_300000_analysis_v1','spatial_learning_contract_v1','realized_value_credit_diagnostic_v3')
    sources=[Path(__file__),source,header]+[BASE/s/'receipt.json' for s in stages]
    sources += [BASE/s/'results.json' for s in stages if (BASE/s/'results.json').exists()]
    sources += [BASE/'front_and_design_proof_v6/visual_qa.json']
    for name in ('v7_checkpoint_359999_analysis_v1','v7_complete_curve_audit_v1'):
        if (BASE/name/'receipt.json').exists():
            sources += [BASE/name/'receipt.json',BASE/name/'results.json']
    sources += [BASE/'联合与只学前瞻的可比性补充.md']
    for name in ('minimal_factors_30000_analysis_v1','minimal_factors_60000_analysis_v1',
                 'minimal_factorial_complete_audit_v1','minimal_first_batch_credit_v1',
                 'minimal_late_credit_audit_v1'):
        sources += [BASE/name/'receipt.json',BASE/name/'results.json']
    corners=BASE/'minimal_factor_corner_figures_v2'
    sources += [corners/'visual_qa.json',corners/'source_data.csv',
                corners/'joint_corners.pdf',corners/'fixed_corners.pdf',
                BASE/'credit_horizon_design_v1.md']
    sources += [BASE/f'credit_horizon_{mode}_pilot_v1'/name
                for mode in ('joint','fixed') for name in ('registration.json','receipt.json')]
    for name in ('credit_horizon_30000_analysis_v1','credit_horizon_60000_analysis_v1',
                 'credit_horizon_complete_audit_v1','saved_actor_gradient_decomposition_v1',
                 'v7_selected_lock3_v1','v7_long_time_regions_v1','v7_selected_exploration_coverage_v1',
                 'preview_exploration_60000_analysis_v1','v7_cruise_request_diagnosis_v1'):
        sources += [BASE/name/'receipt.json',BASE/name/'results.json']
    sources += [BASE/'preview_exploration_design_v1.md',
        BASE/'preview_exploration_corners_v1/preview_corners.pdf',
        BASE/'preview_exploration_corners_v1/source_data.csv',
        BASE/'preview_exploration_corners_v1/visual_qa.json']
    stage=register('minimal_research_report_pdf_'+args.version,sources,
        dict(pandoc_calls=1,xelatex_passes=2,pdf_render_calls=1,model_calls=0,physics_ticks=0),
        'Compile the source-grounded state/reward report; final publication and full-page visual acceptance remain separate.')
    external=Path('D:/PPO_article_joint_runs')/stage.name
    external.mkdir(exist_ok=False)
    shutil.copy2(source,external/'report.md')
    for name in ('joint_corners.pdf','fixed_corners.pdf'):
        shutil.copy2(corners/name,external/name)
    shutil.copy2(BASE/'preview_exploration_corners_v1/preview_corners.pdf',external/'preview_corners.pdf')
    head=header.read_text(encoding='utf-8').replace('[BoldFont=SimHei]{SimSun}',
        '[BoldFont=SimHei,ItalicFont=KaiTi,SlantedFont=KaiTi]{SimSun}')
    (external/'header.tex').write_text(head+'\n'+r'\usepackage{needspace}\pagestyle{plain}'+'\n',encoding='utf-8')
    texbin=Path('D:/texlive/2025/bin/windows')
    env=os.environ.copy()
    env['PATH']=str(texbin)+os.pathsep+env['PATH']
    def run(arguments):
        out=subprocess.run([str(x) for x in arguments],cwd=external,env=env,capture_output=True)
        if out.returncode:
            raise RuntimeError((out.stdout+out.stderr).decode('utf-8',errors='replace')[-5000:])
        return out.stdout
    run(['D:/Anaconda/Scripts/pandoc.exe','report.md','--standalone',
         '--from=markdown+tex_math_dollars+raw_tex','--to=latex','--metadata=lang:en-US',
         '--variable=documentclass:ctexart','--variable=fontsize:11pt',
         '--variable=geometry:a4paper,margin=22mm','--variable=mainfont:Times New Roman',
         '--variable=sansfont:Arial','--variable=monofont:Consolas',
         '--include-in-header=header.tex','--output=report.tex'])
    texfile=external/'report.tex'
    tex=texfile.read_text(encoding='utf-8').replace(r'\usepackage{longtable,booktabs}',r'\usepackage{longtable,booktabs,array}')
    def table(match):
        n=len(match.group(1))
        widths={3:[.2,.34,.4],5:[.27,.1625,.1625,.1625,.1625]}.get(n,[.92/n]*n)
        return r'\begin{longtable}[]{@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\linewidth}' for w in widths)+r'@{}}'
    tex=re.sub(r'\\begin\{longtable\}\[\]\{@\{\}([lcr]+)@\{\}\}',table,tex)
    def keep_small_table(match):
        block=match.group(0)
        # Small comparison tables should not strand a single row across pages.
        # Large information inventories retain longtable's repeated headers.
        if block.count(r'\tabularnewline') <= 6:
            block=block.replace(r'\begin{longtable}[]',r'\begin{tabular}',1)
            block=block.replace(r'\endhead','').replace(r'\end{longtable}',r'\end{tabular}')
            return r'\par\medskip\noindent\begin{minipage}{\linewidth}\centering'+'\n'+block+'\n'+r'\end{minipage}\par\medskip'
        return block
    tex=re.sub(r'\\begin\{longtable\}.*?\\end\{longtable\}',keep_small_table,tex,flags=re.S)
    def breakable(match):
        value=match.group(1).replace(r'\_','_')
        return ''.join(r'\nolinkurl{'+part+'}' if part.isascii() else part
                       for part in re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+',value))
    tex=re.sub(r'\\texttt\{([^{}]+)\}',breakable,tex)
    # Keep a heading with the header and first row of its following long table.
    tex=re.sub(r'(?m)(\\(?:sub)*section\{[^\n]+\}\s*\n)(?=\s*\\begin\{longtable\})',
        lambda m:r'\Needspace{12\baselineskip}'+'\n'+m.group(1),tex)
    texfile.write_text(tex,encoding='utf-8')
    outputs=[run([texbin/'xelatex.exe','-interaction=nonstopmode','-halt-on-error','report.tex']) for _ in range(2)]
    (stage/'compiler_output.txt').write_bytes(b'\n'.join(outputs))
    run([texbin/'pdftotext.exe','-layout','report.pdf','extracted.txt'])
    run([texbin/'pdftoppm.exe','-r','100','-png','report.pdf','page'])
    info=run([texbin/'pdfinfo.exe','report.pdf']).decode('utf-8',errors='replace')
    info='\n'.join(line for line in info.splitlines() if '\ufffd' not in line)
    (stage/'pdfinfo.txt').write_text(info,encoding='utf-8')
    for suffix in ('.md','.tex','.log'):
        shutil.copy2(external/('report'+suffix),stage/('report'+suffix))
    output=stage/'最小充分状态与简洁奖励研究报告.pdf'
    shutil.copy2(external/'report.pdf',output)
    pages=sorted(external.glob('page-*.png'))
    write_json(stage/'build_manifest.json',dict(pdf_sha256=sha(output),pdf=str(output),
        rendered_pages=[str(p) for p in pages],build_directory=str(external),
        visual_acceptance_pending=True,final_manuscript=False,goal_complete=False))
    receipt(stage,dict(status='compiled_rendered_pending_visual_review',pages=len(pages),
        model_calls=0,physics_ticks=0,training_decisions=0))
    print(info)


if __name__=='__main__':
    main()
