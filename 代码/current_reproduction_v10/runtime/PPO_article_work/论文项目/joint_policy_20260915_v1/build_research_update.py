"""Compile the current research update without touching any experiment."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from study_io import BASE, ROOT, register, receipt, write_json, sha


def main():
    source = BASE/'联合策略研究更新.md'
    header = ROOT/'论文项目/research_20260915_joint_policy_v1/report_header.tex'
    compiler = Path('C:/Users/wumin/.codex/plugins/cache/openai-bundled/latex/0.2.6/scripts/compile_latex.py')
    sources = [Path(__file__), source, header, BASE/'v6_selected_corner_comparison_figure_v1/visual_qa.json',
               BASE/'v6_selected_corner_comparison_figure_v1/v6_selected_corner_comparison.pdf',
               BASE/'selected_turning_dynamics_v6/receipt.json', BASE/'long_time_regions_v6/receipt.json',
               BASE/'exploration_learning_analysis_30000_v1/receipt.json']
    for name in ['exploration_learning_analysis_60000_v1', 'ppo_update_size_diagnosis_v2']:
        if (BASE/name/'receipt.json').exists():
            sources.append(BASE/name/'receipt.json')
    sources += [BASE/'research_update_pdf_v1/failure.json', BASE/'research_update_pdf_v2/failure.json',
                BASE/'fixed_distance_search_results_v1/receipt.json', BASE/'learning_design_v7.md']
    stage = register('research_update_pdf_v4', sources,
        dict(models=0, physics_ticks=0, pdf_builds=1),
        'Compile the evidence-based research update and render every page for visual acceptance.')
    external = Path('D:/PPO_article_joint_runs/research_update_pdf_v4')
    external.mkdir(exist_ok=False)
    name = '联合策略研究更新'
    content = source.read_text(encoding='utf-8').replace(
        '(v6_selected_corner_comparison_figure_v1/v6_selected_corner_comparison.pdf)', '(corners.pdf)')
    (external/(name+'.md')).write_text(content, encoding='utf-8')
    shutil.copy2(sources[4], external/'corners.pdf')
    shutil.copy2(header, external/'report_header.tex')
    header_path = external/'report_header.tex'
    header_path.write_text(header_path.read_text(encoding='utf-8').replace(
        '[BoldFont=SimHei]{SimSun}', '[BoldFont=SimHei,ItalicFont=KaiTi,SlantedFont=KaiTi]{SimSun}')+
        '\n'+r'\pagestyle{plain}'+'\n', encoding='utf-8')
    environment = os.environ.copy()
    texbin = Path('D:/texlive/2025/bin/windows')
    environment['PATH'] = str(texbin)+os.pathsep+environment['PATH']
    environment['PYTHONUTF8'] = '1'
    environment['PYTHONIOENCODING'] = 'utf-8'
    def run(args, cwd=external):
        result = subprocess.run([str(x) for x in args], cwd=cwd, env=environment, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stdout.decode('utf-8', errors='replace')[-5000:]+
                               result.stderr.decode('utf-8', errors='replace')[-3000:])
        return result
    run(['D:/Anaconda/Scripts/pandoc.exe', name+'.md', '--standalone',
         '--from=markdown+tex_math_dollars+raw_tex', '--to=latex', '--metadata=lang:en-US',
         '--variable=documentclass:ctexart', '--variable=fontsize:11pt',
         '--variable=geometry:a4paper,margin=22mm', '--variable=mainfont:Times New Roman',
         '--variable=sansfont:Arial', '--variable=monofont:Consolas',
         '--include-in-header=report_header.tex', '--output='+name+'.tex'])
    tex_path = external/(name+'.tex')
    tex = tex_path.read_text(encoding='utf-8')
    tex = tex.replace(r'\usepackage{longtable,booktabs}', r'\usepackage{longtable,booktabs,array}')
    def bounded_table(match):
        count = len(match.group(1))
        widths = {3:[.38,.29,.25], 4:[.23,.23,.23,.23], 5:[.20,.18,.18,.18,.18]}.get(count,[.92/count]*count)
        return r'\begin{longtable}[]{@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\linewidth}' for w in widths)+r'@{}}'
    tex = re.sub(r'\\begin\{longtable\}\[\]\{@\{\}([lcr]+)@\{\}\}', bounded_table, tex)
    def keep_small_table(match):
        block = match.group(0).replace(r'\begin{longtable}[]', r'\begin{tabular}')
        block = block.replace(r'\endhead', '').replace(r'\end{longtable}', r'\end{tabular}')
        return '\\par\\medskip\n\\noindent\\begin{minipage}{\\linewidth}\n'+block+'\n\\end{minipage}\\par\\medskip'
    tex = re.sub(r'\\begin\{longtable\}\[\].*?\\end\{longtable\}', keep_small_table, tex, flags=re.S)
    def breakable_path(match):
        value = match.group(1).replace(r'\_', '_')
        # URL verbatim fonts do not provide xeCJK fallback. Keep CJK as text.
        return ''.join(r'\nolinkurl{'+part+'}' if part.isascii() else part
                       for part in re.findall(r'[\x00-\x7f]+|[^\x00-\x7f]+', value))
    tex = re.sub(r'\\texttt\{([^{}]+)\}', breakable_path, tex)
    tex_path.write_text(tex, encoding='utf-8')
    # The wrapper's text-mode pipe fails on mixed Windows output encodings.
    # Reuse detected XeLaTeX and capture bytes end-to-end.
    compiler_output = []
    for _ in range(2):
        compiler_output.append(run([texbin/'xelatex.exe', '-interaction=nonstopmode',
                                   '-halt-on-error', name+'.tex']).stdout)
    (stage/'compiler_result.txt').write_bytes(b'\n'.join(compiler_output))
    run([texbin/'pdftotext.exe', '-layout', name+'.pdf', 'extracted.txt'])
    run([texbin/'pdftoppm.exe', '-r', '95', '-png', name+'.pdf', 'page'])
    info = run([texbin/'pdfinfo.exe', name+'.pdf']).stdout.decode('utf-8', errors='replace')
    info = '\n'.join(line for line in info.splitlines() if '\ufffd' not in line and not line.startswith('File size:'))
    info += '\nActual file size: '+str((external/(name+'.pdf')).stat().st_size)+' bytes\n'
    (stage/'pdfinfo.txt').write_text(info, encoding='utf-8')
    for suffix in ['.md','.tex','.pdf','.log']:
        shutil.copy2(external/(name+suffix), stage/(name+suffix))
    for name in ['selected_turning_dynamics_v6/turning_dynamics.csv', 'long_time_regions_v6/time_regions.csv']:
        shutil.copy2(BASE/name, stage/Path(name).name)
    write_json(stage/'build_manifest.json', dict(pdf_sha256=sha(stage/'联合策略研究更新.pdf'),
        build_directory=str(external), rendered_pages=[str(p) for p in sorted(external.glob('page-*.png'))],
        visual_acceptance_pending=True, final_manuscript=False, goal_complete=False))
    receipt(stage, dict(status='research_update_compiled_rendered_pending_visual_review', models=0, physics_ticks=0))
    print(info)


if __name__ == '__main__':
    main()
