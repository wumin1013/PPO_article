"""Present all enlarged-path comparisons as complete times and time reductions."""
import argparse
import hashlib
import json
from pathlib import Path

CASES=('piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10')
METHODS=('JNN','FIXED','S2','S2R','SCMO','COPP','NNC')
LABEL=dict(JNN='J-NNC',FIXED='Fixed preview',S2='Two-step (S2-Q)',S2R='Two-step (S2-R)',SCMO='SCMO',COPP='COPP',NNC='NNC')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def time_text(v,d):return '--' if v is None else f'{v:.{d}f}'
def generate(root,base,out):
    path=root/'canonical/final_comparison.json';data=json.loads(path.read_text('utf-8'))
    assert len(data['rows'])==56
    rows={(r['case'],r['method'],r['dt_s']):r for r in data['rows']}
    table=[r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Complete command times and J-NNC time reductions on the enlarged synthetic paths. Reductions are relative to each comparator at 1 ms. A dash indicates no complete execution or no applicable comparison.}',
        r'\label{tab:representative_comparison}',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Method & 1 ms (s) & 0.5 ms (s) & Reduction (\%) \\',r'\midrule']
    science=[]
    for index,(case,name) in enumerate(zip(CASES,('Curvature-variation path','Neck contour'))):
        if index:table.append(r'\midrule')
        table.extend([r'\multicolumn{4}{l}{\textit{'+name+r'}} \\',r'\addlinespace[2pt]'])
        j=rows[(case,'JNN',.001)]['metrics']['complete_time_s']
        for method in METHODS:
            a,b=[rows[(case,method,d)]['metrics']['complete_time_s'] for d in (.001,.0005)]
            saving=None if method=='JNN' or a is None else 100*(a-j)/a
            assert saving is None or saving>0
            cells=[LABEL[method],time_text(a,3),time_text(b,4),'--' if saving is None else f'{saving:.2f}']
            table.append(' & '.join(cells)+r' \\')
            science.append(dict(case=case,method=method,time_1ms_s=a,time_0p5ms_s=b,jnn_reduction_percent_1ms=saving))
    table.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (out/'comparison_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
    (out/'scaled_motion_results.tex').write_text(
        r'J-NNC shortens complete command execution on both the curvature-variation path and the neck contour. '
        r'Table~\ref{tab:representative_comparison} reports the complete times and the corresponding reductions relative to each feasible comparator. '
        r'Figure~\ref{fig:scaled_motion} presents the path geometry, separated speed profiles and completion-time differences.'+'\n',encoding='utf-8')
    if (out/'figure9_v9_caption.tex').is_file():
        caption=(out/'figure9_v9_caption.tex').read_text('utf-8').strip()
        (out/'scaled_figure.tex').write_text(r'\begin{figure}[htbp]'+'\n'+r'\centering'+'\n'+
            r'\includegraphics[width=\linewidth]{figure9_v9.pdf}'+'\n'+r'\caption{'+caption+'}\n'+
            r'\label{fig:scaled_motion}'+'\n'+r'\end{figure}'+'\n',encoding='utf-8')
    (out/'extension_efficiency_data.json').write_text(json.dumps(science,indent=2)+'\n',encoding='utf-8')
    scripts=['generate_paper_v9.py','plot_main_v9.py','plot_figure8_v9.py','plot_figure9_v9.py',
        'plot_extension_clarity_v9.py','format_extension_results_v9.py']
    receipt=dict(comparison_sha256=sha(path),formatter_sha256=sha(__file__),
        source_chain={n:sha(root/'src'/n) for n in scripts if (root/'src'/n).is_file()},
        unchanged_scientific_grid=True,retained_methods=list(METHODS),case_count=2,
        outputs={n:sha(out/n) for n in ('comparison_table.tex','scaled_motion_results.tex','scaled_figure.tex','extension_efficiency_data.json')})
    (out/'extension_presentation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(formatted=True,comparison_sha256=receipt['comparison_sha256'])))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.root,a.base,a.output)
