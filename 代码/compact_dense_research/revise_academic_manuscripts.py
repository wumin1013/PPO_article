"""Keep manuscript logic; align bilingual text with the revised vector figures."""
import re,shutil,difflib
from revise_original_vectors_academic import ROOT,OLD,PREV,OUT,FIG,WORK,sha,write

def paragraph(text,start,new):
    a=text.index(start);b=text.index('\n\n',a);return text[:a]+new+text[b:]
def caption(text,label,new):
    end=text.index('\\label{'+label+'}');a=text.rindex('\\caption{',0,end);pos=a+len('\\caption{');depth=1;b=pos
    while depth:
        if text[b]=='{':depth+=1
        elif text[b]=='}':depth-=1
        b+=1
    return text[:pos]+new+text[b-1:]
def reorder_tables(text,lang):
    # Original values are carried through verbatim; only row order and emphasis change.
    for case in (['Long route','Dense case'] if lang=='en' else ['长路径','密集案例']):
        for first in ['NNC','S2']:
            marker=case+' & '+first+' & '
            if marker not in text:continue
            a=text.index(marker);b=text.index('\\midrule' if case in ['Long route','长路径'] else '\\bottomrule',a)
            lines=text[a:b].strip().splitlines();rows={line.split('&')[1].strip():line.split('&') for line in lines}
            ordered=[]
            for i,m in enumerate(['J-NNC','Two-action' if lang=='en' else '两动作','S2','NNC']):
                if m not in rows:continue
                fields=rows[m];fields[0]=case+' ' if i==0 else ' '
                if m=='J-NNC':
                    fields[1]=' \\textbf{J-NNC} '
                    value_index=3 if first=='NNC' else 2
                    value=fields[value_index].strip();fields[value_index]=' \\textbf{'+value+'} '
                elif m=='S2':fields[1]=' Two-step planner ' if lang=='en' else ' 两步规划法 '
                ordered.append('&'.join(fields))
            text=text[:a]+'\n'.join(ordered)+'\n'+text[b:]
    # Resolution table: place the proposed method first, then the same comparator order.
    for old,new in ([
        ('Case & S2 (s) & Two-action (s) & J-NNC (s)',r'Case & J-NNC (s) & Two-action (s) & Two-step planner (s)'),
        ('Long route & 445.1850 & 444.7310 & 444.1950','Long route & 444.1950 & 444.7310 & 445.1850'),
        ('Dense case & 19.9565 & 15.8955 & 14.8785','Dense case & 14.8785 & 15.8955 & 19.9565')]
        if lang=='en' else [
        ('案例 & S2（s） & 两动作（s） & J-NNC（s）','案例 & J-NNC（s） & 两动作（s） & 两步规划法（s）'),
        ('长路径 & 445.1850 & 444.7310 & 444.1950','长路径 & 444.1950 & 444.7310 & 445.1850'),
        ('密集案例 & 19.9565 & 15.8955 & 14.8785','密集案例 & 14.8785 & 15.8955 & 19.9565')]):
        assert old in text;text=text.replace(old,new)
    return text.replace(r'\setlength{\tabcolsep}{5pt}',r'\setlength{\tabcolsep}{3.5pt}')
def edit_en(t):
    t=reorder_tables(t,'en')
    t=t.replace('the traditional two-step baseline S2 first smooths the geometry and then schedules jerk-limited motion.',
        'the two-step planner first smooths the geometry and then schedules jerk-limited motion.')
    t=t.replace('the worst S2-normalized complete time','the worst complete time normalized by the two-step planner')
    t=re.sub(r'\bS2\b','the two-step planner',t)
    t=t.replace('; the two-step planner travels','; the two-step planner travels')
    t=caption(t,'fig:comparison_arch','Comparison of conventional two-step planning and J-NNC joint motion planning. The policy schematic shows 14 current inputs and three coordinated actions. The KCM maps the combined request to feasible planar motion.')
    t=caption(t,'fig:framework','Closed-loop J-NNC framework. Fixed forward geometry and current motion form the 14-dimensional state. The learned policy supplies steering, feedrate and Adaptive Preview requests, while the KCM determines feasible planar commands. The dashed outer connection updates policy parameters during learning and refinement.')
    t=paragraph(t,'Figure~\\ref{fig:full_utilization} extends',r'Figure~\ref{fig:full_utilization} compares the methods on common axes at the prescribed $90^\circ$ corner C02. J-NNC maintains higher speed through the transition while respecting the same acceleration and jerk limits. The integrated loss panel accounts for both the depth and duration of deceleration: loss is 86.1\% lower than for the two-step planner and 36.7\% lower than for the two-action controller. Complete V/A/J records and the final stop are retained in \figref{fig:complete_support}.')
    t=t.replace('local utilization is resolved in \\figref{fig:local_shapes}.','local utilization is resolved in \\figref{fig:full_utilization}.')
    t=caption(t,'fig:full_utilization',r'Direct comparison at the prescribed $90^\circ$/C02 corner, using the same 1 ms records. Panels (a)--(c) align speed, normalized vector acceleration and normalized vector jerk by reference progress; dotted horizontal lines denote the common limits. Panel (d) reports integrated speed deficit and minimum speed over the common $\pm10$ mm window. The curves span $\pm20$ mm to include approach and recovery.')
    t=paragraph(t,'Figure~\\ref{fig:local_shapes} relates',r'Figure~\ref{fig:local_shapes} relates the motion to the executed corner shape. The upper row shows the prescribed $135^\circ$, $90^\circ$ and $45^\circ$ corners in the incoming-tangent frame, with the inward normal as the vertical axis. The lower row enlarges the marked transition regions. Equal aspect and common scales within each row preserve the geometric comparison. J-NNC turns inward before the vertex and distributes the direction change through the tolerance band, supporting the higher cornering feedrate in \figref{fig:full_utilization}.')
    t=caption(t,'fig:local_shapes',r'Local executed trajectories at the prescribed $135^\circ$/C01, $90^\circ$/C02 and $45^\circ$/C06 corners. Panels (a)--(c) show approach and recovery; dotted rectangles identify the enlarged regions in (d)--(f). Shading denotes the 1.25 mm Euclidean corridor. Each row uses equal aspect and common scales. The same frozen model and 1 ms execution records are used throughout.')
    t=t.replace('The local utilization curves explain why trajectory shape matters.','The trajectory details and local utilization curves explain why trajectory shape matters.')
    support=r'''
\begin{figure}[p]
\centering
\renewcommand{\thefigure}{C.\arabic{figure}}
\setcounter{figure}{0}
\includegraphics[width=0.98\linewidth,height=0.75\textheight,keepaspectratio]{Figure_C1.pdf}
\caption{Complete long-route records at 1 ms. Columns show J-NNC, the two-action controller and the two-step planner; rows show normalized vector speed, acceleration and jerk. Display reduction preserves extrema. The lower panel shows complete braking and at least 20 ms of physical rest. These full records support the local comparison and the complete-execution statistics in the main text.}
\label{fig:complete_support}
\end{figure}
\clearpage

'''
    return t.replace('\\bibliographystyle',support+'\\bibliographystyle')
def edit_zh(t):
    t=reorder_tables(t,'zh').replace('传统两步法S2','两步规划法').replace('S2','两步规划法')
    t=t.replace('\\usepackage{float}', '\\usepackage{float}\n\\usepackage{placeins}')
    t=t.replace('\\section{结论}', '\\FloatBarrier\n\\section{结论}')
    t=caption(t,'fig:comparison_arch','传统两步规划与J-NNC联合运动规划的对比。策略示意图包含14维当前输入和三个协调动作，KCM将组合请求转换为可行的平面运动。')
    t=caption(t,'fig:framework','J-NNC闭环框架。固定前向几何与当前运动量构成14维状态；策略输出转向、进给和自适应前瞻请求，KCM确定可行平面指令。外围虚线表示学习与精化阶段对策略参数的更新。')
    t=paragraph(t,'\\figref{fig:full_utilization}将受约束',r'\figref{fig:full_utilization}在既定的$90^\circ$角点C02处，用共同坐标直接比较三种方法。J-NNC在相同加速度和加加速度约束下，维持了更高的过弯速度。右侧积分亏损同时计入降速深度与持续时间：相较两步规划法降低86.1\%，相较两动作控制器降低36.7\%。完整V/A/J记录及最终停稳曲线保留在\figref{fig:complete_support}中。')
    t=t.replace('局部利用情况见\\figref{fig:local_shapes}。','局部利用情况见\\figref{fig:full_utilization}。')
    t=caption(t,'fig:full_utilization',r'既定$90^\circ$/C02角点的直接比较，均使用相同的1 ms执行记录。（a）至（c）按参考进度对齐速度、归一化向量加速度和归一化向量加加速度，水平点线为共同上限；（d）给出相同$\pm10$ mm窗口内的积分速度亏损及最低速度。曲线展示$\pm20$ mm范围，包含入弯与出弯恢复过程。')
    t=paragraph(t,'\\figref{fig:local_shapes}将上述',r'\figref{fig:local_shapes}将运动表现与实际角点轨迹对应起来。首行在入弯切向坐标系中显示既定的$135^\circ$、$90^\circ$和$45^\circ$角点，纵轴正向为内法向；第二行放大虚线框内的过渡区域。各行采用相同尺度和等比例坐标，保证几何形状可比。J-NNC在到达顶点前开始向内转向，将方向变化分布在允差带内，与\figref{fig:full_utilization}中较高的过弯进给相对应。')
    t=caption(t,'fig:local_shapes',r'既定$135^\circ$/C01、$90^\circ$/C02和$45^\circ$/C06角点的实际局部轨迹。（a）至（c）展示入弯及出弯恢复，虚线框对应（d）至（f）的放大区域。阴影为1.25 mm欧氏允差带，各行采用共同尺度与等比例坐标。所有图均使用同一冻结模型及1 ms执行记录。')
    t=t.replace('局部利用率曲线说明了轨迹形状对速度的影响。','轨迹细节和局部利用率曲线共同说明了路径形状对速度的影响。')
    support=r'''
\begin{figure}[p]
\centering
\renewcommand{\thefigure}{C.\arabic{figure}}
\setcounter{figure}{0}
\includegraphics[width=0.98\linewidth,height=0.75\textheight,keepaspectratio]{Figure_C1.pdf}
\caption{1 ms下的完整长路径记录。各列依次为J-NNC、两动作控制器和两步规划法，各行为归一化向量速度、加速度与加加速度。显示降采样保留极值。底部子图给出完整制动及至少20 ms的物理停稳过程，为正文的局部比较与完整执行统计提供全程依据。}
\label{fig:complete_support}
\end{figure}
\clearpage

'''
    support=support.replace('\\begin{figure}[p]','\\begin{figure}[!htbp]')
    return t.replace('\\bibliographystyle',support+'\\bibliographystyle')
def main():
    for lang,edit in [('en',edit_en),('zh',edit_zh)]:
        src=PREV/f'latex_source_{lang}';dst=OUT/f'latex_source_{lang}';dst.mkdir(exist_ok=True)
        (dst/'figures_lowdim_revision').mkdir(exist_ok=True)
        name='manuscript.tex' if lang=='en' else 'manuscript_zh.tex';old=(src/name).read_text(encoding='utf-8');new=edit(old)
        assert not re.search(r'\bS2\b',new)
        assert new.split('\n\\appendix\n')[0].count('\\begin{figure}')==9
        assert new.count('\\begin{figure}')==10
        (dst/name).write_text(new,encoding='utf-8')
        for p in src.iterdir():
            if p.suffix in ['.bib','.bbl','.cls','.bst']:shutil.copy2(p,dst/p.name)
        for p in FIG.glob('*.pdf'):shutil.copy2(p,dst/'figures_lowdim_revision'/p.name)
        diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='prior_delivery/'+name,tofile='academic_revision/'+name))
        (OUT/f'changes_{lang}.diff').write_text(diff,encoding='utf-8')
    write(OUT/'editorial_scope.json',dict(prior_delivery=str(PREV/'delivery_manifest.json'),main_figures=9,supporting_figures=1,
        model_changed=False,method_equations_changed=False,reference_list_changed=False,
        changes_outside_experiments='Two-step planner naming throughout; original Figure1/3 vector-label correction and captions; complete1ms motion plot added to AppendixC.',
        main_result_tables=3,parameters_remain_in_appendix=True,original_figure2_unchanged=True))
    print('BILINGUAL_SOURCES_READY',str(OUT))
if __name__=='__main__':main()
