"""Author-facing change guide; never inserted into the manuscript."""
import json,re,sys,subprocess,shutil
from pathlib import Path
from revise_original_vectors_academic import OUT,FIG,WORK,write,sha
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def locations(info):
    text=Path(info['pdf']).with_suffix('.aux').read_text(encoding='utf-8')
    return {k:dict(number=n,page=int(p)) for k,n,p in re.findall(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^}]+)\}\{(\d+)\}',text)}
def main():
    inv=read(OUT/'render_inventory_delivery.json');inv['zh']=read(OUT/'render_inventory_compact.json')['zh'];write(OUT/'render_inventory.json',inv)
    loc={k:locations(v) for k,v in inv.items()};write(OUT/'figure_table_locations.json',loc)
    rows=[('fig:comparison_arch','1','原框架矢量修订：14维输入、三动作、平面KCM'),('fig:feasible_region','2','原允差带示意，保持不变'),('fig:framework','3','原执行链矢量修订及训练反馈线改道'),('fig:case_execution','4','原长路径、极值几何及全部34角'),('fig:nnc_failure','5','NNC与J-NNC的向量A/J可行性'),('fig:full_utilization','6','C02同轴V/A/J比较与积分亏损'),('fig:local_shapes','7','135°、90°、45°轨迹及二次放大'),('fig:all_corners','8','全部34角速度亏损与全程巡航'),('fig:ablation_dense','9','密集案例及两动作比较'),('fig:complete_support','C.1','附录：完整V/A/J与物理停稳')]
    md=f'''# 图件优化与实验之外的修改说明

本轮英文{inv['en']['pages']}页，中文{inv['zh']['pages']}页。正文保留九幅主图；完整V/A/J与停稳曲线作为附录图C.1。唯一原18页基稿、历史修订稿和原图文件均保留。没有新增训练、仿真、模型选择或实验条件变化。

## 1. 两步法的名称与主方法顺序

英文统一使用 **Two-step planner**，中文使用 **两步规划法**。其含义是先平滑几何路径，再进行满足加加速度约束的进给调度。正文、附录和图表不再用S2作为显示名称；证据文件的旧键名仍保留，以维持数据追溯。

所有比较图图例、图6(d)统计行、图8巡航表、正文表1和表2均先列J-NNC。主方法使用蓝色实线；两动作方法使用橙色点线；两步规划法使用灰色虚线。结果表只强调J-NNC的方法名和相应速度或时间结果，不把并非最优的几何距离强行标为最优。

## 2. 图6为什么这样改

原图把三种方法分列展示，读者需要跨列比较速度下降；原(d)只显示各方法完成制动，不能说明效率优势。本轮将同一C02角点的三种方法放入共同坐标：左侧依次为实际速度、向量加速度利用率、向量加加速度利用率，横轴均为角点前后的参考进度。右侧(d)改为该角的积分速度亏损，J-NNC位于第一行。

| 方法 | 最低速度（mm/s） | 积分亏损（mm） |
|---|---:|---:|
| J-NNC | 80.757 | 1.403 |
| Two-action | 77.853 | 2.217 |
| Two-step planner | 46.040 | 10.080 |

积分统计采用原先登记的±10 mm窗口，曲线展示±20 mm以看清入弯与恢复。J-NNC在该角的亏损相较两步规划法降低86.1%，相较两动作方法降低36.7%。这是局部亏损收益，不能写成完整时间降幅。C02是此前已确定的代表角，本轮没有换模型或重新选择更有利的记录。图8继续覆盖全部34角，完整时间仍由正文表1给出。

完整长路径V/A/J和20 ms停稳没有删除，移至附录图C.1。A/J的作用是说明同一运动学边界下的执行情况，不把越接近上限一律解释为越好。

## 3. 当前图号与页码

| 图 | 当前内容 | 英文页 | 中文页 |
|---|---|---:|---:|
'''
    for k,n,desc in rows:md+=f'| {n} | {desc} | {loc["en"][k]["page"]} | {loc["zh"][k]["page"]} |\n'
    md+='''
图4保留真实10 mm网格、最长和最短连续直线、最大和最小内夹角及典型圆弧；删去原(b)中多余的“10 mm”尺寸箭头。图5的J-NNC图例置前，NNC仍显示同一前80 ms窗口内的真实越限。图7采用三列代表角与两级局部放大，同一行统一尺度，避免全路径重叠；45°放大范围覆盖主方法的转弯最内侧。图8不使用大块填色，采用角点线图和累计非巡航时间。图9保留全部84角累计亏损及90°、45°局部轨迹。

## 4. 原图1、2、3的实际处理

本轮已按“能可靠修改现有PDF就直接修改”的最新授权操作副本。原PDF含矢量路径和文字，能够移除目标旧对象、保留原框架并加入新的矢量文字及局部节点；没有把图转换成位图，也没有用白块遮盖旧标签。原文件没有覆盖。

### 图1

| 原内容 | 本轮内容与理由 |
|---|---|
| J-NNC end-to-end Planning | J-NNC joint motion planning，对应联合规划而非未经执行投影的直接输出。 |
| PPO Agent | Learned policy: static features + network，与最终策略表示一致。 |
| X1…Xn及两个Y输出 | s1、s2、s3…s14及uθ、uv、uL三个输出；输入与隐藏节点沿用原结构。 |
| 不一致的层编号 | 简化为14 inputs、Hidden layer、Hidden layer、3 actions，不把示意节点数当作网络宽度。 |
| 仅Jerk Constraint及反算 | Planar V/A/J projection与Feasible acceleration release。 |
| Safe Servo Command | Feasible motion command，与平面执行量一致。 |
| Interpretation | Interpolation，修正插补术语。 |
| 紫色箭头、荧光绿模块 | 保留位置与结构，改为克制的蓝色箭头和浅色KCM条。 |

### 图2

保持原矢量PDF逐字节不变。原图用ε表示总带宽、ε/2表示单侧允差，与正文定义一致，本轮没有必须改动的技术内容。

### 图3

| 原内容 | 本轮内容与理由 |
|---|---|
| Contour Tolerance ε | Full band width ε，明确总宽度。 |
| 模糊的多尺度前瞻描述 | Ordered projection、Fixed forward geometry、2/6/15/30 mm，区分固定观测距离和动作前瞻。 |
| 旧分块状态式 | s∈R14；Tracking error、Planar kinematics、Forward geometry。 |
| PPO Policy Network与随机策略式 | Learned three-action policy及u=fθ(s)，对应部署时的确定性策略。 |
| 动作输出 | u=[uθ,uv,uL]，Decode(α,F,L)，Held for20 ms。 |
| 标量裁剪和反算框 | V/A/J projection、Acceleration release、Next velocity selection；约束式改为向量范数。 |
| Safe Command与ω、v | Feasible command与二维下一步速度v(k+1)。 |
| 状态更新中的η | 改为位置、速度/加速度、有序进度及轮廓距离。 |
| Progress/Tracking/Smoothness/Time奖励列举 | Elapsed time、Progress potential、Failure penalty，与正文时间目标一致。 |
| 穿过KCM的训练虚线 | 改为绕框架外围进入策略的参数更新线；明确PPO学习和离线精化只更新策略参数。 |

网络小图在原模块位置简化重绘，原模块边界、执行箭头、反馈框架和页面尺寸保留。图内细小标签字号经过打印尺寸检查。

## 5. 实验之外具体改了什么

1. 引言中的图1图注随新标签修订，说明14维输入、三动作与KCM；没有重写研究动机与贡献逻辑。
2. 方法章节仅修改图3本身及其图注。状态、动作、奖励、KCM的正文段落和公式与上一轮一致。
3. 全文和附录的S2显示名称改为Two-step planner/两步规划法，包括结论中的对应结果表述；数字不变。
4. 附录C增加完整1 ms执行图C.1，原数值分辨率检查保留。正文仍以1 ms为主，不反复展开0.5 ms检查。
5. 中文稿增加浮动图边界，保证图7—9留在实验部分，不能漂移到结论或附录。
6. 标题、原文章节顺序、文献库及24项引用保持。结果表仍在实验正文；参数表和KCM算法仍在附录。

本说明、探索过程、失败修复及逐轮记录均不写入论文。正文只保留支持结论所需的实验条件、结果及必要证据边界。

## 6. Illustrator中继续使用

- 图1和图3的修订PDF可直接作为后续矢量编辑文件；它们是在原导出PDF的矢量对象上修改的副本。
- 没有获得原Illustrator SVG/AI源文件，因此不声称保留了作者原来的图层、编组名称和编辑历史。
- Figure_1_editable_labels.svg和Figure_3_editable_labels.svg只包含本轮新增文字和局部节点，画布尺寸与对应PDF一致；它们不是完整原图SVG。
- 图4—9和C.1同时提供完整PDF及带可编辑文本的SVG。绘图CSV及来源绑定在图件包内，不能拿不同模型的曲线替换。
- 原来“作者必须自行修改图1/3旧标签”的待办已由本轮矢量副本修改完成。此前详细指南的旧标签待办和旧图号页码，以本说明为准。

## 7. 核验与维护

已核对原文件保留、当前图件与中英文源码一致、引用与公式一致、方法名统一、正文九图及三张结果表、附录单幅完整执行图、统计值与既有轨迹绑定。逐页视觉检查包括所有变动页，其余页面用像素一致性核验对应已检查页。

统一方法命名、配色和统计来源以减少重复（KISS/DRY）；仅完成本轮图文修订，不扩展训练或方法（YAGNI）；原矢量修改、数据绘图、文稿更新、交付核验分开处理（单一职责）。
'''
    (OUT/'图件优化与非实验修改说明.md').write_text(md,encoding='utf-8')
    # A concise four-page companion PDF; detailed exact changes remain in the Markdown guide.
    table='\n'.join(f'{n} & {desc} & {loc["en"][k]["page"]} / {loc["zh"][k]["page"]}\\\\' for k,n,desc in rows)
    tex=r'''\documentclass[UTF8,10pt]{ctexart}
\usepackage[a4paper,margin=20mm]{geometry}
\usepackage{graphicx,booktabs,tabularx,enumitem,amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\setlist{nosep,leftmargin=1.7em}
\begin{document}
\begin{center}\LARGE 图件优化与非实验修改说明\end{center}
\textbf{本轮交付：}英文ENPAGES页，中文ZHPAGES页；正文九幅主图，附录一幅完整执行图。没有新增训练、仿真或更换模型。本文档仅供作者核对，不进入论文正文或附录。
\section*{1. 图6现在如何呈现优势}
将三种方法放入相同坐标，按同一C02角点的参考进度比较实际速度、向量加速度利用率与向量加加速度利用率。原(d)停稳图改为积分速度亏损，J-NNC位于第一行。
\begin{center}\begin{tabular}{lrr}\toprule 方法 & 最低速度(mm/s) & 积分亏损(mm)\\\midrule
J-NNC &80.757&1.403\\Two-action&77.853&2.217\\Two-step planner&46.040&10.080\\\bottomrule\end{tabular}\end{center}
积分统计采用既定的$\pm10$ mm窗口，曲线展示$\pm20$ mm以包含入弯和恢复。J-NNC的局部亏损相较两步规划法降低86.1\%，相较两动作方法降低36.7\%。这是局部亏损收益，不是全程时间降幅；C02也是此前已确定的代表角。
全34角统计、全程巡航及完整用时仍保留。完整V/A/J与20 ms停稳移至附录图C.1。A/J利用率用于解释共同物理边界，不将更接近上限一律视为更好。
\section*{2. 名称、顺序与当前图件}
两步法统一称\textbf{Two-step planner／两步规划法}，即几何平滑后进行加加速度约束进给调度。比较图图例、统计行和主要结果表均先列J-NNC；蓝色实线表示J-NNC、橙色点线表示两动作、灰色虚线表示两步规划法。
\begin{tabularx}{\linewidth}{lXr}\toprule 图 & 内容 & 英/中页\\\midrule
TABLE
\bottomrule\end{tabularx}
\newpage
\section*{3. 原图1：在原矢量PDF副本上完成修订}
\begin{center}\includegraphics[width=.86\linewidth]{FIGPATH/Figure_1.pdf}\end{center}
\begin{tabularx}{\linewidth}{p{.29\linewidth}X}\toprule 原内容 & 现在的内容与依据\\\midrule
end-to-end Planning & joint motion planning，避免将带执行投影的联合规划画成直接无约束输出。\\
PPO Agent & Learned policy: static features + network，与最终策略表示一致。\\
X1\ldots Xn、两个Y输出 & 14维当前输入；$u_\theta,u_v,u_L$三个协调动作。\\
Layer 0/1/3/4 & 14 inputs、两个Hidden layer、3 actions；示意图不暗示实际层宽。\\
Jerk Constraint／反算 & Planar V/A/J projection；Feasible acceleration release。\\
Safe Servo Command & Feasible motion command，符合平面执行量。\\
Interpretation & Interpolation，修正插补术语。\\
紫色箭头、荧光绿条 & 沿用位置与连接，改为蓝色箭头和浅色KCM条。\\\bottomrule\end{tabularx}
\textbf{原图2保持不变。}其总宽度$\varepsilon$与单侧$\varepsilon/2$定义正确，修订稿使用的PDF与原图逐字节一致。
\newpage
\section*{4. 原图3：执行链和当前实现对齐}
\begin{center}\includegraphics[width=\linewidth]{FIGPATH/Figure_3.pdf}\end{center}
\begin{tabularx}{\linewidth}{p{.28\linewidth}X}\toprule 位置 & 已完成修改\\\midrule
几何输入 & Full band width $\varepsilon$；Ordered projection；固定2/6/15/30 mm前向几何。\\
状态 & $s\in\mathbb R^{14}$，当前跟踪误差、平面运动学与前向几何。\\
策略 & Learned three-action policy，部署映射$u=f_\theta(s)$；原模块内的网络小图简化为三输出。\\
动作 & $u=[u_\theta,u_v,u_L]$，解码$(\alpha,F,L)$，请求保持20 ms。\\
KCM & 向量V/A/J投影、加速度可行释放、下一步速度选择；约束均用向量范数。\\
执行指令 & Feasible command；二维速度$v_{k+1}\in\mathbb R^2$，移除旧$\omega,v$标记。\\
状态更新 & 位置、速度/加速度、有序进度、轮廓距离；移除旧干预状态$\eta$。\\
奖励 & Elapsed time、Progress potential、Failure penalty。\\
训练反馈 & 虚线从外围直接进入策略，不再穿过KCM；标明PPO学习与离线精化只更新策略参数。\\\bottomrule\end{tabularx}
\textbf{保留范围：}原模块边界、执行箭头、反馈结构与页面尺寸保留。旧文字／公式对象确实移除，没有通过位图或白色遮盖隐藏；原文件和历史版本不覆盖。
\newpage
\section*{5. 图4至图9的呈现调整}
\begin{itemize}
\item 图4：保留10 mm真实网格、最长／最短直线、最大／最小内夹角及典型圆弧；删除(b)中多余的10 mm尺寸标记。
\item 图5：保留相同前80 ms真实记录，统一J-NNC优先图例。主文前1 s峰值与图中80 ms峰值分别说明。
\item 图6：同轴比较C02速度及A/J约束使用，右侧量化积分亏损，替换原停稳子图。
\item 图7：135°、90°、45°三列，轨迹范围与第二级放大一一对应；同一行使用共同尺度和等比例坐标。
\item 图8：全部34角亏损和累计非巡航时间，不铺大块填色；巡航表J-NNC置首并突出。
\item 图9：保留全84角累计速度亏损和90°、45°局部轨迹，同一冻结模型与对照。
\end{itemize}
\section*{6. 实验之外的修改边界}
\begin{enumerate}
\item 图1图注说明14维输入、三动作和KCM；研究动机、贡献逻辑及标题保留。
\item 方法章节仅修改图3及图注；状态、动作、奖励、KCM的段落和公式保持上一轮版本。
\item 全文及附录替换S2显示名称，数字不变；两步规划法的具体含义在实验设置中保留。
\item 附录C增加1 ms完整执行图C.1。正文仍以1 ms为主，0.5 ms检查集中在附录。
\item 中文稿限制浮动图边界，确保全部实验主图位于实验部分。
\item 文献库、24项引用和原章节顺序保留。三张结果表在正文，参数表和KCM算法在附录。
\end{enumerate}
探索过程、失败修复、逐轮记录及本说明均不进入文稿。正文只保留支持结论所需的条件、结果与必要证据边界。
\section*{7. 后续在Illustrator中编辑}
修订图1/3的PDF仍是矢量文件，可直接作为后续编辑基础。没有原Illustrator SVG/AI源文件，因此不声称恢复原图层名称、编组或编辑历史。两个editable\_labels.svg文件仅含本轮新增标签与局部节点，画布尺寸与PDF一致；它们不是完整原图SVG。图4—9和C.1提供完整PDF与可编辑文本SVG。

原“作者自行修改图1/3旧标签”的待办已在本轮完成。更详细的逐项替换说明见同名Markdown；此前指南中的旧标签待办及旧图号页码由本说明替代。
\end{document}
'''
    tex=tex.replace('ENPAGES',str(inv['en']['pages'])).replace('ZHPAGES',str(inv['zh']['pages'])).replace('TABLE',table).replace('FIGPATH',FIG.as_posix())
    source=WORK/'academic_revision_notes.tex';source.write_text(tex,encoding='utf-8')
    build=WORK/'notes_build';build.mkdir(exist_ok=True)
    cmd=['D:/texlive/2025/bin/windows/xelatex.exe','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),str(source)]
    for _ in range(2):
        r=subprocess.run(cmd,capture_output=True);(build/'compile.txt').write_bytes(r.stdout+r.stderr);assert r.returncode==0,(r.stdout+r.stderr).decode('utf-8',errors='replace')[-2000:]
    dest=OUT/'图件优化与非实验修改说明.pdf';shutil.copy2(build/'academic_revision_notes.pdf',dest)
    sys.path.append('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
    import pypdfium2 as pdfium
    doc=pdfium.PdfDocument(str(dest));renders=[]
    for i,p in enumerate(doc):
        out=WORK/f'notes_page_{i+1}.png';p.render(scale=1.4).to_pil().save(out);renders.append(str(out))
    issues=re.findall(r'^.*(?:Overfull|Missing character).*$',(build/'academic_revision_notes.log').read_text(encoding='utf-8',errors='replace'),re.M)
    write(OUT/'notes_render_inventory.json',dict(pages=len(doc),renders=renders,compile_issues=issues,pdf_sha256=sha(dest)))
    print('NOTES',len(doc),'pages',issues,flush=True)
if __name__=='__main__':main()
