"""Bilingual editorial revision; keep methods, citations and protected figures."""
from pathlib import Path
import json, re, shutil
from revise_nnc_utilization_figures import OLD, OUT, FIG, sha, write

def utilization_table(lang):
    stats=json.loads((OUT/'utilization_statistics.json').read_text(encoding='utf-8'))
    en=lang=='en'
    s=(r'''\begin{table}[htbp]
\centering
\caption{Kinematic-limit utilization over complete executions at 1 ms. All entries are percentages of the common vector limits.}
\label{tab:utilization}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llrrr}
\toprule
Case & Method & Mean $V$ use & 95th-pct. $A$ use & 95th-pct. $J$ use\\
\midrule
''' if en else r'''\begin{table}[htbp]
\centering
\caption{1 ms下完整执行的运动学约束利用率。所有数值均为对应平面向量上限的百分比。}
\label{tab:utilization}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{llrrr}
\toprule
案例 & 方法 & 平均$V$利用率 & $A$利用率95分位 & $J$利用率95分位\\
\midrule
''')
    for i,r in enumerate(stats['records']):
        if i==3:s+='\\midrule\n'
        case=(('Long route' if r['case']=='original' else 'Dense case') if en else ('长路径' if r['case']=='original' else '密集案例')) if i%3==0 else ''
        method=r['method'] if en or r['method']!='Two-action' else '两动作'
        s+=f"{case} & {method} & {r['mean_V_percent']:.3f} & {r['p95_A_percent']:.3f} & {r['p95_J_percent']:.3f}"+r'\\'+'\n'
    s+=r'''\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.98\linewidth}\footnotesize
'''
    s+=('Statistics use all equally spaced samples, including startup, braking and rest. The 95th percentiles describe the complete-time distribution, not peak constraint use.\n' if en else '统计包含全部等间隔样本，包括启动、制动和停稳。95分位数描述完整时间内的分布，不代表约束利用率峰值。\n')
    s+=r'''\end{minipage}
\end{table}

'''
    s+=('Table~\\ref{tab:utilization} summarizes the same complete records. On the dense case, mean feedrate use increases from 67.139\\% for S2 and 86.528\\% for the two-action controller to 91.071\\% for J-NNC. Both learned controllers reach 99.900\\% at the acceleration and jerk 95th percentiles, while J-NNC converts that available motion capacity into higher mean feedrate. The low long-route acceleration and jerk percentiles reflect its dominant cruising duration; local utilization is resolved in \\figref{fig:local_shapes}.\n\n' if en else '\\tabref{tab:utilization}汇总相同的完整记录。密集案例中，平均进给利用率由S2的67.139\\%和两动作控制器的86.528\\%提高至J-NNC的91.071\\%。两个学习型控制器的加速度和加加速度利用率95分位数均达到99.900\\%，J-NNC将这些可用运动能力转化为更高的平均进给。长路径的加速度和加加速度95分位数较低，是因为巡航占据大部分执行时间；局部利用情况见\\figref{fig:local_shapes}。\n\n')
    return s

EN = r'''\section{Experiments and Results}
\label{sec:experiments}
\subsection{Experimental Settings}
The experiments examine whether J-NNC produces feasible motion and improves complete efficiency through joint trajectory shaping and feedrate regulation. The original closed route is retained in full: its reference length is 44.420514 m and all 34 corners enter the analysis. Figure~\ref{fig:case_execution} shows the geometry, including the longest and shortest continuous straights (2262.742 and 80 mm), the extreme included angles ($45.000^\circ$ and $175.679^\circ$), and the two $90^\circ$ arcs of radii 600 and 800 mm. Adjacent codirectional collinear blocks are merged when measuring continuous straights, including across the closed-path seam. The 10 mm Cartesian grid is resolved in the enlarged short-straight view. An additional 1438.009 mm closed case contains 84 corners (\figref{fig:ablation_dense}a). These cases emphasize sustained cruising and successive corner transitions, respectively.

All primary results use a 1 ms interpolation interval. The full contour-band width is $\varepsilon=2.5$ mm, giving a one-sided Euclidean tolerance of 1.25 mm. The common planar vector limits are $V_{\max}=100$ mm/s, $A_{\max}=2000$ mm/s$^2$ and $J_{\max}=62500$ mm/s$^3$. The learned constrained controllers update their requests every 20 ms. Complete success requires ordered coverage of the entire path, geometric feasibility over each command interval, complete braking and at least 20 ms of physical rest. All 2362 ordered gates and the final segment of the long route are retained. A supplementary 0.5 ms check confirms that the reported time gains exceed execution-resolution effects (\ref{app:resolution}).

The NNC baseline retains the direct two-action network and execution scheme of the original comparison, motivated by learned CNC smoothing~\cite{Li2020}. It uses its original 14-input observation and 1 ms decision period without KCM projection. Its planar adaptation removes angular-derived feed caps while preserving the learned action mapping. This baseline assesses command feasibility. Among feasible planners, the traditional two-step baseline S2 first smooths the geometry and then schedules jerk-limited motion. Its quintic B\'ezier corner transitions are adapted from Sencer et al.~\cite{Sencer2014}, while circular arcs are retained analytically. No global feedrate reduction is imposed on a comparator.

The principal ablation retains steering and feedrate but removes the learned preview output. It uses the same 14 observation channels and physical executor, with corresponding static-feature capacity and separate fitting, parameter selection and PPO opportunities. The selected two-action controller completes both cases. The dense case is also compared with the fastest valid time among all evaluated two-action models, including valid dense runs from models that did not complete the long route.

For long-route corner $c$, integrated speed deficit is
\begin{equation}
D_c=h\sum_{k:\,|\ell_k-\ell_c|\leq10\ \mathrm{mm}}\max(100-\|v_k\|_2,0).
\label{eq:corner_deficit}
\end{equation}
This measure, in millimetres, combines the depth and duration of deceleration. All 34 complete windows are retained, with numerical ties allowed within $10^{-7}$ mm. Near-limit cruising means $\|v_k\|_2\geq99$ mm/s throughout this study. The utilization curves show $\|v\|_2/V_{\max}$, $\|a\|_2/A_{\max}$ and $\|j\|_2/J_{\max}$, reconstructed from executed XY positions using Eq.~\eqref{eq:physical_differences}. A ratio of one denotes the corresponding limit; vector norms include the motion required to change direction.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth,height=0.77\textheight,keepaspectratio]{Figure_4.pdf}
\caption{Closed-path geometry and angle coverage: (a) full route with the longest and shortest continuous straights, extreme included angles and typical circular arcs; (b) shortest-straight detail on the 10 mm grid; (c) all 34 included angles. C01, C02 and C06 are used for the local comparisons.}
\label{fig:case_execution}
\end{figure}

\subsection{Main Comparative Results}
J-NNC completes both cases within the common geometric and planar kinematic limits (\tabref{tab:main_results}). On the long route, complete time is 444.177 s, saving 1.008 s relative to S2 and 0.537 s relative to the selected two-action controller. On the dense case, J-NNC takes 14.839 s compared with 19.957 s for S2, a reduction of 25.65\%. These results include startup, all corner transitions, braking and terminal rest.

\begin{table}[htbp]
\centering
\caption{Feasibility and complete execution at a 1 ms interpolation interval. Successful methods satisfy full ordered coverage, command-interval geometry, planar V/A/J limits and terminal rest.}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lllrr}
\toprule
Case & Method & Feasible & Complete time (s) & Max. distance (mm)\\
\midrule
Long route & NNC & No & --- & ---\\
 & S2 & Yes & 445.185 & 1.243758\\
 & Two-action & Yes & 444.714 & 1.095198\\
 & J-NNC & Yes & 444.177 & 1.221862\\
\midrule
Dense case & NNC & No & --- & ---\\
 & S2 & Yes & 19.957 & 1.149675\\
 & Two-action & Yes & 15.875 & 1.121741\\
 & J-NNC & Yes & 14.839 & 1.154901\\
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.98\linewidth}\footnotesize
Maximum distance covers complete command intervals. NNC fails the acceleration and jerk bounds; dashes indicate that no admissible complete result is available.
\end{minipage}
\end{table}

The NNC comparison makes the execution constraint explicit. Figure~\ref{fig:nnc_failure} shows the same first 80 ms on both cases: direct NNC commands cross the acceleration and jerk limits, with repeated violations after the shaded initial 20 ms. Over the retained first second, NNC acceleration peaks at 20.70 and 29.54 times the limit on the long and dense cases, respectively; jerk peaks at 841.69 and 1150.81 times the limit. J-NNC remains within the bounds. These prefix records establish NNC infeasibility, rather than a valid complete-time comparison.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{Figure_5.pdf}
\caption{NNC constraint violations at 1 ms: normalized vector acceleration (a,b) and jerk (c,d) over the same first 80 ms of the long and dense cases. Shading marks the initial 20 ms; violations continue beyond it. Dashed lines denote the common limits. The vertical axes are linear from zero to one and logarithmic above one. Peak annotations refer to the displayed window. Curves are reconstructed from recorded XY positions; no complete NNC execution time is inferred.}
\label{fig:nnc_failure}
\end{figure}

Figure~\ref{fig:full_utilization} extends the constrained comparison to the entire long route. Separate columns prevent overlapping full-duration traces from obscuring the motion. All three feasible controllers respect the common vector limits. J-NNC sustains high feedrate use with shallower or fewer local speed reductions; acceleration and jerk are concentrated around transitions. The final panel resolves braking and the stationary tail, so the reported completion times include the physical stop.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth,height=0.75\textheight,keepaspectratio]{Figure_6.pdf}
\caption{Complete long-route V/A/J utilization at 1 ms. Columns (a)--(c) show S2, the two-action controller and J-NNC; rows show normalized vector speed, acceleration and jerk. A value of one is the common limit. Panel (d) enlarges final braking and physical rest. Display reduction preserves extrema from the full records.}
\label{fig:full_utilization}
\end{figure}

Figure~\ref{fig:local_shapes} relates this behavior to the executed corner shape. The upper row enlarges the prescribed $135^\circ$, $90^\circ$ and $45^\circ$ corners in the incoming-tangent frame, with the inward normal as the vertical axis. Equal aspect and common scales preserve the geometric comparison. The lower rows align V/A/J utilization by reference progress through the same corners. J-NNC turns inward before the vertex and spreads the direction change through the tolerance band, allowing higher feedrate use while remaining within the acceleration and jerk bounds.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth,height=0.77\textheight,keepaspectratio]{Figure_7.pdf}
\caption{Local trajectory and dynamic-limit utilization at the prescribed $135^\circ$/C01, $90^\circ$/C02 and $45^\circ$/C06 corners. Panels (a)--(c) show executed trajectories and the 1.25 mm corridor, with equal aspect and common scales. Panels (d)--(f) and the two rows below show normalized vector speed, acceleration and jerk over $\pm20$ mm of reference progress. Zero marks the corner. All curves use the 1 ms records; integrated corner losses use the $\pm10$ mm windows in Eq.~\eqref{eq:corner_deficit}.}
\label{fig:local_shapes}
\end{figure}

The $135^\circ$ corner is crossed at 100 mm/s. At the $90^\circ$ corner C02, the minimum speed is 80.76 mm/s for J-NNC, 46.04 mm/s for S2 and 77.85 mm/s for the two-action controller. Integrated speed deficit decreases from 10.080 mm for S2 and 2.217 mm for the two-action method to 1.403 mm for J-NNC. At the sharper C06 corner, the respective deficits are 13.321, 5.963 and 4.253 mm. The integrated measure captures both the depth and duration of speed reduction, including recovery. Upstream inward turning is present at all 34 long-route corners and all 84 dense-case corners in the executed records.

The local utilization curves explain why trajectory shape matters. Normal acceleration satisfies $a_n=v^2\kappa$, while turning jerk also depends on the variation of curvature along the executed path. Steering and preview distribute the direction change, and feedrate regulates the demand on the available acceleration and jerk. A bounded outward offset during recovery leaves room to complete this transition inside the corridor. The controller is therefore not forced to return immediately to the outgoing reference line. The displayed paths satisfy the common limits and reduce integrated speed loss; outward displacement itself is not rewarded.

Figure~\ref{fig:all_corners} extends the examples to every corner. J-NNC has no greater registered speed deficit than S2 or the trained two-action controller at any of the 34 corners, allowing numerical ties. Summed corner deficit falls from 114.203 mm for S2 and 42.107 mm for the two-action method to 13.902 mm for J-NNC. The full-route cumulative curves remain flat during near-limit cruise, making additional deceleration in the second half of the lap visible.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{Figure_8.pdf}
\caption{All-corner speed loss and full-route cruising at 1 ms: (a) integrated speed deficit in every $\pm10$ mm corner window, with the representative corners shaded; (b) accumulated time below 99 mm/s. The table gives time at or above 99 mm/s, its share of complete time and the longest uninterrupted interval. All 34 corners and the terminal stop are retained.}
\label{fig:all_corners}
\end{figure}

J-NNC accumulates 442.552 s at or above 99 mm/s, compared with 441.238 s for S2 and 440.299 s for the two-action controller. Its longest uninterrupted interval is 111.753 s, compared with 71.569 and 71.524 s. Reduced corner loss thus supports longer sustained cruising as well as a shorter complete time.

To distinguish path shortening from speed effects, let $L_{\mathrm{exec}}=h\sum_k\|v_k\|_2$ and $D_{\mathrm{all}}=h\sum_k(100-\|v_k\|_2)$. Then, up to floating-point roundoff,
\begin{equation}
T=\frac{L_{\mathrm{exec}}+D_{\mathrm{all}}}{100}.
\label{eq:time_decomposition}
\end{equation}
The stationary tail is included in $D_{\mathrm{all}}$. J-NNC travels 44390.406 mm with a total speed deficit of 27.294 mm; S2 travels 44392.151 mm with a deficit of 126.349 mm. Of the 1.008 s saving, approximately 0.991 s comes from reduced speed loss and 0.017 s from shorter travel. Against the two-action model, the 0.537 s saving comprises 0.303 s from lower speed loss and 0.234 s from shorter travel. The dense case examines the same benefit when successive transitions occupy more of the task.

\subsection{Ablation Results}
The developed three-action controller completes the dense case in 14.839 s, compared with 15.875 s for the selected two-action model, a 6.53\% reduction under the same physical constraints. Even against the fastest valid two-action time among all evaluated models, 15.798 s, J-NNC saves 0.959 s (6.07\%).

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{Figure_9.pdf}
\caption{Two-action comparison at 1 ms: (a) dense closed reference and local comparison locations; (b) cumulative speed deficit over the complete execution, including all 84 corners and stopping; (c,d) executed $90^\circ$ and $45^\circ$ transitions. The curves use the developed controllers in \tabref{tab:main_results}.}
\label{fig:ablation_dense}
\end{figure}

The local views in \figref{fig:ablation_dense}(c,d) show the different use of the tolerance band. J-NNC sustains 4.368 s of near-limit cruising on the dense route, compared with 0.114 s for the selected two-action model. Total speed deficit decreases from 213.867 to 132.502 mm. The 1.036 s time saving comprises 0.814 s from reduced speed loss and 0.222 s from shorter travel. The efficiency benefit therefore involves both local geometry and feedrate use across the complete route. A separate matched-start, equal-budget PPO comparison does not resolve a time gain beyond numerical uncertainty (\ref{app:learning}); the larger reductions here compare the developed controllers.

Three independently initialized student models satisfy the physical, complete-time and all-34-corner criteria on both cases (\tabref{tab:selected_students}). Their dense-case gains over the fastest valid two-action result range from 5.94\% to 6.07\%. They share teacher data and feature design, with different subsequent refinement; these results represent three successful selected models rather than repetitions of one uniform training procedure. Seed 923202 is used in all main figures.

\begin{table}[htbp]
\centering
\caption{Complete times of three selected student models at 1 ms. All meet the physical, completion and all-corner criteria. Dense gain is relative to the fastest valid two-action time of 15.798 s.}
\label{tab:selected_students}
\small
\begin{tabular}{lrrr}
\toprule
Student seed & Long route (s) & Dense case (s) & Dense gain (\%)\\
\midrule
923201 & 444.178 & 14.859 & 5.94\\
923202 (main) & 444.177 & 14.839 & 6.07\\
923203 & 444.178 & 14.839 & 6.07\\
\bottomrule
\end{tabular}
\end{table}

Together, the comparisons address the two requirements of joint motion planning. J-NNC supplies feasible planar commands where the direct NNC baseline violates the kinematic bounds. Among feasible methods, anticipatory trajectory shaping and coordinated feed recovery reduce cumulative speed loss. The dense case demonstrates the benefit across successive transitions, while the original long route confirms sustained cruising and the all-corner result over a full lap.

'''

ZH = r'''\section{实验与结果}
\label{sec:experiments}
\subsection{实验设置}
实验考察J-NNC能否生成满足约束的运动，并通过轨迹整形与进给调节的协调提高完整执行效率。保留原闭合长路径的全部内容：参考长度为44.420514 m，全部34个角点进入统计。\figref{fig:case_execution}标出了最长和最短连续直线（2262.742 mm和80 mm）、最大和最小内夹角（$175.679^\circ$和$45.000^\circ$），以及半径600 mm和800 mm的两段$90^\circ$圆弧。连续直线长度在合并相邻同向共线段后计算，包含跨闭合接缝的共线段。最短直线的放大图显示10 mm十字网格。此外设置长度1438.009 mm、含84个角点的闭合密集案例（\figref{fig:ablation_dense}a）。两个案例分别突出持续巡航和连续过弯。

主要结果统一采用1 ms插补周期。允差带总宽度为$\varepsilon=2.5$ mm，单侧欧氏允差为1.25 mm；平面向量上限统一为$V_{\max}=100$ mm/s、$A_{\max}=2000$ mm/s$^2$和$J_{\max}=62500$ mm/s$^3$。学习型受约束控制器每20 ms更新一次动作请求。完整成功要求按顺序覆盖整条路径、每个运动指令区间均满足几何约束，并完成制动及至少20 ms的物理停稳。长路径保留全部2362个有序通过点和完整终段。附加的0.5 ms检查用于确认时间收益超过执行分辨率的影响，详见\ref{app:resolution}。

NNC基线沿用原比较中的两动作网络及直接执行方式，其设计参考了学习型数控轨迹平滑研究~\cite{Li2020}。该基线使用原有14维观测和1 ms决策周期，不经过KCM投影。平面适配保留学习到的动作映射，移除由角运动导出的进给上限。该比较用于检验指令可行性。对于可行规划器，传统两步法S2先平滑几何路径，再规划满足加加速度约束的运动；五次B\'ezier角点过渡参考Sencer等人的方法~\cite{Sencer2014}，圆弧保留解析形式。所有对照均不施加人为的全局进给降速。

主要消融保留转向和进给动作，去除学习型前瞻输出，使用相同的14维观测和物理执行器，并具有对应的静态特征容量及独立拟合、参数选择和PPO更新机会。选定的两动作控制器在两个案例上均完整成功。密集案例还与全部已评价两动作模型中的最快有效用时比较；某模型即使未完成长路径，其有效的密集案例结果仍纳入该比较。

长路径角点$c$的积分速度亏损定义为
\begin{equation}
D_c=h\sum_{k:\,|\ell_k-\ell_c|\leq10\ \mathrm{mm}}\max(100-\|v_k\|_2,0).
\label{eq:corner_deficit}
\end{equation}
该指标以毫米计，同时反映降速的深度和持续时间。保留全部34个完整窗口，允许$10^{-7}$ mm以内的数值并列。全文将$\|v_k\|_2\geq99$ mm/s定义为接近速度上限的巡航。利用率曲线分别为$\|v\|_2/V_{\max}$、$\|a\|_2/A_{\max}$和$\|j\|_2/J_{\max}$，由执行XY位置按式\eqref{eq:physical_differences}重建。比值为1表示达到相应上限；向量模包含改变运动方向所需的运动学分量。

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth,height=0.77\textheight,keepaspectratio]{Figure_4.pdf}
\caption{闭合路径与角度覆盖：（a）完整路径，标注最长和最短连续直线、极值内夹角及典型圆弧；（b）10 mm网格中的最短直线局部；（c）全部34个内夹角。C01、C02和C06用于局部比较。}
\label{fig:case_execution}
\end{figure}

\subsection{主要比较结果}
J-NNC在共同几何和运动学约束下完整完成两个案例（\tabref{tab:main_results}）。长路径用时444.177 s，比S2减少1.008 s，比选定两动作控制器减少0.537 s。密集案例用时14.839 s，S2为19.957 s，降幅为25.65\%。这些时间均包含启动、全部过弯、制动和终端停稳。

\begin{table}[htbp]
\centering
\caption{1 ms插补周期下的可行性与完整执行结果。成功方法均满足完整有序覆盖、运动指令区间几何约束、平面V/A/J上限及终端停稳要求。}
\label{tab:main_results}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lllrr}
\toprule
案例 & 方法 & 可行 & 完整时间（s） & 最大距离（mm）\\
\midrule
长路径 & NNC & 否 & --- & ---\\
 & S2 & 是 & 445.185 & 1.243758\\
 & 两动作 & 是 & 444.714 & 1.095198\\
 & J-NNC & 是 & 444.177 & 1.221862\\
\midrule
密集案例 & NNC & 否 & --- & ---\\
 & S2 & 是 & 19.957 & 1.149675\\
 & 两动作 & 是 & 15.875 & 1.121741\\
 & J-NNC & 是 & 14.839 & 1.154901\\
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{0.98\linewidth}\footnotesize
最大距离覆盖完整运动指令区间。NNC不满足加速度和加加速度上限，横线表示没有可用于比较的完整可行结果。
\end{minipage}
\end{table}

NNC比较直接体现了执行约束的作用。\figref{fig:nnc_failure}显示两个案例相同的前80 ms区间：NNC直接输出的指令越过加速度和加加速度上限，且在阴影标出的初始20 ms之后仍反复越限。在保留的前1 s记录中，NNC的加速度峰值在长路径和密集案例上分别达到上限的20.70倍和29.54倍，加加速度峰值分别达到841.69倍和1150.81倍。J-NNC则保持在约束以内。这些起始段记录用于判定NNC的约束失效，不构成有效的完整时间比较。

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{Figure_5.pdf}
\caption{1 ms下NNC的约束越限：长路径和密集案例相同前80 ms内的归一化向量加速度（a、b）及加加速度（c、d）。阴影为初始20 ms，其后仍存在越限。虚线为共同上限。纵轴在0至1之间为线性刻度，1以上为对数刻度；峰值标注仅对应图示窗口。曲线由记录的XY位置重建，不据此推断NNC的完整执行时间。}
\label{fig:nnc_failure}
\end{figure}

\figref{fig:full_utilization}将受约束方法的比较扩展至完整长路径。各方法分列展示，避免全程曲线重叠后掩盖运动细节。三种可行控制器均满足共同向量上限。J-NNC保持较高的进给利用率，局部降速较浅或次数较少；加速度和加加速度主要集中在过渡区域。底部子图单独展示制动及静止尾段，完整时间包含实际停稳过程。

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth,height=0.75\textheight,keepaspectratio]{Figure_6.pdf}
\caption{1 ms下完整长路径的V/A/J利用率。（a）至（c）列依次为S2、两动作控制器和J-NNC，三行依次为归一化向量速度、加速度及加加速度。比值1表示共同上限。（d）放大最终制动和物理停稳过程。全程曲线显示降采样保留原始记录中的极值。}
\label{fig:full_utilization}
\end{figure}

\figref{fig:local_shapes}将上述运动表现与实际角点轨迹对应起来。首行在入弯切向坐标系中放大既定的$135^\circ$、$90^\circ$和$45^\circ$角点，纵轴正向为内法向；相同尺度及等比例坐标保证几何形状可比。下方按参考进度对齐同一角点的V/A/J利用率。J-NNC在到达顶点前开始向内转向，将方向变化分布在允差带内，从而在满足加速度和加加速度约束的同时维持较高进给。

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth,height=0.77\textheight,keepaspectratio]{Figure_7.pdf}
\caption{既定$135^\circ$/C01、$90^\circ$/C02和$45^\circ$/C06角点的局部轨迹与运动学利用率。（a）至（c）为执行轨迹和1.25 mm允差带，采用等比例坐标及共同尺度；（d）至（f）及下方两行依次为角点前后$\pm20$ mm参考进度内的归一化向量速度、加速度和加加速度。零点表示角点。所有曲线使用1 ms记录，积分角点损失仍按式\eqref{eq:corner_deficit}的$\pm10$ mm窗口计算。}
\label{fig:local_shapes}
\end{figure}

$135^\circ$角点可保持100 mm/s通过。在$90^\circ$角点C02处，J-NNC最低速度为80.76 mm/s，S2和两动作控制器分别为46.04 mm/s和77.85 mm/s。积分速度亏损由S2的10.080 mm和两动作方法的2.217 mm降至J-NNC的1.403 mm。在转向更急的C06处，三者的亏损分别为13.321 mm、5.963 mm和4.253 mm。积分指标同时计入降速深度、持续时间和恢复过程。执行记录中，长路径全部34角和密集案例全部84角均出现角点上游的向内转向。

局部利用率曲线说明了轨迹形状对速度的影响。法向加速度满足$a_n=v^2\kappa$，转弯加加速度还取决于曲率沿执行路径的变化。转向与前瞻共同分配方向变化，进给则调节对加速度和加加速度能力的需求。出弯恢复时允许的有限外向偏移，为允差带内的过渡保留了空间，因此控制器不必立即回到出弯参考直线。图示轨迹在共同约束下减少了累计速度损失，外向偏移本身不获得奖励。

\figref{fig:all_corners}将局部示例扩展至全部角点。允许数值并列时，J-NNC在全部34个角点的既定速度亏损均不高于S2和训练后的两动作控制器。角点亏损总和由S2的114.203 mm和两动作方法的42.107 mm降至J-NNC的13.902 mm。全程累计曲线在接近上限巡航时保持水平，因此能够看清后半圈出现的额外降速。

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{Figure_8.pdf}
\caption{1 ms下的全部角点速度损失与全程巡航：（a）全部$\pm10$ mm角点窗口内的积分速度亏损，阴影标出代表角点；（b）低于99 mm/s的累计时间。图中表格给出不低于99 mm/s的总时间、占完整时间的比例及最长连续区间。保留全部34个角点和终端停稳过程。}
\label{fig:all_corners}
\end{figure}

J-NNC在不低于99 mm/s下累计运行442.552 s，S2和两动作控制器分别为441.238 s和440.299 s；最长连续区间分别为111.753 s、71.569 s和71.524 s。角点速度损失的减少因而体现为更长的持续巡航和更短的完整时间。

为区分路径缩短和速度变化的贡献，令$L_{\mathrm{exec}}=h\sum_k\|v_k\|_2$、$D_{\mathrm{all}}=h\sum_k(100-\|v_k\|_2)$。忽略浮点舍入误差时，
\begin{equation}
T=\frac{L_{\mathrm{exec}}+D_{\mathrm{all}}}{100}.
\label{eq:time_decomposition}
\end{equation}
静止尾段计入$D_{\mathrm{all}}$。J-NNC执行距离为44390.406 mm，全程速度亏损为27.294 mm；S2分别为44392.151 mm和126.349 mm。节省的1.008 s中，约0.991 s来自速度损失减少，0.017 s来自路径缩短。相较两动作模型，0.537 s的节省包括速度损失减少贡献的0.303 s和路径缩短贡献的0.234 s。密集案例进一步考察连续过弯占比提高时的效率收益。

\subsection{消融结果}
在相同物理约束下，最终三动作控制器以14.839 s完成密集案例，选定两动作模型为15.875 s，降幅为6.53\%。即使与全部已评价两动作模型中的最快有效结果15.798 s相比，J-NNC仍节省0.959 s，即6.07\%。

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{Figure_9.pdf}
\caption{1 ms下的两动作比较：（a）密集闭合参考路径及局部比较位置；（b）完整执行的累计速度亏损，包含全部84角及停稳；（c、d）实际执行的$90^\circ$和$45^\circ$过渡。曲线对应\tabref{tab:main_results}中的最终控制器。}
\label{fig:ablation_dense}
\end{figure}

\figref{fig:ablation_dense}（c、d）显示两种控制器对允差带的不同利用。J-NNC在密集路径上的接近上限巡航时间为4.368 s，选定两动作模型为0.114 s；全程速度亏损由213.867 mm降至132.502 mm。节省的1.036 s中，0.814 s来自速度损失减少，0.222 s来自路径缩短，表明完整效率的提高同时涉及局部几何形状和全程进给利用。另设的同起点、等预算PPO比较未得到超过数值不确定性的时间收益（\ref{app:learning}）；本节较大的降幅对应最终控制器之间的比较。

三个独立初始化的学生模型在两个案例上均满足物理约束、完整时间及全部34角要求（\tabref{tab:selected_students}），密集案例相较最快有效两动作结果的降幅为5.94\%至6.07\%。三者共享教师数据和特征设计，后续精化方式不同；这些结果表示三个选定模型均成功，不能等同于同一固定训练流程的重复试验。全部主图使用种子923202的模型。

\begin{table}[htbp]
\centering
\caption{1 ms下三个选定学生模型的完整时间。各模型均满足物理约束、完整执行和全部角点要求。密集案例降幅相对于最快有效两动作时间15.798 s计算。}
\label{tab:selected_students}
\small
\begin{tabular}{lrrr}
\toprule
学生模型种子 & 长路径（s） & 密集案例（s） & 密集案例降幅（\%）\\
\midrule
923201 & 444.178 & 14.859 & 5.94\\
923202（主模型） & 444.177 & 14.839 & 6.07\\
923203 & 444.178 & 14.839 & 6.07\\
\bottomrule
\end{tabular}
\end{table}

上述比较分别对应联合运动规划的两项要求。NNC直接执行存在运动学越限时，J-NNC能够给出可行的平面运动指令；在可行方法之间，提前进行轨迹整形并协调进给恢复，能够减少累计速度损失。密集案例体现连续过渡中的收益，原长路径则验证完整一圈中的持续巡航及全部角点表现。

'''

APP_EN=r'''
\section{Execution-Resolution Check}
\label{app:resolution}
\setcounter{table}{0}
The 1 ms results in the main text were repeated at $h=0.5$ ms with unchanged policy parameters, physical limits and 20 ms request period. Table~\ref{tab:resolution} gives the supplementary complete times; all listed constrained controllers remain feasible. The highest J-NNC command-interval distance is 1.249077 mm on the dense case, below 1.25 mm.

\begin{table}[htbp]
\centering
\caption{Supplementary complete times at 0.5 ms. The primary 1 ms comparison is in \tabref{tab:main_results}.}
\label{tab:resolution}
\small
\begin{tabular}{lrrr}
\toprule
Case & S2 (s) & Two-action (s) & J-NNC (s)\\
\midrule
Long route & 445.1850 & 444.7310 & 444.1950\\
Dense case & 19.9565 & 15.8955 & 14.8785\\
\bottomrule
\end{tabular}
\end{table}

Let $\delta T$ denote the absolute complete-time change from 1 to 0.5 ms. A gain is accepted only when it exceeds $\max\{0.100,5(\delta T_{\mathrm{J\text{-}NNC}}+\delta T_{\mathrm{baseline}})\}$ seconds at both intervals. Long-route gains are 1.008/0.990 s against S2 and 0.537/0.536 s against the selected two-action model, exceeding margins of 0.100 and 0.175 s. Dense-case gains against the fastest valid two-action times (15.798/15.8755 s) are 0.959/0.997 s, exceeding a 0.585 s margin. This is a numerical sensitivity check, not a statistical confidence interval or an independent training repeat.

All 34 corner comparisons also hold at both intervals with wider $\pm20$ and $\pm30$ mm windows. At 0.5 ms, student seeds 923201, 923202 and 923203 take 444.1955, 444.1950 and 444.1955 s on the long route, and 14.8980, 14.8785 and 14.8775 s on the dense case; all remain feasible and meet the all-corner criterion. Their dense gains over the fastest valid two-action result range from 6.16\% to 6.29\%.

'''

APP_ZH=r'''
\section{执行分辨率检查}
\label{app:resolution}
\setcounter{table}{0}
保持策略参数、物理上限和20 ms动作周期不变，将正文的1 ms执行间隔减小至$h=0.5$ ms复核。\tabref{tab:resolution}列出补充完整时间，表中各受约束控制器仍满足可行性要求。J-NNC的最大运动指令区间距离为密集案例中的1.249077 mm，低于1.25 mm上限。

\begin{table}[htbp]
\centering
\caption{0.5 ms下的补充完整时间。主要1 ms比较见\tabref{tab:main_results}。}
\label{tab:resolution}
\small
\begin{tabular}{lrrr}
\toprule
案例 & S2（s） & 两动作（s） & J-NNC（s）\\
\midrule
长路径 & 445.1850 & 444.7310 & 444.1950\\
密集案例 & 19.9565 & 15.8955 & 14.8785\\
\bottomrule
\end{tabular}
\end{table}

令$\delta T$为1 ms与0.5 ms完整时间之差的绝对值。仅当时间收益在两个间隔下均超过$\max\{0.100,5(\delta T_{\mathrm{J\text{-}NNC}}+\delta T_{\mathrm{baseline}})\}$秒时，才认为其通过数值检查。长路径相较S2的收益为1.008/0.990 s，相较选定两动作模型为0.537/0.536 s，分别超过0.100 s和0.175 s裕量。密集案例相较最快有效两动作时间（15.798/15.8755 s）的收益为0.959/0.997 s，超过0.585 s裕量。这是数值敏感性检查，不是统计置信区间，也不代表独立训练重复。

将角点窗口扩大至$\pm20$和$\pm30$ mm后，两个执行间隔下均保持全部34角比较结论。在0.5 ms下，种子923201、923202和923203的长路径完整时间分别为444.1955 s、444.1950 s和444.1955 s，密集案例分别为14.8980 s、14.8785 s和14.8775 s，均保持可行并满足全部角点要求；密集案例相较最快有效两动作结果的降幅为6.16\%至6.29\%。

'''

def main():
    for lang,experiment,app in [('en',EN,APP_EN),('zh',ZH,APP_ZH)]:
        marker='\\begin{figure}[!htbp]\n\\centering\n\\includegraphics[width=0.98\\linewidth,height=0.75\\textheight,keepaspectratio]{Figure_6.pdf}'
        assert experiment.count(marker)==1
        experiment=experiment.replace(marker,utilization_table(lang)+marker)
        source=OLD/f'latex_source_{lang}'
        dest=OUT/f'latex_source_{lang}'
        dest.mkdir(exist_ok=True); (dest/'figures_lowdim_revision').mkdir(exist_ok=True)
        name='manuscript.tex' if lang=='en' else 'manuscript_zh.tex'
        text=(source/name).read_text(encoding='utf-8')
        pat=r'\\section\{'+('Experiments and Results' if lang=='en' else '实验与结果')+r'\}.*?(?=\\section\{'+('Conclusion' if lang=='en' else '结论')+r'\})'
        text,count=re.subn(pat,lambda m:experiment,text,flags=re.S);assert count==1
        if lang=='en':
            text=text.replace('Both advantages persist at a 0.5 ms execution interval. ','')
            text=text.replace('Local trajectories and synchronized action--motion records associate these gains','Local trajectories and normalized motion records associate these gains')
            text=text.replace(', with paired numerical-resolution checks.', '.')
            text=text.replace('These benefits persist at 0.5 ms and exceed the numerical acceptance margins. Local trajectory and action--motion records connect the efficiency improvement','Local trajectories and V/A/J utilization connect the efficiency improvement')
        else:
            text=text.replace('将物理积分间隔减小至0.5 ms后，两项优势均保持。','')
            text=text.replace('局部轨迹及同步记录的动作与运动数据表明','局部轨迹及归一化运动记录表明')
            text=text.replace('，并通过成对积分间隔检查。','。')
            text=text.replace('在0.5 ms下，这些收益继续保持且超过数值验收裕量。局部轨迹以及动作与运动记录表明','局部轨迹以及V/A/J利用率表明')
        pos=text.index('\\bibliographystyle')
        text=text[:pos]+app+text[pos:]
        assert text.count('\\begin{figure}')==9
        assert text.split('\n\\appendix\n')[0].count('0.5 ms')==1
        assert 'fig:mechanism' not in text and 'fig:efficiency' not in text
        original_cites=set(re.findall(r'\\cite\{([^}]+)',(source/name).read_text(encoding='utf-8')))
        assert original_cites==set(re.findall(r'\\cite\{([^}]+)',text))
        (dest/name).write_text(text,encoding='utf-8')
        for p in source.iterdir():
            if p.suffix in ['.bib','.bbl','.cls','.bst']:shutil.copy2(p,dest/p.name)
        for i in range(1,10):shutil.copy2(FIG/f'Figure_{i}.pdf',dest/'figures_lowdim_revision'/f'Figure_{i}.pdf')
    write(OUT/'editorial_scope.json',dict(old_delivery=str(OLD/'delivery_manifest.json'),
        old_en_sha256=sha(OLD/'latex_source_en/manuscript.tex'),old_zh_sha256=sha(OLD/'latex_source_zh/manuscript_zh.tex'),
        new_en_sha256=sha(OUT/'latex_source_en/manuscript.tex'),new_zh_sha256=sha(OUT/'latex_source_zh/manuscript_zh.tex'),
        methodology_unchanged=True,changes_outside_experiments='Abstract/contribution/conclusion: remove repeated0.5ms emphasis and align wording with displayed trajectory/utilization figures. Add compact numerical-check appendix.',
        main_interpolation_interval_ms=1,half_ms_mentions_in_main_text=1,main_figures=9,new_training_or_simulation=False))
    print('EN_ZH_SOURCES_READY',OUT)

if __name__=='__main__':main()
