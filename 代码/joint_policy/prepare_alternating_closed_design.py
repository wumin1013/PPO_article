"""Reuse the audited static builder with the user's revised line-dominant geometry."""
from pathlib import Path
from design_composite_long_case import sha
import json

BASE = Path(__file__).resolve().parent
source = BASE / "design_closed_composite_case.py"
target = BASE / "design_closed_alternating_case.py"
if target.exists():
    raise RuntimeError("Prepared builder already exists; inspect instead of overwriting.")
registered = json.loads((BASE / "paper_evidence_review/composite_closed_case/receipt.json").read_text(encoding="utf-8"))
assert sha(source) == registered["source_sha256"][source.name]
text = source.read_text(encoding="utf-8")
text = text.replace('OUT = BASE / "paper_evidence_review/composite_closed_case"',
                    'OUT = BASE / "paper_evidence_review/composite_closed_alternating_case"')
text = text.replace('current = np.array([1500., 6000.])','current = np.array([1000., 6000.])')
start = text.index('    # Upper edge:')
end = text.index('    ref = np.array(reference)', start)
geometry = '''    # The perimeter alternates long runs and short line changes. Only two
    # quarter-circle connectors remain; all listed coordinates precede rollout.
    acute_run = 100/math.tan(math.radians(67.5))
    for p,label in [([2400,6000],"upper_long_1"),([2480,5920],"upper_short_diagonal_1"),
                    ([3700,5920],"upper_long_2"),([3700,6000],"upper_short_upright_1"),
                    ([5000,6000],"upper_long_3"),([5000+acute_run,6100],"upper_acute_short_rise"),
                    ([5000+2*acute_run,6000],"upper_acute_short_fall"),([6500,6000],"upper_long_4"),
                    ([6580,5920],"upper_short_diagonal_2"),([7900,5920],"upper_long_5"),
                    ([7900,6000],"upper_short_upright_2"),([9500,6000],"upper_long_6"),
                    ([9580,5920],"upper_short_diagonal_3"),([11000,5920],"upper_long_7")]:
        add_line(p,label)
    add_arc([11000,5320],600,-90,"upper_right_quarter_circle")
    for p,label in [([11600,4100],"right_long_1"),([11560,4100],"right_short_cross_1"),
                    ([11560,2600],"right_long_2"),([11600,2560],"right_short_diagonal"),
                    ([11600,1000],"right_long_3"),([11560,1000],"right_short_cross_2"),
                    ([11560,0],"right_long_4"),([9800,0],"lower_long_1"),
                    ([9800,40],"lower_short_upright_1"),([8600,40],"lower_long_2"),
                    ([8600,80],"lower_short_upright_2"),([7300,1380],"lower_long_inward_rise"),
                    ([7280,1380],"lower_short_20mm_platform"),([5980,80],"lower_long_inward_fall"),
                    ([5980,40],"lower_short_upright_3"),([4600,40],"lower_long_3"),
                    ([4560,0],"lower_short_diagonal"),([3000,0],"lower_long_4"),
                    ([3000-acute_run,100],"lower_acute_short_rise"),
                    ([3000-2*acute_run,0],"lower_acute_short_fall"),([900,0],"lower_long_5")]:
        add_line(p,label)
    add_arc([900,700],700,-90,"lower_left_quarter_circle")
    for p,label in [([200,2000],"left_long_1"),([240,2000],"left_short_cross_1"),
                    ([240,3500],"left_long_2"),([200,3540],"left_short_diagonal"),
                    ([200,4900],"left_long_3"),([260,4900],"left_short_cross_2"),
                    ([260,6000],"left_long_4"),([1000,6000],"return_to_start")]:
        add_line(p,label)

'''
text = text[:start] + geometry + text[end:]
text = text.replace('[("obtuse",[3800,5200]),("right_angle",[7100,5400]),("acute",[9200+acute_run,6200])]',
                    '[("obtuse",[2480,5920]),("right_angle",[3700,5920]),("acute",[5000+acute_run,6100])]')
text = text.replace('case_id="composite_closed_case"','case_id="composite_closed_alternating_case"')
text = text.replace('certificate_milestones=proof,sharp_corner_counts=counts,arc_count=4,',
                    'certificate_milestones=proof,sharp_corner_counts=counts,arc_count=2,')
anchor = '    write(OUT/"geometry_audit.json",audit)'
extra = '''    lengths=[p["length_mm"] for p in primitives if p["kind"]=="line"]
    classes=["short" if x<=200 else "long" if x>=700 else "middle" for x in lengths]
    audit.update(straight_count=len(lengths),short_straight_count=sum(x=="short" for x in classes),
                 long_straight_count=sum(x=="long" for x in classes),middle_straight_count=sum(x=="middle" for x in classes),
                 short_definition_mm="<=200",long_definition_mm=">=700",
                 straight_length_mm=sum(lengths),straight_length_fraction=sum(lengths)/s,
                 short_long_neighbor_switches=sum(a!=b and {a,b}=={"short","long"} for a,b in zip(classes[:-1],classes[1:])))
    assert audit["short_straight_count"]>=15 and audit["long_straight_count"]>=20
    assert audit["straight_length_fraction"]>.90
'''
text = text.replace(anchor,extra+anchor)
text = text.replace('xytext=(1000,6700)','xytext=(300,6650)')
text = text.replace('(2600,6300),(1800,6300)','(2300,6300),(1600,6300)')
text = text.replace('ax.text(2700,6330','ax.text(2400,6330')
start = text.index('    ax.text(11750,2950,')
end = text.index('    colors=',start)
text = text[:start]+'''    ax.text(10600,5050,"R = 600 mm",color="#0072B2",fontsize=9)
    ax.text(950,650,"R = 700 mm",color="#0072B2",fontsize=9)
    ax.text(5400,2850,f"{audit['long_straight_count']} long + {audit['short_straight_count']} short straight segments\\n"
                        f"{100*audit['straight_length_fraction']:.1f}% of length is straight",ha="center",fontsize=10)
'''+text[end:]
text = text.replace('positions=[(3300,4600),(6500,4650),(9500,6800)]',
                    'positions=[(2000,5200),(3600,5200),(5300,6700)]')
text = text.replace('xlim=(-1300,14600),ylim=(-650,7400)','xlim=(-350,12100),ylim=(-450,7100)')
text = text.replace('a  One irregular closed reference: long/short lines, three corner classes, and circular arcs',
                    'a  One closed contour with frequent long/short straight alternation and two circular arcs')
text = text.replace('Closed composite path — reference design, not a policy result',
                    'Line-dominant closed path — reference design, not a policy result')
start=text.index('    (OUT/"路径设计说明.md").write_text(')
end=text.index('    write(OUT/"receipt.json"',start)
report='''    (OUT/"路径设计说明.md").write_text(f"""# 长短直线频繁交替的单圈闭合长路径

依据用户最新修正，将原圆弧较多的轮廓改为直线主导：保持单圈凹凸轮廓，圆弧仅保留右上与左下两个连接，不采用往返扫描。尚未执行策略。

| 项目 | 结果 |
|---|---|
| 解析总长 | {s/1000:.6f} m |
| 允许切角后的完整时间保守下界 | **{lower_time:.6f} s > 300 s**，速度上限100 mm/s；未额外计加减速与停稳 |
| 直线 | {audit['straight_count']}段，其中{audit['long_straight_count']}段长直线(≥700 mm)、{audit['short_straight_count']}段短直线(≤200 mm)，中间长度{audit['middle_straight_count']}段 |
| 长短交替 | 直线序列中{audit['short_long_neighbor_switches']}处相邻长/短切换；角点和短段不因总长度放大 |
| 长度范围 | 最短{min(lengths):.3f} mm，最长{max(lengths):.3f} mm；完整逐段表见geometry.json |
| 直线长度占比 | **{100*audit['straight_length_fraction']:.3f}%** |
| 圆弧 | 仅2段90°圆弧，半径600、700 mm，合计{(s-sum(lengths))/1000:.6f} m |
| 三类角 | {counts['obtuse']}个钝角、{counts['right_angle']}个直角、{counts['acute']}个锐角，全部进入同案例统计 |
| 起终点 | (1000,6000) mm，位于同一直线内部；静止起动，完整一圈后制动停稳 |
| 静态审核 | 无自交、无相邻门冲突、无180°方向歧义；非相邻段最小保守间距{clearance:.6f} mm |

主机制图的三个局部窗在运行前固定：B钝角内角135°(2480,5920)，C直角90°(3700,5920)，D锐角45°({5000+acute_run:.6f},6100)。三个角均取同一完整执行记录，绘制三动作、实际航向、曲率和速度。预览局部窗为40×40 mm，指标窗口沿参考弧长每侧最多10 mm并按相邻段裁剪。

长直线提供持续巡航区间；20–114 mm短直线和连续折角提供协调难点。完整统计同时报告各角降速、短段恢复、高速路程占比和全程时间，不能只用大量长直线制造好看的巡航总量。

共同几何文件reference.csv；解析线/弧、全体角点、局部窗与哈希在geometry.json。geometry_audit.json保存截面包围圆距离下界、检查值和分类统计。弧离散误差不加到1.25 mm允差上，后续实际运动还需解析几何、V/A/J、覆盖、制动及停稳审核。

此前四圆弧闭合方案与四行往返草案均保留为已被用户修正的设计，不继续作为主案例。当前只完成设计，没有模型评价、PPO更新或物理执行，也未开启第三轮算法优化。
""",encoding="utf-8")
'''
text=text[:start]+report+text[end:]
text=text.replace('"sharp_corner_counts","reference_points","minimum_nonadjacent_primitive_clearance_lower_bound_mm"]',
                  '"sharp_corner_counts","reference_points","straight_count","long_straight_count","short_straight_count","straight_length_fraction","short_long_neighbor_switches"]')
target.write_text(text,encoding="utf-8")
print(target.name)
