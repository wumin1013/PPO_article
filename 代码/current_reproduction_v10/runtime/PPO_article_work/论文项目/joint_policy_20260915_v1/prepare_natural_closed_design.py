"""Prepare a new natural contour using the existing static geometry audit code."""
from pathlib import Path
import json
from design_composite_long_case import sha

BASE=Path(__file__).resolve().parent
source=BASE/"design_closed_composite_case.py"
target=BASE/"design_natural_closed_case.py"
if target.exists(): raise RuntimeError("Prepared design exists; inspect instead of overwriting.")
receipt=json.loads((BASE/"paper_evidence_review/composite_closed_case/receipt.json").read_text(encoding="utf-8"))
assert sha(source)==receipt["source_sha256"][source.name]
text=source.read_text(encoding="utf-8")
text=text.replace('OUT = BASE / "paper_evidence_review/composite_closed_case"','OUT = BASE / "paper_evidence_review/natural_closed_case"')
text=text.replace('current = np.array([1500., 6000.])','current = np.array([1800., 7800.])')
text=text.replace('previous_serpentine_design','previous_designs')
text=text.replace('retained_as_rejected_draft_not_current_case','retained_as_superseded_drafts_not_current_case')
start=text.index('    # Upper edge:')
end=text.index('    ref = np.array(reference)',start)
body='''    # Macro-scale recesses and projections replace tiny rectangular edge jogs.
    # Long and short lines alternate, but their directions and lengths vary.
    acute_x = 6750 + 1800/math.tan(math.radians(67.5))
    for p,label in [([3100,7800],"upper_cruise"),([3500,7400],"upper_obtuse_descent"),
                    ([5100,7400],"upper_recess_floor"),([5100,7900],"upper_right_angle_rise"),
                    ([6400,9200],"upper_large_projection"),([6750,9000],"upper_short_shoulder"),
                    ([acute_x,7200],"deep_acute_recess_in"),
                    ([acute_x+500/math.tan(math.radians(67.5)),7700],"deep_acute_recess_out"),
                    ([9200,8200],"upper_oblique_long"),([9700,7900],"upper_short_return"),
                    ([11200,7900],"upper_cruise_2"),([11200,7300],"upper_short_drop"),
                    ([12500,6000],"right_oblique_approach")]:
        add_line(p,label)
    radius=600.
    center=current+radius*np.array([-1.,-1.])/math.sqrt(2)
    add_arc(center,radius,-90,"right_convex_arc")
    y0=float(current[1])
    for p,label in [([11400,y0-1100],"right_long_inward"),([11400,y0-1500],"right_short_vertical"),
                    ([12700,y0-2800],"right_large_projection"),([12500,y0-3300],"right_short_tip"),
                    ([10700,y0-4100],"lower_oblique_cruise"),([10400,y0-3600],"lower_short_rise"),
                    ([8900,y0-2900],"lower_large_recess_in"),([8600,y0-3200],"lower_short_shoulder"),
                    ([7300,y0-4500],"lower_large_recess_out"),([7300,y0-5000],"lower_short_drop"),
                    ([5700,y0-5000],"lower_long_floor"),([5450,y0-4600],"lower_short_rise_2"),
                    ([4200,y0-2900],"lower_oblique_rise"),([4120,y0-2900],"lower_80mm_plateau"),
                    ([2850,y0-4300],"lower_oblique_fall"),([2300,y0-4500],"lower_short_shoulder_2"),
                    ([1000,y0-3400],"left_oblique_approach")]:
        add_line(p,label)
    tangent=np.array([-1300.,1100.]); tangent/=np.linalg.norm(tangent)
    center=current+800*np.array([tangent[1],-tangent[0]])
    add_arc(center,800,-90,"left_convex_arc")
    heading=primitives[-1]["end_heading_rad"]
    add_line(current+1100*np.array([math.cos(heading),math.sin(heading)]),"left_long_departure")
    add_line(current+[400.,0.],"left_short_inward")
    for p,label in [([700,5300],"left_large_recess"),([900,5700],"left_short_projection"),
                    ([0,7000],"left_long_shoulder"),([350,7400],"left_short_shoulder"),
                    ([1000,7800],"upper_left_return"),([1800,7800],"return_to_start")]:
        add_line(p,label)

'''
text=text[:start]+body+text[end:]
text=text.replace('[("obtuse",[3800,5200]),("right_angle",[7100,5400]),("acute",[9200+acute_run,6200])]',
                  '[("obtuse",[3500,7400]),("right_angle",[5100,7400]),("acute",[acute_x,7200])]')
text=text.replace('case_id="composite_closed_case"','case_id="natural_closed_case"')
text=text.replace('certificate_milestones=proof,sharp_corner_counts=counts,arc_count=4,',
                  'certificate_milestones=proof,sharp_corner_counts=counts,arc_count=2,')
anchor='    write(OUT/"geometry_audit.json",audit)'
extra='''    line_lengths=[p["length_mm"] for p in primitives if p["kind"]=="line"]
    classes=["short" if x<=700 else "long" for x in line_lengths]
    angle_values=[abs(e["signed_heading_change_deg"]) for e in events]
    audit.update(straight_count=len(line_lengths),short_straight_count=classes.count("short"),
                 long_straight_count=classes.count("long"),short_definition_mm="<=700",long_definition_mm=">700",
                 straight_length_mm=sum(line_lengths),straight_length_fraction=sum(line_lengths)/s,
                 short_long_neighbor_switches=sum(a!=b for a,b in zip(classes[:-1],classes[1:])),
                 min_heading_change_deg=min(angle_values),max_heading_change_deg=max(angle_values),
                 min_interior_angle_deg=min(e["interior_angle_deg"] for e in events),
                 max_interior_angle_deg=max(e["interior_angle_deg"] for e in events),
                 reference_geometry_depends_on_control_period=False,
                 actual_control_periods="not selected; 20 ms / 1 ms are historical defaults, not hard constraints")
    assert audit["short_straight_count"]>=15 and audit["long_straight_count"]>=18
    assert audit["straight_length_fraction"]>.90
'''
text=text.replace(anchor,extra+anchor)
text=text.replace('xytext=(1000,6700)','xytext=(600,8500)')
text=text.replace('(2600,6300),(1800,6300)','(3000,8150),(2150,8150)')
text=text.replace('ax.text(2700,6330','ax.text(3150,8150')
start=text.index('    ax.text(11750,2950,')
end=text.index('    colors=',start)
text=text[:start]+'''    ax.text(12450,5350,"R = 600 mm",color="#0072B2",fontsize=8)
    ax.text(900,2480,"R = 800 mm",color="#0072B2",fontsize=8)
    ax.text(6300,4300,f"{audit['long_straight_count']} long + {audit['short_straight_count']} short lines; two arcs\\n"
                       f"Heading changes {min(angle_values):.1f}°–{max(angle_values):.1f}°",ha="center",fontsize=10)
'''+text[end:]
text=text.replace('positions=[(3300,4600),(6500,4650),(9500,6800)]',
                  'positions=[(3000,6550),(4950,6550),(7950,6550)]')
text=text.replace('xlim=(-1300,14600),ylim=(-650,7400)','xlim=(-850,13900),ylim=(-650,9800)')
text=text.replace('fig=plt.figure(figsize=(11.5,8.0)','fig=plt.figure(figsize=(11.5,9.0)')
text=text.replace('height_ratios=[1.7,1]','height_ratios=[2.1,1]')
text=text.replace('a  One irregular closed reference: long/short lines, three corner classes, and circular arcs',
                  'a  One asymmetric closed contour: deep recesses, prominent corners, long/short lines, and two arcs')
text=text.replace('Closed composite path — reference design, not a policy result',
                  'Natural composite contour — reference design, not a policy result')
start=text.index('    (OUT/"路径设计说明.md").write_text(')
end=text.index('    write(OUT/"receipt.json"',start)
report='''    (OUT/"路径设计说明.md").write_text(f"""# 自然复杂的单圈闭合长路径案例

主体采用不对称深凹槽、显著凸角和斜向长边；不用矩形轮廓上的微小台阶凑复杂度。长短直线在不同方向频繁交替，仅保留两段圆弧。此图是参考几何，不是主方法运行结果。

| 项目 | 设计及静态审核结果 |
|---|---|
| 路径 | 一条无自交闭合轮廓，完整一圈 |
| 解析长度 | {s/1000:.6f} m |
| 考虑允差切角后的保守时间下界 | **{lower_time:.6f} s > 300 s**，仅用速度上限100 mm/s与必须顺序覆盖的门截面；未计加减速与停稳 |
| 直线数量 | {audit['straight_count']}段：长直线{audit['long_straight_count']}段、短直线{audit['short_straight_count']}段 |
| 长短定义与范围 | 本案例短段≤700 mm、长段>700 mm；实际最短{min(line_lengths):.3f} mm、最长{max(line_lengths):.3f} mm |
| 长短交替 | {audit['short_long_neighbor_switches']}处相邻长/短切换，起终点所在直线分作首尾段 |
| 直线长度占比 | {100*audit['straight_length_fraction']:.3f}% |
| 航向转角范围 | {min(angle_values):.3f}°–{max(angle_values):.3f}°；以真实方向变化解释转弯强度 |
| 内角范围 | {audit['min_interior_angle_deg']:.3f}°–{audit['max_interior_angle_deg']:.3f}° |
| 三类角 | {counts['obtuse']}个钝角、{counts['right_angle']}个直角、{counts['acute']}个锐角，全部纳入角点统计 |
| 圆弧 | 两段90°弧，半径600、800 mm，切向连接相邻直线 |
| 起终点 | (1800,7800) mm，几何切向一致；实际仍静止起动、最后完整制动停稳 |
| 静态审核 | 无自交、无门冲突、无180°歧义；非相邻原语最小保守间距{clearance:.3f} mm |

显著构形包括：上侧约1800 mm深的锐角凹槽、约1300 mm高的凸起，下侧多个不等深的大凹槽与80 mm短平台；不再把全部复杂度限制在全图几乎看不到的几十毫米小台阶。

机制图按同一条完整轨迹取三列：B钝角135°(3500,7400)，C直角90°(5100,7400)，D锐角45°({acute_x:.6f},7200)。它们对应航向转角+45°、+90°、+135°。全图表示整体形状，40×40 mm局部窗表示1.25 mm允差与实际过渡；统计窗沿参考弧长每侧最多10 mm、按相邻段裁剪。

**控制周期可设计。** 20 ms策略决策和1 ms执行是现有实现的历史设置，不是本案例不可修改的限制。新方法可以采用其他固定周期、多速率或事件触发方式，但应公开真实时间戳、动作保持/插值、执行与审核分辨率；周期变化不能漏掉几何、V/A/J违规。停稳时长仍至少20 ms，不能误改为20个采样点。

共同参考reference.csv；解析线/弧、全部角点、统计窗及哈希在geometry.json。原语、截面和完整时间下界在geometry_audit.json。弦高误差不能加到1.25 mm物理允差上。该下界与策略更新周期无关，只依赖共同几何、有序覆盖和100 mm/s速度上限。

此前往返、四圆弧和矩形微台阶方案保留为已被用户修正的设计。本阶段无模型评价、PPO更新、拟合、参数搜索或物理仿真；不存在已经验证的主方法收益。
""",encoding="utf-8")
'''
text=text[:start]+report+text[end:]
text=text.replace('"sharp_corner_counts","reference_points","minimum_nonadjacent_primitive_clearance_lower_bound_mm"]',
                  '"sharp_corner_counts","reference_points","straight_count","long_straight_count","short_straight_count","straight_length_fraction","short_long_neighbor_switches","min_heading_change_deg","max_heading_change_deg"]')
target.write_text(text,encoding="utf-8")
print(target.name)
