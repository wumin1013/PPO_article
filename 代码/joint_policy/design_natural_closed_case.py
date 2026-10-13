"""Static design of one irregular closed contour; no learning or motion rollout."""
import math
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from geometry_spec import Corridor
from design_composite_long_case import sha, write, wrap

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper_evidence_review/natural_closed_case"
EPS, SPEED, CHORD_ERROR, LINE_STEP = 1.25, 100., .002, 25.


def main():
    if OUT.exists():
        raise RuntimeError("Closed-case design already exists; inspect instead of rerunning.")
    OUT.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    write(OUT / "registration.json", {
        "created_utc": now, "purpose": "One irregular closed contour inspired by the user's outline, with distinct geometry, long/short lines, three angle classes, arcs, and a strict >300 s ordered-travel lower bound.",
        "work_type": "static_geometry_design_only", "policy_rollouts": 0,
        "training_updates": 0, "physics_ticks": 0,
        "previous_designs": "retained_as_superseded_drafts_not_current_case",
        "reference_spacing_mm": LINE_STEP, "arc_chord_error_max_mm": CHORD_ERROR,
        "source_sha256": {Path(__file__).name: sha(Path(__file__)),
                          "design_composite_long_case.py": sha(BASE / "design_composite_long_case.py"),
                          "geometry_spec.py": sha(BASE / "geometry_spec.py")},
    })
    (OUT / "figure_contract.md").write_text(
        "# Closed-contour geometry preview\n\n"
        "This is one designed reference, not a simulated tool trajectory. The user's supplied image is the explicit "
        "visual reference: one irregular closed boundary, concave/convex portions, line/arc transitions and local windows. "
        "Use a different perimeter, dimensions and feature arrangement. Do not copy the image's numerical performance annotations. "
        "Python/matplotlib; true XY aspect; full contour above three fixed obtuse/right/acute windows; editable SVG, PDF and PNG. "
        "No external publication precedent or policy-success claim.\n", encoding="utf-8")
    for name in ["PPO_article_研究目标_图表与写作指南.md", "当前执行状态.json"]:
        (OUT / ("guide_before.md" if name.endswith(".md") else "execution_state_before.json")).write_bytes((BASE / name).read_bytes())

    current = np.array([1800., 7800.])
    reference, primitives, shapes, ends = [current.copy()], [], [], []
    s = 0.

    def add_line(point, label):
        nonlocal current, s
        point = np.array(point, float)
        delta = point - current
        length = float(np.linalg.norm(delta))
        h = math.atan2(delta[1], delta[0])
        samples = np.linspace(current, point, int(math.ceil(length / LINE_STEP)) + 1)
        start_index = len(reference) - 1
        reference.extend(samples[1:])
        primitives.append(dict(kind="line", label=label, start_mm=current.tolist(), end_mm=point.tolist(),
                               length_mm=length, start_s_mm=s, end_s_mm=s+length,
                               start_heading_rad=h, end_heading_rad=h,
                               start_reference_index=start_index, end_reference_index=len(reference)-1))
        shapes.append(LineString(samples)); ends.append(len(reference)-1)
        current, s = point, s + length

    def add_arc(center, radius, sweep_degrees, label):
        nonlocal current, s
        center = np.array(center, float)
        assert abs(np.linalg.norm(current-center)-radius) < 1e-8
        start = math.atan2(*(current-center)[::-1])
        sweep = math.radians(sweep_degrees)
        direction = 1 if sweep > 0 else -1
        chunks = int(round(abs(sweep_degrees) / 22.5))
        n_min = math.ceil(abs(sweep)/(2*math.acos(1-CHORD_ERROR/radius)))
        n = int(math.ceil(n_min/chunks)*chunks)
        theta = np.linspace(start, start+sweep, n+1)
        samples = center + radius*np.column_stack([np.cos(theta), np.sin(theta)])
        samples[0] = current
        end = np.round(samples[-1], 10)
        samples[-1] = end
        start_index = len(reference)-1
        reference.extend(samples[1:])
        length = radius*abs(sweep)
        primitives.append(dict(kind="arc", label=label, center_mm=center.tolist(), radius_mm=radius,
                               start_angle_rad=start, sweep_rad=sweep, sample_segments=n,
                               start_mm=current.tolist(), end_mm=end.tolist(),
                               start_heading_rad=wrap(start+direction*math.pi/2),
                               end_heading_rad=wrap(start+sweep+direction*math.pi/2),
                               length_mm=length, start_s_mm=s, end_s_mm=s+length,
                               chord_error_bound_mm=radius*(1-math.cos(abs(sweep)/n/2)),
                               start_reference_index=start_index, end_reference_index=len(reference)-1,
                               certificate_indices=[start_index+j*(n//chunks) for j in range(1,chunks)]))
        shapes.append(LineString(samples)); ends.append(len(reference)-1)
        current, s = end, s+length

    # Macro-scale recesses and projections replace tiny rectangular edge jogs.
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

    ref = np.array(reference)
    ref[-1] = ref[0]
    assert LineString(ref).is_ring and LineString(ref).is_simple
    assert Polygon(ref).is_valid
    corridor = Corridor(ref, EPS, closed=True)
    conflicts = corridor.adjacent_gate_conflicts()
    assert not corridor.ambiguities and not conflicts
    assert corridor.lengths.min() > 1e-9
    events, tangent_errors = [], []
    for i,(a,b) in enumerate(zip(primitives,primitives[1:]+primitives[:1])):
        assert np.linalg.norm(np.array(a["end_mm"])-b["start_mm"]) < 1e-8
        delta = math.degrees(wrap(b["start_heading_rad"]-a["end_heading_rad"]))
        if "arc" in [a["kind"],b["kind"]]:
            tangent_errors.append(abs(delta)); assert abs(delta)<1e-8
        elif abs(delta)>1e-7:
            beta = 180-abs(delta)
            category = "right_angle" if abs(beta-90)<1e-6 else "obtuse" if beta>90 else "acute"
            events.append(dict(id=f"corner_{len(events):02d}", point_mm=a["end_mm"],
                               reference_vertex_index=a["end_reference_index"], analytic_s_mm=a["end_s_mm"],
                               interior_angle_deg=beta, signed_heading_change_deg=delta, category=category,
                               window_before_mm=min(10,a["length_mm"]/2), window_after_mm=min(10,b["length_mm"]/2)))
    seam_heading_error = abs(math.degrees(wrap(primitives[0]["start_heading_rad"]-primitives[-1]["end_heading_rad"])))
    assert seam_heading_error < 1e-8
    chosen=[]
    for category,point in [("obtuse",[3500,7400]),("right_angle",[5100,7400]),("acute",[acute_x,7200])]:
        event=next(e for e in events if e["category"]==category and np.allclose(e["point_mm"],point))
        event["mechanism_column"]=category; chosen.append(event)
    counts={c:sum(e["category"]==c for e in events) for c in ["obtuse","right_angle","acute"]}
    assert all(counts[c]>0 for c in counts)
    clearance=float("inf"); nearest=None
    for i in range(len(primitives)):
        for j in range(i+2,len(primitives)):
            if i==0 and j==len(primitives)-1: continue
            allowance=sum(primitives[k].get("chord_error_bound_mm",0.) for k in [i,j])
            value=shapes[i].distance(shapes[j])-allowance
            if value<clearance: clearance,nearest=value,[i,j]
    assert clearance>2*EPS

    selected=set([0,len(ref)-1]+ends)
    for primitive in primitives:
        selected.update(primitive.get("certificate_indices",[]))
    selected=sorted(selected)
    centers=ref[selected]
    radii=[]; proof=[]
    for i in selected:
        if i in [0,len(ref)-1]:
            bound=EPS
        else:
            cosine=math.sqrt(max(0.,(1+float(corridor.tangents[i-1]@corridor.tangents[i]))/2))
            assert cosine>1e-8
            bound=EPS/cosine
            assert any(g.vertex_index==i for g in corridor.gates)
        radii.append(bound)
        proof.append(dict(reference_vertex_index=i,point_mm=ref[i].tolist(),crossing_radius_bound_mm=bound))
    distances=np.linalg.norm(np.diff(centers,axis=0),axis=1)
    lower_length=float(np.maximum(0,distances-np.array(radii[:-1])-np.array(radii[1:])).sum())
    lower_time=lower_length/SPEED
    assert lower_time>300
    np.savetxt(OUT/"reference.csv",ref,delimiter=",",header="x_mm,y_mm",comments="",fmt="%.12f",encoding="utf-8")
    saved=np.loadtxt(OUT/"reference.csv",delimiter=",",skiprows=1)
    assert np.max(np.abs(saved-ref))<1e-9
    poly_s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(saved,axis=0),axis=1))]
    for event in events: event["reference_s_mm"]=float(poly_s[event["reference_vertex_index"]])
    write(OUT/"geometry.json",dict(case_id="natural_closed_case",closed=True,lap_count=1,
        start_end_mm=ref[0].tolist(),primitives=primitives,sharp_corners=events,
        mechanism_columns=[e["id"] for e in chosen],analytic_length_mm=s,
        reference_file="reference.csv",reference_sha256=sha(OUT/"reference.csv"),
        line_sampling_max_mm=LINE_STEP,arc_chord_error_limit_mm=CHORD_ERROR))
    audit=dict(status="static_closed_geometry_checks_passed",closed=True,one_simple_loop=True,
        analytic_length_mm=s,sampled_reference_length_mm=float(poly_s[-1]),
        reference_constant_100_time_s=s/SPEED,
        ordered_gate_lower_length_mm=lower_length,ordered_gate_time_lower_bound_s=lower_time,
        strict_duration_above_300s=True,
        proof="For each mandatory gate, the intersection with its two finite segment corridors is contained in a disk of radius epsilon/cos(delta/2) about the vertex. Straight gates have radius epsilon. The sum of center distances minus adjacent disk radii lower-bounds a complete ordered lap. Endpoints allow epsilon. Divide by Vmax=100. Acceleration, braking and rest only add time.",
        certificate_milestones=proof,sharp_corner_counts=counts,arc_count=2,
        arc_radii_mm=[p["radius_mm"] for p in primitives if p["kind"]=="arc"],
        max_arc_chord_error_mm=max(p.get("chord_error_bound_mm",0) for p in primitives),
        max_analytic_arc_join_heading_error_deg=max(tangent_errors),seam_heading_error_deg=seam_heading_error,
        minimum_nonadjacent_primitive_clearance_lower_bound_mm=clearance,nearest_primitive_pair=nearest,
        adjacent_gate_conflicts=conflicts,ambiguities=corridor.ambiguities,reference_points=len(ref),
        ordered_gates=len(corridor.gates),minimum_intrinsic_straight_mm=min(p["length_mm"] for p in primitives if p["kind"]=="line"),
        maximum_intrinsic_straight_mm=max(p["length_mm"] for p in primitives if p["kind"]=="line"),
        actor_calls=0,physics_ticks=0,training_updates=0,goal_complete=False,
        motion_not_checked=["actual_path_completion","vector_V_A_J","braking_and_rest","S2_scheduler_compatibility","comparative_efficiency"],
        future_audit_requirement="Shared discretized reference and original analytic line/arc corridor must be distinguished. The chord error is not added to the 1.25 mm tolerance. Check the analytic corridor on actual commands as well.")
    line_lengths=[p["length_mm"] for p in primitives if p["kind"]=="line"]
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
    write(OUT/"geometry_audit.json",audit)

    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9,"svg.fonttype":"none","pdf.fonttype":42,"axes.spines.top":False,
        "axes.spines.right":False,"path.simplify":False})
    fig=plt.figure(figsize=(11.5,9.0),layout="constrained")
    grid=fig.add_gridspec(2,3,height_ratios=[2.1,1])
    ax=fig.add_subplot(grid[0,:])
    ax.plot(ref[:,0],ref[:,1],color="#444B52",lw=1.6)
    for p,shape in zip(primitives,shapes):
        if p["kind"]=="arc":
            xy=np.array(shape.coords); ax.plot(xy[:,0],xy[:,1],color="#0072B2",lw=1.7)
    ax.scatter(*ref[0],marker="s",s=30,color="black",zorder=5)
    ax.annotate("Start / end: one lap",ref[0],xytext=(600,8500),arrowprops=dict(arrowstyle="-",lw=.8))
    ax.annotate("",(3000,8150),(2150,8150),arrowprops=dict(arrowstyle="->",color="#444B52",lw=1))
    ax.text(3150,8150,"Travel direction",fontsize=8)
    ax.text(12450,5350,"R = 600 mm",color="#0072B2",fontsize=8)
    ax.text(900,2480,"R = 800 mm",color="#0072B2",fontsize=8)
    ax.text(6300,4300,f"{audit['long_straight_count']} long + {audit['short_straight_count']} short lines; two arcs\n"
                       f"Heading changes {min(angle_values):.1f}°–{max(angle_values):.1f}°",ha="center",fontsize=10)
    colors=["#A76920","#0072B2","#785A99"]
    positions=[(3000,6550),(4950,6550),(7950,6550)]
    for event,letter,color,position in zip(chosen,"BCD",colors,positions):
        point=np.array(event["point_mm"])
        ax.scatter(*point,s=22,color=color,zorder=5)
        ax.annotate(letter,point,xytext=position,color=color,fontweight="bold",arrowprops=dict(arrowstyle="-",lw=.8,color=color))
    ax.set(aspect="equal",xlabel="X (mm)",ylabel="Y (mm)",xlim=(-850,13900),ylim=(-650,9800))
    ax.set_title("a  One asymmetric closed contour: deep recesses, prominent corners, long/short lines, and two arcs",loc="left",fontsize=10)
    ax.grid(alpha=.16)
    for col,(event,color) in enumerate(zip(chosen,colors)):
        local_ax=fig.add_subplot(grid[1,col]); center=np.array(event["point_mm"])
        # Get a contiguous reference slice around the already fixed vertex.
        i=event["reference_vertex_index"]; local=ref[max(0,i-12):min(len(ref),i+13)]
        region=LineString(local).buffer(EPS,resolution=40)
        bx,by=region.exterior.xy; local_ax.fill(bx,by,color="#E7EFF5")
        local_ax.plot(bx,by,color="#8095A8",ls="--",lw=.7)
        local_ax.plot(local[:,0],local[:,1],color=color,lw=1.2)
        local_ax.scatter(*center,s=16,color=color)
        local_ax.set(aspect="equal",xlim=(center[0]-20,center[0]+20),ylim=(center[1]-20,center[1]+20),xlabel="X (mm)",ylabel="Y (mm)")
        title=["Obtuse","Right angle","Acute"][col]
        local_ax.set_title(f"{'bcd'[col]}  {title}: interior {event['interior_angle_deg']:.0f}°",loc="left",fontsize=10)
        local_ax.text(.03,.04,f"Heading change {event['signed_heading_change_deg']:+.0f}°\n1.25 mm corridor",transform=local_ax.transAxes,fontsize=8,bbox=dict(facecolor="white",edgecolor="none",alpha=.85))
    fig.suptitle(f"Natural composite contour — reference design, not a policy result\nLength {s/1000:.3f} m; complete-lap time lower bound {lower_time:.3f} s at 100 mm/s",fontsize=12)
    for ext in ["png","svg","pdf"]:
        fig.savefig(OUT/f"path_preview.{ext}",dpi=200,facecolor="white")
    plt.close(fig)
    (OUT/"路径设计说明.md").write_text(f"""# 自然复杂的单圈闭合长路径案例

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
    write(OUT/"receipt.json",dict(created_utc=now,status="completed_static_closed_composite_design",
        source_sha256={Path(__file__).name:sha(Path(__file__))},
        outputs_sha256={str(p.relative_to(BASE)):sha(p) for p in OUT.iterdir() if p.is_file()},
        reference_sha256=sha(OUT/"reference.csv"),time_lower_bound_s=lower_time,
        actor_calls=0,physics_ticks=0,training_updates=0,goal_complete=False,method_frozen=False,
        visual_qa="pending"))
    print(json.dumps({k:audit[k] for k in ["analytic_length_mm","ordered_gate_time_lower_bound_s","sharp_corner_counts","reference_points","straight_count","long_straight_count","short_straight_count","straight_length_fraction","short_long_neighbor_switches","min_heading_change_deg","max_heading_change_deg"]}))


if __name__=="__main__":
    main()
