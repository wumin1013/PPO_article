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
OUT = BASE / "paper_evidence_review/composite_closed_alternating_case"
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
        "previous_serpentine_design": "retained_as_rejected_draft_not_current_case",
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

    current = np.array([1000., 6000.])
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

    # The perimeter alternates long runs and short line changes. Only two
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
    for category,point in [("obtuse",[2480,5920]),("right_angle",[3700,5920]),("acute",[5000+acute_run,6100])]:
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
    write(OUT/"geometry.json",dict(case_id="composite_closed_alternating_case",closed=True,lap_count=1,
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
    lengths=[p["length_mm"] for p in primitives if p["kind"]=="line"]
    classes=["short" if x<=200 else "long" if x>=700 else "middle" for x in lengths]
    audit.update(straight_count=len(lengths),short_straight_count=sum(x=="short" for x in classes),
                 long_straight_count=sum(x=="long" for x in classes),middle_straight_count=sum(x=="middle" for x in classes),
                 short_definition_mm="<=200",long_definition_mm=">=700",
                 straight_length_mm=sum(lengths),straight_length_fraction=sum(lengths)/s,
                 short_long_neighbor_switches=sum(a!=b and {a,b}=={"short","long"} for a,b in zip(classes[:-1],classes[1:])))
    assert audit["short_straight_count"]>=15 and audit["long_straight_count"]>=20
    assert audit["straight_length_fraction"]>.90
    write(OUT/"geometry_audit.json",audit)

    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":9,"svg.fonttype":"none","pdf.fonttype":42,"axes.spines.top":False,
        "axes.spines.right":False,"path.simplify":False})
    fig=plt.figure(figsize=(11.5,8.0),layout="constrained")
    grid=fig.add_gridspec(2,3,height_ratios=[1.7,1])
    ax=fig.add_subplot(grid[0,:])
    ax.plot(ref[:,0],ref[:,1],color="#444B52",lw=1.6)
    for p,shape in zip(primitives,shapes):
        if p["kind"]=="arc":
            xy=np.array(shape.coords); ax.plot(xy[:,0],xy[:,1],color="#0072B2",lw=1.7)
    ax.scatter(*ref[0],marker="s",s=30,color="black",zorder=5)
    ax.annotate("Start / end: one lap",ref[0],xytext=(300,6650),arrowprops=dict(arrowstyle="-",lw=.8))
    ax.annotate("",(2300,6300),(1600,6300),arrowprops=dict(arrowstyle="->",color="#444B52",lw=1))
    ax.text(2400,6330,"Travel direction",fontsize=8)
    ax.text(10600,5050,"R = 600 mm",color="#0072B2",fontsize=9)
    ax.text(950,650,"R = 700 mm",color="#0072B2",fontsize=9)
    ax.text(5400,2850,f"{audit['long_straight_count']} long + {audit['short_straight_count']} short straight segments\n"
                        f"{100*audit['straight_length_fraction']:.1f}% of length is straight",ha="center",fontsize=10)
    colors=["#A76920","#0072B2","#785A99"]
    positions=[(2000,5200),(3600,5200),(5300,6700)]
    for event,letter,color,position in zip(chosen,"BCD",colors,positions):
        point=np.array(event["point_mm"])
        ax.scatter(*point,s=22,color=color,zorder=5)
        ax.annotate(letter,point,xytext=position,color=color,fontweight="bold",arrowprops=dict(arrowstyle="-",lw=.8,color=color))
    ax.set(aspect="equal",xlabel="X (mm)",ylabel="Y (mm)",xlim=(-350,12100),ylim=(-450,7100))
    ax.set_title("a  One closed contour with frequent long/short straight alternation and two circular arcs",loc="left",fontsize=10)
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
    fig.suptitle(f"Line-dominant closed path — reference design, not a policy result\nLength {s/1000:.3f} m; complete-lap time lower bound {lower_time:.3f} s at 100 mm/s",fontsize=12)
    for ext in ["png","svg","pdf"]:
        fig.savefig(OUT/f"path_preview.{ext}",dpi=200,facecolor="white")
    plt.close(fig)
    (OUT/"路径设计说明.md").write_text(f"""# 长短直线频繁交替的单圈闭合长路径

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
    write(OUT/"receipt.json",dict(created_utc=now,status="completed_static_closed_composite_design",
        source_sha256={Path(__file__).name:sha(Path(__file__))},
        outputs_sha256={str(p.relative_to(BASE)):sha(p) for p in OUT.iterdir() if p.is_file()},
        reference_sha256=sha(OUT/"reference.csv"),time_lower_bound_s=lower_time,
        actor_calls=0,physics_ticks=0,training_updates=0,goal_complete=False,method_frozen=False,
        visual_qa="pending"))
    print(json.dumps({k:audit[k] for k in ["analytic_length_mm","ordered_gate_time_lower_bound_s","sharp_corner_counts","reference_points","straight_count","long_straight_count","short_straight_count","straight_length_fraction","short_long_neighbor_switches"]}))


if __name__=="__main__":
    main()
