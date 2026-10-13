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
OUT = BASE / "paper_evidence_review/composite_closed_case"
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

    current = np.array([1500., 6000.])
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

    # Upper edge: long runs, a flat-bottomed recess with a 20 mm short line,
    # a right-angle recess and a sharper outward tooth. Angles are analytic.
    acute_run = 200/math.tan(math.radians(67.5))
    for p, label in [([3000,6000],"upper_long_1"),([3800,5200],"obtuse_entry"),
                     ([3820,5200],"short_20mm"),([4620,6000],"obtuse_exit"),
                     ([6500,6000],"upper_long_2"),([7100,5400],"right_entry"),
                     ([7700,6000],"right_exit"),([9200,6000],"upper_long_3"),
                     ([9200+acute_run,6200],"acute_entry"),
                     ([9200+2*acute_run,6000],"acute_exit"),([10800,6000],"upper_long_4")]:
        add_line(p,label)
    add_arc([10800,3000],3000,-180,"right_outer_semicircle")
    # Lower edge has a broad asymmetric inward feature, not a raster row.
    for p,label in [([9300,0],"lower_long_1"),([9300,1400],"lower_upright"),
                    ([8200,2500],"lower_inward_rise"),([5700,0],"lower_inward_fall"),
                    ([3500,0],"lower_long_2")]:
        add_line(p,label)
    add_arc([3500,1300],1300,-90,"lower_left_round")
    add_line([2200,2000],"left_short_upright")
    add_arc([1600,2000],600,90,"left_concave_round")
    add_line([1000,2600],"left_bridge")
    add_arc([1000,4300],1700,-180,"left_outer_semicircle")
    add_line([1500,6000],"return_to_start")

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
    for category,point in [("obtuse",[3800,5200]),("right_angle",[7100,5400]),("acute",[9200+acute_run,6200])]:
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
    write(OUT/"geometry.json",dict(case_id="composite_closed_case",closed=True,lap_count=1,
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
        certificate_milestones=proof,sharp_corner_counts=counts,arc_count=4,
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
    ax.annotate("Start / end: one lap",ref[0],xytext=(1000,6700),arrowprops=dict(arrowstyle="-",lw=.8))
    ax.annotate("",(2600,6300),(1800,6300),arrowprops=dict(arrowstyle="->",color="#444B52",lw=1))
    ax.text(2700,6330,"Travel direction",fontsize=8)
    ax.text(11750,2950,"R = 3000 mm",color="#0072B2",fontsize=9)
    ax.text(0,4100,"R = 1700 mm",color="#0072B2",fontsize=8)
    ax.text(2500,600,"R = 1300 mm",color="#0072B2",fontsize=8)
    ax.text(2400,2450,"R = 600 mm",color="#0072B2",fontsize=8)
    colors=["#A76920","#0072B2","#785A99"]
    positions=[(3300,4600),(6500,4650),(9500,6800)]
    for event,letter,color,position in zip(chosen,"BCD",colors,positions):
        point=np.array(event["point_mm"])
        ax.scatter(*point,s=22,color=color,zorder=5)
        ax.annotate(letter,point,xytext=position,color=color,fontweight="bold",arrowprops=dict(arrowstyle="-",lw=.8,color=color))
    ax.set(aspect="equal",xlabel="X (mm)",ylabel="Y (mm)",xlim=(-1300,14600),ylim=(-650,7400))
    ax.set_title("a  One irregular closed reference: long/short lines, three corner classes, and circular arcs",loc="left",fontsize=10)
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
    fig.suptitle(f"Closed composite path — reference design, not a policy result\nLength {s/1000:.3f} m; complete-lap time lower bound {lower_time:.3f} s at 100 mm/s",fontsize=12)
    for ext in ["png","svg","pdf"]:
        fig.savefig(OUT/f"path_preview.{ext}",dpi=200,facecolor="white")
    plt.close(fig)
    (OUT/"路径设计说明.md").write_text(f"""# 单圈闭合综合长路径

按用户给出的闭合凹凸轮廓图构思，重新设计尺寸、凹槽、凸角和圆弧位置；不复制原图的误差数字或运动结果。往返草案已停止采用。

**当前只完成几何设计与静态审核，不代表主方法已成功。**

| 项目 | 设计与审核结果 |
|---|---|
| 拓扑 | 无自交的单一闭合轮廓，完整走一圈 |
| 起终点 | (1500, 6000) mm，位于同一长直线内部，几何切向连续；实际仍从静止起动、最后制动停稳 |
| 解析总长 | {s/1000:.6f} m |
| 共同离散参考长 | {poly_s[-1]/1000:.6f} m |
| 沿参考100 mm/s匀速时间 | {s/100:.6f} s；这是参考值，不冒充执行时间 |
| 考虑切角的保守时间下界 | **{lower_time:.6f} s > 300 s**，来自必须按序通过的截面；不是靠降速凑时长 |
| 短直线 | 20 mm，连接上侧凹槽的两个135°内角 |
| 最长直线 | {audit['maximum_intrinsic_straight_mm']:.6f} mm |
| 三类角 | {counts['obtuse']}个钝角、{counts['right_angle']}个直角、{counts['acute']}个锐角，全部纳入角点统计 |
| 圆弧 | 4段，半径3000、1300、600、1700 mm；含凸/凹连接 |
| 几何核验 | 无自交、无相邻门冲突、无180°方向歧义；非相邻段最小保守间距{clearance:.6f} mm |
| 弧离散 | 最大弦高误差{audit['max_arc_chord_error_mm']:.9f} mm，不加到1.25 mm允差上 |

机制图三列固定为B：135°钝角(3800,5200)，C：90°直角(7100,5400)，D：45°锐角({9200+acute_run:.6f},6200)。三列均取同一完整模型执行记录，六行为转向、请求进给、前瞻、实际航向、实际曲率和实际速度。图中的20 mm半窗只用于预览；统计前/后窗按geometry.json固定，每边最多10 mm并受短段长度限制。

闭合轮廓不意味着周期运动：首次从静止起动，完成全部有序覆盖后回到起点附近并完整制动、至少20 ms停稳；不能把终端制动后的静止状态与起动段拼接来计算虚构的接缝曲率。

time下界使用每个选定门截面的包围圆：半径ε/cos(δ/2)，直线门为ε。连续包围圆中心距离减去两端半径的和，是任何依次穿过这些门的运动长度下界；除以100 mm/s后仍大于300秒。证明、门索引和半径在geometry_audit.json，可独立核验。

执行器读取同一reference.csv，保留geometry.json中的解析圆弧定义；后续必须检查真实命令相对解析线/弧的1.25 mm距离，不能让离散误差放宽物理允差。CSV接口和Corridor构造已核查；S2调度、策略执行与V/A/J尚未运行。

下一阶段仅围绕这一条公开开发案例取得完整成功、较少拐角降速、更多高速巡航和完整时间收益，并与S2和强固定D作同案例比较。不要求修复另外三类旧路径；一次局部快段不能代替完整一圈。设计本身不冻结最终策略，也不增加算法轮次预算。
""",encoding="utf-8")
    write(OUT/"receipt.json",dict(created_utc=now,status="completed_static_closed_composite_design",
        source_sha256={Path(__file__).name:sha(Path(__file__))},
        outputs_sha256={str(p.relative_to(BASE)):sha(p) for p in OUT.iterdir() if p.is_file()},
        reference_sha256=sha(OUT/"reference.csv"),time_lower_bound_s=lower_time,
        actor_calls=0,physics_ticks=0,training_updates=0,goal_complete=False,method_frozen=False,
        visual_qa="pending"))
    print(json.dumps({k:audit[k] for k in ["analytic_length_mm","ordered_gate_time_lower_bound_s","sharp_corner_counts","reference_points","minimum_nonadjacent_primitive_clearance_lower_bound_mm"]}))


if __name__=="__main__":
    main()
