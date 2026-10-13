"""Design and statically audit one >300 s composite path, without a policy rollout."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from geometry_spec import Corridor

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper_evidence_review/composite_long_case"
EPS, VMAX, ARC_ERROR, SPACING = 1.25, 100.0, .002, 25.0
SPAN, PITCH, RADIUS = 7800.0, 500.0, 250.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    if OUT.exists():
        raise RuntimeError("Geometry design already exists; preserve it and inspect instead of rerunning.")
    OUT.mkdir()
    now = datetime.now(timezone.utc).isoformat()
    write(OUT / "registration.json", {
        "created_utc": now, "work_type": "static_geometry_design_no_training_or_rollout",
        "user_requirement": "One complete composite path with long/short straights, obtuse/right/acute corners, arcs, and completion duration >300 seconds.",
        "rows": 4, "row_span_mm": SPAN, "row_pitch_mm": PITCH,
        "line_sampling_max_mm": SPACING, "arc_chord_error_max_mm": ARC_ERROR,
        "policy_evaluations_allowed_in_this_script": 0, "physics_ticks": 0,
        "source_sha256": {"design_composite_long_case.py": sha(Path(__file__)),
                          "geometry_spec.py": sha(BASE / "geometry_spec.py")},
    })
    (OUT / "figure_contract.md").write_text(
        "# Geometry preview contract\n\n"
        "Claim: one fully specified reference contains long/short lines, all three interior-angle classes and arcs, "
        "with an ordered-gate travel lower bound above 300 s at 100 mm/s. This is not an executed policy trajectory.\n\n"
        "Python/matplotlib only. Use the user's existing equal-aspect full-XY plus fixed local-window visual grammar; "
        "three local windows correspond to obtuse/right/acute columns. No external journal precedent is claimed. "
        "The preview is a geometry-design artifact, not a submission-ready performance figure. "
        "Export editable SVG, vector PDF and PNG; inspect the whole preview and preserve all source coordinates.\n",
        encoding="utf-8")
    acute_run = 50 / math.tan(math.radians(67.5))
    local = np.array([[0, 0], [100, 0], [900, 0], [940, 40], [960, 40], [1000, 0],
                      [2200, 0], [2250, 50], [2300, 0], [4200, 0],
                      [4200 + acute_run, 50], [4200 + 2 * acute_run, 0],
                      [7700, 0], [7800, 0]], dtype=float)
    primitives, row_vertices = [], []
    reference = [np.array([0., 0.])]
    primitive_shapes, cumulative = [], 0.0

    def add_line(a, b, row):
        nonlocal cumulative
        vector = b - a
        length = float(np.linalg.norm(vector))
        heading = math.atan2(vector[1], vector[0])
        samples = np.linspace(a, b, int(math.ceil(length / SPACING)) + 1)
        reference.extend(samples[1:])
        record = dict(kind="line", row=row, start_mm=a.tolist(), end_mm=b.tolist(),
                      length_mm=length, start_s_mm=cumulative, end_s_mm=cumulative + length,
                      start_heading_rad=heading, end_heading_rad=heading)
        cumulative += length
        primitives.append(record)
        primitive_shapes.append(LineString(samples))

    def add_arc(row):
        nonlocal cumulative
        direction = 1 if row % 2 == 0 else -1
        x = SPAN if direction == 1 else 0.
        center = np.array([x, row * PITCH + RADIUS])
        start, sweep = -math.pi / 2, direction * math.pi
        n = int(math.ceil(abs(sweep) / (2 * math.acos(1 - ARC_ERROR / RADIUS))))
        angles = np.linspace(start, start + sweep, n + 1)
        samples = center + RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])
        samples[0] = [x, row * PITCH]
        samples[-1] = [x, (row + 1) * PITCH]
        reference.extend(samples[1:])
        length = RADIUS * abs(sweep)
        primitives.append(dict(kind="arc", after_row=row, center_mm=center.tolist(), radius_mm=RADIUS,
                               start_angle_rad=start, sweep_rad=sweep, sample_segments=n,
                               start_mm=samples[0].tolist(), end_mm=samples[-1].tolist(),
                               start_heading_rad=wrap(start + direction * math.pi / 2),
                               end_heading_rad=wrap(start + sweep + direction * math.pi / 2),
                               chord_error_bound_mm=RADIUS * (1 - math.cos(abs(sweep) / n / 2)),
                               length_mm=length, start_s_mm=cumulative, end_s_mm=cumulative + length))
        cumulative += length
        primitive_shapes.append(LineString(samples))

    for row in range(4):
        vertices = local.copy()
        if row % 2:
            vertices[:, 0] = SPAN - vertices[:, 0]
        vertices[:, 1] += row * PITCH
        row_vertices.append(vertices)
        for a, b in zip(vertices[:-1], vertices[1:]):
            add_line(a, b, row)
        if row < 3:
            add_arc(row)

    ref = np.array(reference)
    line = LineString(ref)
    assert line.is_simple
    assert np.linalg.norm(ref[0] - ref[-1]) > 1e-6
    corridor = Corridor(ref, EPS, closed=False)
    conflicts = corridor.adjacent_gate_conflicts()
    assert not corridor.ambiguities and not conflicts
    assert corridor.lengths.min() > 1e-8
    connections = [float(np.linalg.norm(np.array(a["end_mm"]) - b["start_mm"]))
                   for a, b in zip(primitives[:-1], primitives[1:])]
    assert max(connections) < 1e-8

    events = []
    arc_tangent_errors = []
    for i, (a, b) in enumerate(zip(primitives[:-1], primitives[1:])):
        delta = math.degrees(wrap(b["start_heading_rad"] - a["end_heading_rad"]))
        if a["kind"] == "arc" or b["kind"] == "arc":
            arc_tangent_errors.append(abs(delta))
            assert abs(delta) < 1e-8
        elif abs(delta) > 1e-7:
            beta = 180 - abs(delta)
            category = "right_angle" if abs(beta - 90) < 1e-6 else "obtuse" if beta > 90 else "acute"
            events.append(dict(id=f"corner_{len(events):02d}", primitive_before=i, row=a["row"],
                               point_mm=a["end_mm"], analytic_s_mm=a["end_s_mm"],
                               interior_angle_deg=beta, signed_heading_change_deg=delta, category=category,
                               window_before_mm=min(10., a["length_mm"] / 2),
                               window_after_mm=min(10., b["length_mm"] / 2)))
    # Select the three mechanism columns by geometry before any policy evaluation.
    chosen = []
    for category, point in [("obtuse", [940., 40.]), ("right_angle", [2250., 50.]),
                            ("acute", [4200 + acute_run, 50.])]:
        event = next(e for e in events if e["row"] == 0 and
                     e["category"] == category and np.allclose(e["point_mm"], point))
        event["mechanism_column"] = category
        chosen.append(event)
    counts = {c: sum(e["category"] == c for e in events) for c in ["obtuse", "right_angle", "acute"]}
    assert counts == {"obtuse": 32, "right_angle": 4, "acute": 4}

    minimum_clearance = float("inf")
    nearest_pair = None
    for i in range(len(primitives)):
        for j in range(i + 2, len(primitives)):
            # Adjacent primitives share an endpoint by definition; compare the rest.
            clearance = primitive_shapes[i].distance(primitive_shapes[j])
            allowance = sum(primitives[k].get("chord_error_bound_mm", 0.) for k in [i, j])
            conservative = clearance - allowance
            if conservative < minimum_clearance:
                minimum_clearance, nearest_pair = conservative, [i, j]
    assert minimum_clearance > 2 * EPS

    # The selected milestones lie on straight, collinear, actual corridor gates.
    # Their feasible crossing sets fit disks of radius EPS about the centers.
    # The triangle inequality then lower-bounds every complete ordered path.
    centers, certificate_gates = [ref[0]], []
    for row in range(4):
        for local_x in [100., 7700.]:
            point = np.array([local_x if row % 2 == 0 else SPAN - local_x, row * PITCH])
            matches = np.flatnonzero(np.linalg.norm(ref - point, axis=1) < 1e-7)
            assert len(matches) == 1
            vertex = int(matches[0])
            assert np.allclose(corridor.tangents[vertex - 1], corridor.tangents[vertex], atol=1e-12)
            assert any(g.vertex_index == vertex for g in corridor.gates)
            centers.append(point)
            certificate_gates.append(dict(vertex_index=vertex, center_mm=point.tolist(),
                                          crossing_set_radius_bound_mm=EPS))
    centers.append(ref[-1])
    centers = np.array(centers)
    center_distance = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    # EPS is conservatively allowed at both endpoints as well as each gate.
    lower_length = float(np.maximum(0., center_distance - 2 * EPS).sum())
    lower_time = lower_length / VMAX
    assert lower_time > 300
    assert np.all(np.diff([g["vertex_index"] for g in certificate_gates]) > 0)

    csv = OUT / "reference.csv"
    np.savetxt(csv, ref, delimiter=",", header="x_mm,y_mm", comments="", fmt="%.12f", encoding="utf-8")
    roundtrip = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert np.max(np.abs(roundtrip - ref)) < 1e-9
    poly_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(roundtrip, axis=0), axis=1))]
    for event in events:
        vertex = int(np.argmin(np.linalg.norm(roundtrip - event["point_mm"], axis=1)))
        assert np.linalg.norm(roundtrip[vertex] - event["point_mm"]) < 1e-8
        event["reference_vertex_index"] = vertex
        event["reference_s_mm"] = float(poly_s[vertex])
    write(OUT / "geometry.json", dict(case_id="composite_long_case", closed=False,
                                     primitives=primitives, sharp_corners=events,
                                     mechanism_columns=[e["id"] for e in chosen],
                                     analytic_length_mm=cumulative, reference_file="reference.csv",
                                     reference_sha256=sha(csv), sample_max_straight_step_mm=SPACING,
                                     arc_chord_error_limit_mm=ARC_ERROR))
    audit = dict(status="static_geometry_checks_passed_no_dynamic_success_claim",
                 analytic_length_mm=cumulative, sampled_reference_length_mm=float(poly_s[-1]),
                 reference_constant_100_time_s=cumulative / VMAX,
                 ordered_gate_lower_length_mm=lower_length,
                 ordered_gate_speed_only_time_lower_bound_s=lower_time,
                 duration_above_300s_certified=True,
                 lower_bound_proof="Sum max(0, distance between consecutive milestone centers - 2*epsilon)/Vmax. Milestones are mandatory collinear straight-section gates, start and terminal endpoint. Braking and rest only increase time.",
                 milestone_gates=certificate_gates, self_intersection=False,
                 minimum_nonadjacent_primitive_clearance_lower_bound_mm=minimum_clearance,
                 nearest_primitive_pair=nearest_pair, adjacent_gate_conflicts=conflicts,
                 geometry_ambiguities=corridor.ambiguities, sharp_corner_counts=counts,
                 arc_count=3, arc_radius_mm=RADIUS,
                 max_arc_chord_error_mm=max(p.get("chord_error_bound_mm", 0) for p in primitives),
                 max_analytic_arc_join_heading_error_deg=max(arc_tangent_errors),
                 reference_points=len(ref), ordered_gate_count=len(corridor.gates),
                 min_intrinsic_line_mm=min(p["length_mm"] for p in primitives if p["kind"] == "line"),
                 max_intrinsic_line_mm=max(p["length_mm"] for p in primitives if p["kind"] == "line"),
                 actor_calls=0, physics_ticks=0, training_updates=0,
                 policy_success=None, formal_model_adoption=False,
                 common_reference_format="CSV XY read by numpy.loadtxt; Corridor construction and gate conflicts checked. S2 scheduling and policy physical execution not run.")
    write(OUT / "geometry_audit.json", audit)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "path.simplify": False})
    fig = plt.figure(figsize=(11.3, 6.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    full = fig.add_subplot(gs[0, :])
    full.plot(ref[:, 0], ref[:, 1], color="#495866", lw=1.2)
    for primitive, shape in zip(primitives, primitive_shapes):
        if primitive["kind"] == "arc":
            xy = np.array(shape.coords)
            full.plot(xy[:, 0], xy[:, 1], color="#0072B2", lw=2)
    for row in range(4):
        full.text(5800, row * PITCH + 110, f"Row {row + 1}  " + ("→" if row % 2 == 0 else "←"), fontsize=9)
    full.plot(*ref[0], marker="^", ms=6, color="black")
    full.plot(*ref[-1], marker="s", ms=5, color="black")
    full.annotate("Start", ref[0], xytext=(150, -160))
    full.annotate("End", ref[-1], xytext=(150, 1630))
    colors = ["#A76920", "#0072B2", "#6B5192"]
    for event, letter, color in zip(chosen, "BCD", colors):
        point = np.array(event["point_mm"])
        full.scatter(*point, s=25, color=color, zorder=4)
        full.annotate(letter, point, xytext=(point[0] + 90, -220), color=color, fontweight="bold",
                      arrowprops=dict(arrowstyle="-", color=color, lw=.8))
    full.set(aspect="equal", xlabel="X (mm)", ylabel="Y (mm)", xlim=(-420, 8220), ylim=(-330, 1760))
    full.set_title("a  One complete reference: four rows, short features, three angle classes, and three circular arcs", loc="left")
    for col, (event, color) in enumerate(zip(chosen, colors)):
        ax = fig.add_subplot(gs[1, col])
        center = np.array(event["point_mm"])
        start, end = center - 20, center + 20
        mask = np.all((ref >= start - 30) & (ref <= end + 30), axis=1)
        indices = np.flatnonzero(mask)
        local_ref = ref[max(0, indices[0] - 1):min(len(ref), indices[-1] + 2)]
        region = LineString(local_ref).buffer(EPS, resolution=32)
        bx, by = region.exterior.xy
        ax.fill(bx, by, color="#E7EFF5")
        ax.plot(bx, by, color="#8195A5", ls="--", lw=.8)
        ax.plot(local_ref[:, 0], local_ref[:, 1], color=color, lw=1.3)
        ax.scatter(*center, s=18, color=color)
        ax.set(xlim=(start[0], end[0]), ylim=(start[1], end[1]), aspect="equal", xlabel="X (mm)", ylabel="Y (mm)")
        title = ["Obtuse", "Right angle", "Acute"][col]
        ax.set_title(f"{'bcd'[col]}  {title}: interior angle {event['interior_angle_deg']:.0f}°", loc="left", fontsize=10)
        ax.text(.03, .04, f"Heading change {event['signed_heading_change_deg']:+.0f}°\nCorridor half-width 1.25 mm",
                transform=ax.transAxes, fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    fig.suptitle(f"Reference design only — no policy rollout\nLength {cumulative/1000:.3f} m; mandatory-gate time lower bound {lower_time:.3f} s at 100 mm/s", fontsize=12)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"path_preview.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    report = f"""# 单一综合长路径设计

状态：几何设计及静态审核完成，尚未执行策略，不代表目标实验已经成功。

- 一条开放连续路径，4行，每行横向跨度7800 mm，行间距500 mm；3段半径250 mm的相切半圆连接相邻行。
- 解析参考总长 **{cumulative/1000:.6f} m**；离散共同参考 **{poly_s[-1]/1000:.6f} m**。
- 沿解析参考以100 mm/s匀速走完需 **{cumulative/100:.6f} s**，但此值不是允许切角运动的严格下界。
- 根据8个必须按序穿过的直线截面及起终点，允许每处1.25 mm误差后，实际路程仍至少 **{lower_length/1000:.6f} m**，所以任意完整合规执行的时间至少 **{lower_time:.3f} s > 300 s**。这一下界还未计加减速与停稳。
- 40个解析尖角：32个钝角、4个直角、4个锐角；三种角每行均有。短平台20 mm；其余斜线和长直线保留实际毫米尺度，未把短特征等比放大。
- 机制图三列预先选定为首行：135°钝角、90°直角、45°锐角。其位置、符号、固定指标窗和参考弧长见geometry.json；全路径所有40个角均进入完整角点统计。
- 静态检查：无中心线自交、无180°方向歧义、无相邻门冲突；非相邻几何段最小保守间距{minimum_clearance:.6f} mm；弧离散最大弦高误差{audit['max_arc_chord_error_mm']:.8f} mm。
- 解析弧与直线连接处航向连续，曲率存在跳变；实际运动的V/A/J仍须由后续完整执行审核，不能由几何图推断。

共同参考：reference.csv。解析构成、角点和局部窗：geometry.json。下界证明及检查值：geometry_audit.json。图：path_preview.png/svg/pdf。

该案例是按用户要求专门设计的公开开发案例。所有方法必须用同一参考与相同物理限制；高速统计同时报告占比/路程、角点速度亏损和完整时间，避免长直线掩盖角点退化。既有S2/固定法的旧路径时间不能直接作为本案例对照时间。

本阶段只固定几何，不冻结最终方法。没有PPO更新、离线拟合、参数搜索、actor调用或物理仿真；不消耗新的算法轮次。后续先提交一个有界的同案例实现与对照方案，不能自动扩展为修复全部旧路径或无限优化。
"""
    (OUT / "路径设计说明.md").write_text(report, encoding="utf-8")
    write(OUT / "receipt.json", dict(created_utc=now, status="completed_static_long_composite_geometry_design",
                                     source_sha256={"design_composite_long_case.py": sha(Path(__file__))},
                                     outputs_sha256={str(p.relative_to(BASE)): sha(p) for p in OUT.iterdir() if p.is_file()},
                                     reference_sha256=sha(csv), duration_lower_bound_s=lower_time,
                                     actor_calls=0, physics_ticks=0, training_updates=0,
                                     goal_complete=False, method_frozen=False,
                                     figure_visual_review="pending"))
    print(json.dumps({k: audit[k] for k in ["analytic_length_mm", "ordered_gate_speed_only_time_lower_bound_s", "sharp_corner_counts", "reference_points", "minimum_nonadjacent_primitive_clearance_lower_bound_mm"]}))


if __name__ == "__main__":
    main()
