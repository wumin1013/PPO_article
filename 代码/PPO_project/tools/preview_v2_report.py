"""Read-only V2 report from saved validation runs; never executes a policy.

Default excludes all test results. --include-frozen-test is an explicit reporting
gate and requires freeze.json; it still only reads previously saved evaluations.
All source hashes, failure rows and paired-success sample definitions are saved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import revision_plot as common

ROOT = common.ROOT
DATA = ROOT / "PPO_project/data/preview_v2"
DEFAULT_RUNS = ROOT / "论文项目/revision_20260908/preview_v2"
VARIANTS = ("fixed", "residual", "absolute")
COLORS = {"pretrain_short":"#60666c", "pretrain_mid":"#d78332", "pretrain_long":"#1f77b4",
          "fixed":"#60666c", "residual":"#1f77b4", "absolute":"#d78332", "pretrain_selected":"#1f77b4"}
LABELS = {"pretrain_short":"Fixed short", "pretrain_mid":"Fixed medium", "pretrain_long":"Fixed long",
          "fixed":"Adaptation off", "residual":"Residual preview", "absolute":"Absolute preview", "pretrain_selected":"Selected fixed"}
FAMILIES = ("mixed_lengths", "dense_corners", "line_curve")
SOURCES = common.SOURCES
sys.path.insert(0,str(ROOT/"PPO_project"))
from src.utils.comparison_metrics import project_points_to_polyline, interpolate_polyline


def trace(path):
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    z = np.load(path)
    return dict(zip(z["columns"].tolist(), z["data"].T))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n",encoding="utf-8")


def discover(base, include_frozen_test=False):
    rows, budgets = [], []
    if include_frozen_test:
        freeze = common.read_json(base/"freeze.json")
        manifest_hash = hashlib.sha256((DATA/"manifest.json").read_bytes()).hexdigest()
        if manifest_hash != freeze["path_manifest_sha256"]:
            raise ValueError("Frozen test manifest mismatch")
        for relative, expected in freeze["models"].items():
            if hashlib.sha256((base/relative/"checkpoint.pt").read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen checkpoint mismatch: {relative}")
        for relative, expected in freeze["configs"].items():
            if hashlib.sha256((base/relative/"config.yaml").read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen configuration mismatch: {relative}")
    for stage in ["screen","pretrain","finetune","nnc_auxiliary/runs"]:
        for path in sorted((base/stage).glob("*/training_summary.json")):
            budgets.append(dict(common.read_json(path),stage=stage,source=path.relative_to(ROOT).as_posix()))
        for path in sorted((base/stage).glob("*/evaluation_*/summary.json")):
            if "test" in path.parent.name and not include_frozen_test:
                continue
            if "test" not in path.parent.name and "validation" not in path.parent.name and "regression" not in path.parent.name:
                continue
            for entry in common.read_json(path).get("rows", []):
                row=dict(entry, stage=stage,evaluation=path.parent.name,source=path.relative_to(ROOT).as_posix(),run_dir=str(path.parent.parent))
                rows.append(row)
    return rows,budgets


def family(path_id):
    return next((f for f in FAMILIES if path_id.startswith(f)),path_id)


def aggregate(rows):
    output=[]
    keys=sorted({(r["stage"],r["evaluation"],r["variant"],r["path_id"]) for r in rows})
    for stage,evaluation,variant,path_id in keys:
        subset=[r for r in rows if (r["stage"],r["evaluation"],r["variant"],r["path_id"])==(stage,evaluation,variant,path_id)]
        mean,sd=common.mean_sd([r["ordered_progress"] for r in subset])
        error_mean,error_sd=common.mean_sd([r["contour_error_max_mm"] for r in subset])
        mean_error,mean_error_sd=common.mean_sd([r["contour_error_mean_mm"] for r in subset])
        output.append(dict(stage=stage,evaluation=evaluation,variant=variant,path_id=path_id,n=len(subset),
            independent_seeds=sorted({r["seed"] for r in subset}),successes=sum(r["success"] for r in subset),
            arrivals=sum(r["path_completed"] for r in subset),geometry_failures=sum(not r["geometry_feasible"] for r in subset),
            ordered_progress_mean=mean,ordered_progress_sd=sd,peak_error_mean_mm=error_mean,peak_error_sd_mm=error_sd,
            mean_error_mean_mm=mean_error,mean_error_sd_mm=mean_error_sd,
            failure_counts={reason:sum(reason in r["failure_reasons"] for r in subset) for reason in sorted({reason for r in subset for reason in r["failure_reasons"]})}))
    return output


def paired_summary(rows, evaluation):
    rows=[r for r in rows if r["stage"]=="finetune" and r["evaluation"]==evaluation]
    lookup={v:{(r["seed"],r["path_id"]):r for r in rows if r["variant"]==v} for v in VARIANTS}
    common_all=set.intersection(*[{key for key,row in lookup[v].items() if row["success"]} for v in VARIANTS])
    result={"evaluation":evaluation,"all_method_common_successful_pairs":[list(k) for k in sorted(common_all)],"pairwise":[]}
    for adaptive in ("residual","absolute"):
        common_keys=sorted({key for key,row in lookup["fixed"].items() if row["success"]} & {key for key,row in lookup[adaptive].items() if row["success"]})
        by_path=[]
        for path_id in sorted({r["path_id"] for r in rows}):
            keys=[key for key in common_keys if key[1]==path_id]
            differences=[lookup["fixed"][key]["elapsed_time_s"]-lookup[adaptive][key]["elapsed_time_s"] for key in keys]
            mean,sd=common.mean_sd(differences)
            relative=[100*(lookup["fixed"][key]["elapsed_time_s"]-lookup[adaptive][key]["elapsed_time_s"])/lookup["fixed"][key]["elapsed_time_s"] for key in keys]
            rmean,rsd=common.mean_sd(relative)
            by_path.append(dict(path_id=path_id,n_common=len(keys),paired_seeds=[key[0] for key in keys],
                fixed_minus_adaptive_time_mean_s=mean,fixed_minus_adaptive_time_sd_s=sd,
                relative_time_reduction_mean_percent=rmean,relative_time_reduction_sd_percent=rsd))
        result["pairwise"].append(dict(adaptive=adaptive,common_keys=[list(k) for k in common_keys],by_path=by_path))
    return result


def nnc_validation_summary(rows, budgets, base):
    """Audit the separately frozen NNC records without touching any test trace."""
    selected=[r for r in rows if r["stage"]=="nnc_auxiliary/runs" and "validation" in r["evaluation"]]
    if not selected:
        return None
    freeze=common.read_json(base/"nnc_auxiliary/freeze.json")
    for key,filename in [("models","checkpoint.pt"),("configs","config.yaml")]:
        for relative,expected in freeze[key].items():
            path=base/"nnc_auxiliary"/relative/filename
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual!=expected:
                raise ValueError(f"NNC frozen {filename} mismatch: {relative}")
            SOURCES[path.relative_to(ROOT).as_posix()]=actual
    for row in selected:
        relative="runs/"+Path(row["run_dir"]).name
        if row["checkpoint_sha256"]!=freeze["models"][relative] or row["config_sha256"]!=freeze["configs"][relative]:
            raise ValueError(f"NNC evaluation provenance mismatch: {relative}")
    costs=[r for r in budgets if r["stage"]=="nnc_auxiliary/runs"]
    groups=[]
    for item in aggregate(selected):
        subset=[r for r in selected if (r["evaluation"],r["path_id"])==(item["evaluation"],item["path_id"])]
        groups.append(dict(item,constraint_failures={
            key:sum(not row["constraints"][key]["satisfied"] for row in subset)
            for key in sorted(subset[0]["constraints"])},
            kinematic_failures=sum(any(not flag["satisfied"] for flag in row["constraints"].values()) for row in subset)))
    return dict(scope="Auxiliary NNC validation only; excluded from paired adaptive-preview comparisons",
        independent_training_seeds=sorted({r["seed"] for r in costs}),
        steps_per_seed=sorted({r["steps"] for r in costs}),executed_training_steps=sum(r["steps"] for r in costs),
        inherited_training_steps=sum(r.get("pretraining_steps",0) for r in costs),
        training_wall_s=sum(r["training_wall_s"] for r in costs),groups=groups,
        success_count=sum(r["success"] for r in selected),arrival_count=sum(r["path_completed"] for r in selected),
        run_count=len(selected),freeze=freeze,
        terminal_rest_checked=all(r["terminal_rest_checked"] for r in selected))


def nnc_validation_tex(summary):
    if not summary:
        return []
    seeds=summary["independent_training_seeds"]
    steps=", ".join(f"{step:,}" for step in summary["steps_per_seed"])
    out=[r"\subsection{Auxiliary NNC validation audit}",
        f"The existing repository NNC configuration was independently retrained with {len(seeds)} seeds ({', '.join(str(s) for s in seeds)}) and {steps} steps per seed, totaling {summary['executed_training_steps']:,} executed learning steps with no inherited pretraining. It retains the common V2 ordered-progress objective and disables KCM, preview observations, preview-based reference guidance, and adaptive corner weighting. This auxiliary configuration is reported separately from the paired preview comparison. Its frozen checkpoint and configuration hashes were checked against every saved validation record.",""]
    outcome_rows=[]
    violation_rows=[]
    for item in summary["groups"]:
        label=common.FAMILY_LABEL.get(family(item["path_id"]),common.escaped(item["path_id"]))
        peak=f"${item['peak_error_mean_mm']:.3f}\\pm{item['peak_error_sd_mm']:.3f}$" if item['peak_error_sd_mm'] is not None else f"${item['peak_error_mean_mm']:.3f}$"
        mean=f"${item['mean_error_mean_mm']:.3f}\\pm{item['mean_error_sd_mm']:.3f}$" if item['mean_error_sd_mm'] is not None else f"${item['mean_error_mean_mm']:.3f}$"
        outcome_rows.append([label,item["n"],f"{item['arrivals']}/{item['n']}",f"{item['successes']}/{item['n']}",
            f"{item['geometry_failures']}/{item['n']}",f"{item['kinematic_failures']}/{item['n']}",peak,mean])
        flags=item["constraint_failures"]
        violation_rows.append([label]+[f"{flags[key]}/{item['n']}" for key in ["feed_mm_s","feed_acceleration_mm_s2","feed_jerk_mm_s3",
            "angular_velocity_rad_s","angular_acceleration_rad_s2","angular_jerk_rad_s3"]])
    out+=common.table("Auxiliary NNC nominal-validation outcomes. Arrival means ordered path coverage and start/end proximity; strict success also requires whole-trace contour tolerance and the independently audited scalar-feed and turning constraints. Error mean and sample SD are across independently trained seeds on the same path; all failed runs remain.",
        "tab:preview_v2_nnc_validation",["Family","n","Arrival","Strict success","Geometry failures","Motion failures","Peak error/mm","Mean error/mm"],outcome_rows)
    constraint_table=common.table("NNC validation runs with at least one violation in each independently reconstructed motion quantity. A run may violate several quantities and appears in every applicable column; counts are not interpolator samples.",
        "tab:preview_v2_nnc_constraints",["Family",r"$v$",r"$\dot v$",r"$\ddot v$",r"$\omega$",r"$\dot\omega$",r"$\ddot\omega$"],violation_rows)
    out += [line.replace(r"\end{tabular}}",r"\end{tabular}") for line in constraint_table if not line.startswith(r"\resizebox")]
    out += [f"The auxiliary audit contains {summary['arrival_count']}/{summary['run_count']} ordered arrivals and {summary['success_count']}/{summary['run_count']} strict successes. An environment termination labeled success does not override an independent constraint failure. Arrival times of constraint-violating traces are excluded from feasible-efficiency claims.",
        "The evaluated episodes do not impose terminal rest. Cartesian axis derivatives are diagnostic outputs because separate axis limits are not implemented in this comparison. These simulation traces are not machine-ready command trajectories.",""]
    return out


def screen_figure(rows):
    screen=[r for r in rows if r["stage"]=="screen" and r["evaluation"]=="evaluation_validation_nominal"]
    if not screen:
        return
    fig,axes=plt.subplots(3,2,figsize=(7.2,7.0),constrained_layout=True)
    for i,name in enumerate(FAMILIES):
        reference_path=DATA/"paths"/f"{name}_validation.csv"
        SOURCES[reference_path.relative_to(ROOT).as_posix()]=hashlib.sha256(reference_path.read_bytes()).hexdigest()
        reference=np.genfromtxt(reference_path,delimiter=",",skip_header=1)
        axes[i,0].plot(reference[:,0],reference[:,1],color="#bbbbbb",lw=2,label="Reference")
        for row in screen:
            if family(row["path_id"])!=name:
                continue
            path=ROOT/Path(row["source"]).parent/f"{row['path_id']}.npz"
            d=trace(path)
            config=common.read_yaml(ROOT/Path(row["source"]).parent/"effective_config.yaml")
            distance=config["reward_weights"]["lookahead"]["distance"]
            label=f"{distance:g} mm: {'success' if row['success'] else row['termination_reason']}"
            color=COLORS.get(row["variant"],"#333333")
            axes[i,0].plot(d["x_mm"],d["y_mm"],color=color,label=label)
            axes[i,0].scatter(d["x_mm"][-1],d["y_mm"][-1],s=16,marker="o" if row["success"] else "x",color=color)
            axes[i,1].plot(d["timestamp_s"],d["v_mm_s"],color=color,label=f"{distance:g} mm")
        axes[i,0].set_aspect("equal",adjustable="datalim")
        axes[i,0].set(xlabel="x (mm)",ylabel="y (mm)",title=common.FAMILY_LABEL[name])
        axes[i,1].set(xlabel="Elapsed simulation time (s)",ylabel="Feed (mm/s)")
        axes[i,0].legend(fontsize=6,loc="best")
        for j in range(2):
            axes[i,j].text(-.13,1.03,chr(97+i*2+j),transform=axes[i,j].transAxes,weight="bold")
            axes[i,j].grid(alpha=.15,ls=":")
    fig.suptitle("V2 fixed-distance screening: every validation trajectory retained",fontsize=10)
    common.save(fig,"preview_v2_validation_screen")


def curve_diagnosis(rows, output):
    subset=[r for r in rows if r["stage"]=="screen" and r["variant"]=="pretrain_long" and r["path_id"]=="line_curve_validation" and r["evaluation"]=="evaluation_validation_nominal"]
    if not subset:
        return None
    row=subset[0]
    path=ROOT/Path(row["source"]).parent/f"{row['path_id']}.npz"
    d=trace(path)
    config=common.read_yaml(ROOT/Path(row["source"]).parent/"effective_config.yaml")
    t=d["timestamp_s"];finite=np.isfinite(d["control_status__theta_ref"])
    heading_error=(d["heading_rad"]-d["control_status__theta_ref"]+np.pi)%(2*np.pi)-np.pi
    reversal=np.flatnonzero(finite & (d["control_status__heading_rate_intent"]<0) & (d["omega_rad_s"]>0) & (heading_error>0))
    crossing_start=next((int(i) for i in reversal if t[i]>.5*t[-1]),int(reversal[0]) if len(reversal) else None)
    plateau=np.flatnonzero(np.isclose(d["reward_components__ordered_progress"],row["ordered_progress"],atol=1e-10))
    compact_keys=["timestamp_s","x_mm","y_mm","v_mm_s","lookahead_mm","heading_rad","control_status__theta_ref",
        "control_status__heading_rate_policy","control_status__heading_rate_intent","omega_rad_s","alpha_rad_s2",
        "angular_jerk_rad_s3","contour_error_env_mm","reward_components__ordered_progress"]
    last={key:float(d[key][-1]) for key in compact_keys}
    reference=np.genfromtxt(DATA/"paths/line_curve_validation.csv",delimiter=",",skip_header=1)
    geometric_probes=[]
    distances=common.read_json(output.parent/"protocol.json")["fixed_screen_distances_mm"]
    for index in np.unique(np.linspace(1,len(t)-1,7).astype(int)):
        position=np.array([d["x_mm"][index],d["y_mm"][index]])
        s=float(project_points_to_polyline(position[None],reference,closed=False).raw_s[0])
        for distance in distances:
            target=interpolate_polyline(reference,s+distance,closed=False)
            delta=target-position
            chord=position+np.linspace(0,1,101)[:,None]*delta
            sampled_error=project_points_to_polyline(chord,reference,closed=False).contour_error
            geometric_probes.append({"timestamp_s":float(t[index]),"position_mm":position.tolist(),
                "preview_mm":distance,"los_heading_rad":float(np.arctan2(delta[1],delta[0])),
                "sampled_chord_error_max_mm":float(sampled_error.max()),"chord_samples":101})
    audit={"source":path.relative_to(ROOT).as_posix(),"scope":"saved validation trace only; descriptive diagnosis, no counterfactual policy evaluation",
        "source_sha256":SOURCES[path.relative_to(ROOT).as_posix()],"final":last,
        "final_heading_minus_reference_rad":float(heading_error[-1]),
        "positive_rotation_after_negative_intent_samples":int(len(reversal)),
        "late_reversal_sample":None if crossing_start is None else {key:float(d[key][crossing_start]) for key in compact_keys},
        "final_ordered_progress_plateau_begins_s":float(t[plateau[0]]) if len(plateau) else None,
        "geometry_feasible":row["geometry_feasible"],"constraint_flags":{k:v["satisfied"] for k,v in row["constraints"].items()},
        "hypothesis":"Long LOS anticipates the upcoming curve; high-gain heading correction and jerk/acceleration-limited turn reversal produce overshoot before the first straight ends.",
        "causal_limit":"Distances were independently trained. This trace cannot isolate preview distance from learned policy or common heading-control dynamics.",
        "path_independent_candidate_features":["multi-scale LOS chord deviation relative to one-sided tolerance", "signed heading error together with angular velocity/acceleration", "remaining lateral tolerance and jerk-limited angular stopping excursion", "curvature-sign changes across preview scales"],
        "same_state_geometric_probes":geometric_probes,
        "geometric_probe_limit":"Only reference geometry is recomputed at stored identical physical states. No policy was executed; sampled chord error is not a continuous bound or a performance counterfactual.",
        "suggested_next_diagnostic":"Compare preview directions from stored identical physical states; any controller change must be shared with a retrained fixed-preview control. No path-ID-conditioned distance rule."}
    write_json(output/"validation_curve_diagnosis.json",audit)
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.8),constrained_layout=True)
    axes[0,0].plot(reference[:,0],reference[:,1],color="#bbbbbb",lw=2,label="Reference")
    axes[0,0].plot(d["x_mm"],d["y_mm"],color="#1f77b4",label="Fixed long")
    axes[0,0].scatter(d["x_mm"][-1],d["y_mm"][-1],marker="x",s=35,color="#9a4a43")
    axes[0,0].set(xlim=(-.3,7.0),ylim=(-.5,4.3),xlabel="x (mm)",ylabel="y (mm)",title="First line-to-curve transition")
    axes[0,0].set_aspect("equal",adjustable="box")
    axes[0,0].legend(fontsize=6)
    axes[0,1].plot(t,d["heading_rad"],color="#1f77b4",label="Executed heading")
    axes[0,1].plot(t,d["control_status__theta_ref"],color="#d78332",label="LOS reference")
    axes[0,1].set(ylabel="Heading (rad)",xlabel="Time (s)",title="Heading overshoot")
    axes[0,1].legend(fontsize=6)
    axes[1,0].plot(t,d["control_status__heading_rate_intent"],color="#d78332",label="Guided intent")
    axes[1,0].plot(t,d["omega_rad_s"],color="#1f77b4",label="Executed")
    axes[1,0].axhline(0,color="#999999",lw=.6)
    axes[1,0].set(ylabel="Angular velocity (rad/s)",xlabel="Time (s)",title="Intent reversal precedes output reversal")
    axes[1,0].legend(fontsize=6)
    axes[1,1].plot(t,d["contour_error_env_mm"],color="#1f77b4",label="Contour error")
    axes[1,1].axhline(config["environment"]["epsilon"]/2,color="#9a4a43",ls="--",label="One-sided tolerance")
    axes[1,1].set(ylabel="Error (mm)",xlabel="Time (s)",title="Geometric failure at low feed")
    axes[1,1].legend(fontsize=6)
    for i,ax in enumerate(axes.ravel()):
        ax.text(-.12,1.03,chr(97+i),transform=ax.transAxes,weight="bold");ax.grid(alpha=.15,ls=":")
    common.save(fig,"preview_v2_curve_diagnosis")
    md=["# V2 验证曲线失败诊断","","来源为已保存的固定长前瞻筛选轨迹；未运行测试集或新的反事实策略。",
        f"终止时间 {last['timestamp_s']:.3f} s，位置 ({last['x_mm']:.6f}, {last['y_mm']:.6f}) mm，进给 {last['v_mm_s']:.6f} mm/s。",
        f"末端航向 {last['heading_rad']:.6f} rad，而 LOS 参考为 {last['control_status__theta_ref']:.6f} rad；偏差 {heading_error[-1]:.6f} rad。",
        f"指导角速度意图为 {last['control_status__heading_rate_intent']:.6f} rad/s，实际角速度仍为 {last['omega_rad_s']:.6f} rad/s。",
        f"末端轮廓误差 {last['contour_error_env_mm']:.6f} mm；独立轨迹判据记录几何失败。所有原始运动学判据结果：{audit['constraint_flags']}。","",
        "长 LOS 在直线结束前已预见后面的弯曲，角速度意图经过高增益航向校正后快速反转；有界角加速度与角捷度限制了转向制动，输出继续转向造成过冲。因这些距离模型分别训练，当前证据说明时序一致的机制假设，不能单独证明距离的因果影响。","",
        "可供后续同状态诊断的特征：多尺度 LOS 弦偏离/单侧允差；有符号航向误差与角速度、角加速度的联合状态；剩余横向裕量与角制动位移；预瞻范围内曲率符号变化。禁止按 path_id 选择距离。若修改共同转向引导，需要同时为固定前瞻重训对照，不能把公共修复收益归为自适应收益。",""]
    md += ["## 已保存同状态几何探针","","在末端相同物理状态计算各距离的 LOS 方向和101点采样弦偏离；不执行策略、不产生新轨迹，也不据此声称性能反事实。全部7个时刻的探针保存在JSON中。","",
           "| 前瞻/mm | LOS方向/rad | 采样弦最大偏离/mm |","|---:|---:|---:|"]
    for probe in geometric_probes:
        if probe["timestamp_s"]==float(t[-1]):
            md.append(f"| {probe['preview_mm']:g} | {probe['los_heading_rad']:.6f} | {probe['sampled_chord_error_max_mm']:.6f} |")
    md.append("")
    (output/"validation_curve_diagnosis.md").write_text("\n".join(md),encoding="utf-8")
    return audit


def preview_mechanism(rows):
    selected=[r for r in rows if r["stage"]=="finetune" and r["evaluation"]=="evaluation_validation_nominal"]
    if not selected:
        return False
    seed=min(r["seed"] for r in selected)
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.2),constrained_layout=True)
    for j,name in enumerate(FAMILIES):
        for row in selected:
            if family(row["path_id"])!=name or row["seed"]!=seed:
                continue
            d=trace(ROOT/Path(row["source"]).parent/f"{row['path_id']}.npz")
            for i,metric in enumerate(["lookahead_mm","v_mm_s"]):
                axes[i,j].plot(d["timestamp_s"],d[metric],color=COLORS[row["variant"]],label=LABELS[row["variant"]])
        axes[0,j].set_title(common.FAMILY_LABEL[name],fontsize=9)
        for i in range(2):
            axes[i,j].set_xlabel("Time (s)");axes[i,j].grid(alpha=.15,ls=":")
            axes[i,j].text(-.13,1.05,chr(97+i*3+j),transform=axes[i,j].transAxes,weight="bold")
    axes[0,0].set_ylabel("Preview (mm)");axes[1,0].set_ylabel("Feed (mm/s)")
    axes[0,0].legend(fontsize=6)
    fig.suptitle(f"V2 validation traces, lowest available seed {seed}; failed traces included",fontsize=9)
    common.save(fig,"preview_v2_preview_mechanism")
    return True


def tex_report(rows,budgets,protocol,selection,aggregates,paired,audit,has_mechanism,include_test,nnc_summary=None):
    out=["% Generated from saved V2 runs by PPO_project/tools/preview_v2_report.py.",r"\section{Second revision: bounded preview and ordered progress}",r"\label{sec:preview_v2}","",
        "The second revision (V2) uses a new fixed path split. Previously inspected V1 test paths are development data, and three newly generated geometries are reserved for the frozen evaluation. The common objective rewards newly covered ordered arc length, integrates running costs over physical time, and gates completion reward on geometric and ordered completion. Training costs, initialization and endpoint corrections are shared by the fixed and adaptive controls.","",
        f"The declared budget is {protocol['screen_steps']:,} steps for each fixed-distance screening model, {protocol['pretrain_steps']:,} steps per shared fixed-preview initialization, and {protocol['paired_finetune_steps']:,} steps for each paired final configuration. Thus a final method uses {protocol['total_learning_steps_per_final_method_seed']:,} learning steps per seed including its inherited pretraining. Residual preview changes a bounded distance around the selected fixed value. Absolute preview and adaptation-off controls share the same initialization.","",
        r"\subsection{Validation screening and observed failure}"]
    screen_rows=[]
    for row in rows:
        if row["stage"]=="screen" and row["evaluation"]=="evaluation_validation_nominal":
            distance=next(r["distance_mm"] for r in selection["ranked"] if r["variant"]==row["variant"])
            screen_rows.append([f"{distance:g}",common.FAMILY_LABEL[family(row["path_id"])],str(row["success"]),
                f"{row['ordered_progress']:.3f}",f"{row['contour_error_max_mm']:.3f}",common.escaped(row["termination_reason"]),
                f"{row['elapsed_time_s']:.3f}" if row["success"] else "--"])
    out+=common.table("V2 screening on validation paths. Each distance is independently trained using the screening seed. Failed times are omitted from efficiency comparisons; all path outcomes remain. Successful-time cells may describe different successful path sets and are not paired method-effect estimates.",
        "tab:preview_v2_screen",["Distance/mm","Validation family","Success","Progress","Peak error/mm","Termination","Success time/s"],screen_rows)
    selected=next(item for item in selection["ranked"] if item["distance_mm"]==selection["fixed_distance_mm"])
    out += [f"Validation screening selected a fixed distance of ${selection['fixed_distance_mm']:g}$ mm, with {selected['successes']} successful paths. Selection does not imply viability on all validation geometries. All candidate failures remain in the record, and no test-path-specific distance is selected.",""]
    out+=common.figure("preview_v2_validation_screen","All fixed-distance screening trajectories and feed histories on the validation set. Endpoint markers distinguish successful arrival from failure. Trajectories end at their actual termination; failure time is not treated as an efficiency result.","fig:preview_v2_screen")
    if audit:
        last=audit["final"]
        out += [f"The long-preview model failed on the first straight--curve transition at ${last['timestamp_s']:.3f}$ s and feed ${last['v_mm_s']:.3f}$ mm/s. Its final heading exceeded its LOS reference by ${audit['final_heading_minus_reference_rad']:.3f}$ rad. The guided angular-velocity intent had become negative while the constrained output continued positive rotation. This supports a coupling between anticipatory LOS guidance and finite turning-brake response. Because distances were trained independently, the observation does not isolate a causal distance effect. It motivates same-state geometric diagnostics using multi-scale chord deviation, signed heading error and angular stopping margin, without conditioning distance on path identity.",""]
        out+=common.figure("preview_v2_curve_diagnosis","Saved validation trace of the selected long-distance screening model. (a) Early line--curve transition. (b) Heading and LOS reference. (c) Guided intent and constrained angular velocity. (d) Error and one-sided tolerance. This is a descriptive failed-case diagnostic; it is not a counterfactual experiment or a test-set result.","fig:preview_v2_curve_probe")
    out += [r"\subsection{Recorded repeated training and paired evaluation}"]
    cost_rows=[]
    for stage in sorted({r["stage"] for r in budgets}):
        subset=[r for r in budgets if r["stage"]==stage]
        cost_rows.append([common.escaped(stage),len(subset),len({r["seed"] for r in subset}),sum(r["steps"] for r in subset),
            sum(r.get("pretraining_steps",0) for r in subset),f"{sum(r['training_wall_s'] for r in subset):.1f}"])
    out+=common.table("Recorded V2 training cost at report generation. Inherited pretraining is shown for each fine-tuned method; its execution is counted once in the separate pretraining stage, not executed again for every inherited checkpoint. Partial stages remain partial.",
        "tab:preview_v2_cost",["Stage","Completed runs","Distinct seeds","Executed steps","Inherited steps","Train wall/s"],cost_rows)
    for stage in ["pretrain","finetune"]:
        stage_rows=[]
        for item in aggregates:
            if item["stage"]!=stage:
                continue
            group=[r for r in rows if (r["stage"],r["evaluation"],r["variant"],r["path_id"])==(stage,item["evaluation"],item["variant"],item["path_id"])]
            stage_rows.append([LABELS.get(item["variant"],common.escaped(item["variant"])),common.escaped(item["evaluation"].removeprefix("evaluation_")),
                common.FAMILY_LABEL.get(family(item["path_id"]),common.escaped(item["path_id"])),item["n"],f"{item['successes']}/{item['n']}",
                common.stat_text([r["ordered_progress"] for r in group]),common.stat_text([r["contour_error_max_mm"] for r in group])])
        if stage_rows:
            out+=common.table(f"V2 {stage} outcomes available at report generation. Means and sample SD are across independent seeds within a path. Every failure enters progress and observed-error summaries; low error on an incomplete trace is not completion accuracy.",
                f"tab:preview_v2_{stage}",["Method","Evaluation","Family","n","Success","Progress","Peak error/mm"],stage_rows)
        else:
            out += [f"The {stage} evaluation stage has no completed saved records in this report; no results are imputed.",""]
    for comparison in paired:
        table_rows=[]
        for method in comparison["pairwise"]:
            for item in method["by_path"]:
                mean,sd=item["fixed_minus_adaptive_time_mean_s"],item["fixed_minus_adaptive_time_sd_s"]
                value="--" if mean is None else (f"${mean:.4f}\\pm{sd:.4f}$" if sd is not None else f"${mean:.4f}$")
                table_rows.append([LABELS[method["adaptive"]],common.escaped(item["path_id"]),item["n_common"],value])
        if table_rows:
            out+=common.table("Paired successful-time differences (fixed minus adaptive) for "+common.escaped(comparison["evaluation"])+". Only common successful seed/path pairs enter. Positive differences favor adaptive timing on that conditional set; full success and failure counts must also be considered. No empty-set statistic is filled with failed stopping times.",
                "tab:preview_v2_paired_"+comparison["evaluation"].replace("evaluation_",""),["Adaptive method","Path","Common pairs","Time difference/s"],table_rows)
    if has_mechanism:
        out+=common.figure("preview_v2_preview_mechanism","Validation preview distance and feed for the lowest available seed, showing all available final methods and failures. Preview variation is a mechanism display, not independent evidence of superiority.","fig:preview_v2_mechanism")
    out+=nnc_validation_tex(nnc_summary)
    out += ["Only previously saved evaluations were read to generate this report. "+("Frozen test records are included under the explicit reporting gate." if include_test else "No V2 test result was read or evaluated by this report."),"",
        "The complete source table retains termination reasons, all constraint flags, terminal feed, and runtime information. Timing remains conditional on successful completion and cannot establish superiority when successful samples are too few or failures differ. Cartesian motion and low-speed heading observability retain the limitations described in the implementation audit.",""]
    return "\n".join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS)
    parser.add_argument("--include-frozen-test",action="store_true")
    args=parser.parse_args()
    common.style()
    output=args.runs_dir/"report"
    output.mkdir(parents=True,exist_ok=True)
    protocol=common.read_json(args.runs_dir/"protocol.json")
    selection=common.read_json(args.runs_dir/"selection.json")
    rows,budgets=discover(args.runs_dir,args.include_frozen_test)
    aggregates=aggregate(rows)
    nnc_summary=nnc_validation_summary(rows,budgets,args.runs_dir)
    if nnc_summary:
        write_json(output/"nnc_validation_summary.json",nnc_summary)
    pairs=[paired_summary(rows,evaluation) for evaluation in sorted({r["evaluation"] for r in rows if r["stage"]=="finetune"})]
    screen_figure(rows)
    audit=curve_diagnosis(rows,output)
    mechanism=preview_mechanism(rows)
    text=tex_report(rows,budgets,protocol,selection,aggregates,pairs,audit,mechanism,args.include_frozen_test,nnc_summary)
    (ROOT/"论文项目/generated/preview_v2_results.tex").write_text(text,encoding="utf-8")
    write_json(output/"summary.json",{"include_frozen_test":args.include_frozen_test,"rows":rows,"budgets":budgets,"aggregates":aggregates,"paired":pairs,"nnc_validation":nnc_summary,
        "source_hashes":SOURCES,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statistical_unit":"independent seed within path; paired time comparisons require both methods successful on the same seed/path",
        "figure_contract":"Python vector geometry/trajectory and synchronized heading diagnostics; only available validation traces, all failures retained; existing manuscript style"})
    fields=["stage","evaluation","variant","seed","path_id","success","path_completed","ordered_progress","contour_error_max_mm","contour_error_mean_mm","contour_error_rmse_mm","elapsed_time_s","terminal_feed_mm_s","termination_reason","failure_reasons","source"]
    with (output/"rows.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    for row in rows:
        print(row["stage"],row["variant"],row["path_id"],"success="+str(row["success"]),"progress="+format(row["ordered_progress"],".3f"),row["termination_reason"])
    print("Report:",output,"; test included:",args.include_frozen_test)


if __name__=="__main__":
    main()
