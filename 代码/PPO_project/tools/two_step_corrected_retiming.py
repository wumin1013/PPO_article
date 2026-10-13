"""Corrected two-step diagnostic: unchanged fixed geometry, consistent retiming.

The original baseline is not edited. This version replaces its per-step dynamic
speed/endpoint clipping with a sampled quintic arclength law, then verifies the
actual position commands. Angular feasibility is audited, not presumed: a fixed
polyline has tangent jumps that scalar time scaling need not remove.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project"))
from src.algorithms.two_step_baseline import SmoothingParams, smooth_reference_path
from src.utils.revision_metrics import independent_kinematics
from terminal_metrics import evaluate_terminal_trajectory

DEFAULT_OUT = ROOT / "论文项目/revision_20260910_continued/corrected_two_step"
LIMITS = dict(MAX_VEL=100., MAX_ACC=2000., MAX_JERK=62500.,
              MAX_ANG_VEL=2*math.pi, MAX_ANG_ACC=10*math.pi, MAX_ANG_JERK=1000.)
CHANNELS = {"feed_mm_s": "MAX_VEL", "feed_acceleration_mm_s2": "MAX_ACC", "feed_jerk_mm_s3": "MAX_JERK",
            "angular_velocity_rad_s": "MAX_ANG_VEL", "angular_acceleration_rad_s2": "MAX_ANG_ACC",
            "angular_jerk_rad_s3": "MAX_ANG_JERK"}
DT = .001
TOLERANCE = 1.25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def sampled_quintic_commands(path, motion_steps, dt=DT, hold_steps=25):
    """Return commands on exactly the supplied fixed polyline, without clipping.

    s=L q(t/T), q(u)=10u^3-15u^4+6u^5. The last sample reaches L by
    construction; the separate stationary samples are part of the timed command.
    Returned discrete v/a/j are computed from executed positions, not derivatives
    of the continuous law. No endpoint snap or internal-value substitution.
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Need an Nx2 path")
    if int(motion_steps) != motion_steps or motion_steps < 4 or dt <= 0 or hold_steps < 25:
        raise ValueError("Need >=4 motion samples, positive dt and >=25 rest samples")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("Adjacent duplicate/degenerate fixed-path vertices")
    arc = np.r_[0., np.cumsum(lengths)]
    u = np.arange(motion_steps+1, dtype=float)/motion_steps
    # Symmetry avoids cancellation near the endpoint without clipping q or s.
    z = np.minimum(u, 1-u)
    q_half = z**3*(10+z*(-15+6*z))
    q = np.where(u <= .5, q_half, 1-q_half)
    s = arc[-1]*q
    positions = np.c_[np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])]
    positions = np.vstack([positions, np.repeat(points[-1][None, :], hold_steps, axis=0)])
    times = np.arange(len(positions))*dt
    s = np.r_[s, np.full(hold_steps, arc[-1])]
    initial = dict(initial_velocity=[0., 0.], initial_acceleration=[0., 0.],
                   initial_heading=float(np.arctan2(*(points[1]-points[0])[::-1])),
                   initial_angular_velocity=0., initial_angular_acceleration=0.)
    return positions, times, s, initial


def peaks_and_ratios(positions, times, initial, limits):
    values = independent_kinematics(positions, times, **initial)
    peaks = {name: float(np.nanmax(np.abs(values[name]))) for name in CHANNELS}
    ratios = {name: peak/float(limits[CHANNELS[name]]) for name, peak in peaks.items()}
    return values, peaks, ratios


def retime_fixed_path(path, *, limits=None, max_motion_s=59.975):
    """Bounded search of whole-trajectory times; never crop an infeasible step.

    This is a transparent scalar-correctness repair, not a time-optimal or angular
    feasibility solver. All attempted durations and failures remain in the log.
    """
    limits = dict(LIMITS if limits is None else limits)
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    lower = max(1.875*length/limits["MAX_VEL"], math.sqrt(5.773502692*length/limits["MAX_ACC"]),
                (60*length/limits["MAX_JERK"])**(1/3), .004)
    duration = lower*1.02
    attempts, selected = [], None
    scalar_names = list(CHANNELS)[:3]
    for _ in range(28):
        steps = math.ceil(duration/DT)
        if steps*DT > max_motion_s:
            break
        p, t, s, initial = sampled_quintic_commands(path, steps)
        values, peaks, ratios = peaks_and_ratios(p, t, initial, limits)
        scalar_ok = all(ratios[n] <= 1+1e-5 for n in scalar_names)
        angular_ok = all(ratios[n] <= 1+1e-5 for n in list(CHANNELS)[3:])
        attempts.append(dict(motion_s=steps*DT, scalar_satisfied=scalar_ok, angular_satisfied=angular_ok,
                             peaks=peaks, limit_ratios=ratios))
        if scalar_ok:
            selected = (p, t, s, initial, values)
            break
        factor = max(ratios["feed_mm_s"], math.sqrt(ratios["feed_acceleration_mm_s2"]),
                     ratios["feed_jerk_mm_s3"]**(1/3), 1.1)
        duration = steps*DT*max(1.1, min(factor*1.02, 2.0))
    if selected is None:
        return None, attempts
    # A small, preregistered slower-time probe quantifies the angular boundary.
    # It does not silently replace the scalar correction or discard failures.
    selected_duration = attempts[-1]["motion_s"]
    for factor in (2., 4.):
        steps = math.ceil(selected_duration*factor/DT)
        if steps*DT > max_motion_s:
            continue
        p, t, _, initial = sampled_quintic_commands(path, steps)
        _, peaks, ratios = peaks_and_ratios(p, t, initial, limits)
        attempts.append(dict(role="angular_duration_probe", motion_s=steps*DT,
                             scalar_satisfied=all(ratios[n] <= 1+1e-5 for n in scalar_names),
                             angular_satisfied=all(ratios[n] <= 1+1e-5 for n in list(CHANNELS)[3:]),
                             peaks=peaks, limit_ratios=ratios))
    return selected, attempts


def evaluate_case(reference, params, output):
    began = time.perf_counter()
    smoothed = smooth_reference_path(reference, half_epsilon=TOLERANCE, closed=False, params=params)
    fixed = smoothed.smoothed_path
    selected, attempts = retime_fixed_path(fixed)
    output.mkdir(parents=True, exist_ok=False)
    np.savetxt(output/"reference.csv", reference, delimiter=",", header="x_mm,y_mm", comments="")
    np.savetxt(output/"fixed_smoothed_path.csv", fixed, delimiter=",", header="x_mm,y_mm", comments="")
    result = dict(method="corrected two-step: quintic fixed-polyline retiming", smoothing=asdict(params),
                  actual_transition_ratio=smoothed.transition_ratio, actual_safety_factor=smoothed.safety_factor,
                  fixed_geometry_error_mm=smoothed.max_smoothed_path_error_mm,
                  attempts=attempts, plan_found=selected is not None,
                  guarantees="No angular feasibility or axis-motion guarantee is assumed; independent audit is authoritative.")
    if selected is not None:
        positions, times, arc, initial, kin = selected
        audit = evaluate_terminal_trajectory(reference, positions, times, tolerance_mm=TOLERANCE,
            limits=LIMITS, witness_spacing_mm=.01, **initial)
        np.savez_compressed(output/"command.npz", positions=positions, timestamps=times, arclength=arc,
                            **{name: kin[name] for name in CHANNELS})
        columns = np.column_stack([times, positions, arc, *[kin[name] for name in CHANNELS]])
        np.savetxt(output/"command.csv", columns, delimiter=",",
            header="timestamp_s,x_mm,y_mm,s_mm,"+",".join(CHANNELS), comments="")
        result.update(audit=audit, command_sha256=sha(output/"command.npz"), initial=initial,
                      scalar_motion_satisfied=all(audit["constraints"][n]["satisfied"] for n in list(CHANNELS)[:3]),
                      full_success=audit["success"])
    result["planning_and_audit_wall_s"] = time.perf_counter()-began
    result["artifacts"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output/"result.json", result)
    return result


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Retain all corrected-baseline experiments; use a new --output")
    candidates = [dict(name=f"ratio{ratio:g}_angle{angle:g}", transition_ratio=ratio, angle_threshold_deg=angle)
                  for ratio in (.25, .45) for angle in (8., 1.)]
    source_files = [Path(__file__), ROOT/"PPO_project/src/algorithms/two_step_baseline.py",
                    ROOT/"PPO_project/src/utils/revision_metrics.py", ROOT/"PPO_project/tools/terminal_metrics.py"]
    source = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}
    protocol = dict(version="corrected-two-step-quintic-v1", constraints=LIMITS, dt_s=DT,
        one_sided_tolerance_mm=TOLERANCE, max_elapsed_s=60., rest_hold_steps=25,
        initial_and_final_state="Fixed path endpoints; zero initial command velocity/acceleration; final 20 ms stationary audit",
        candidates=candidates, source_sha256=source,
        geometry="Original smooth_reference_path output retained exactly for each unified parameter candidate",
        retiming="Global sampled quintic s(t); whole-trajectory duration search; no per-step speed or position clip",
        angular_probe="Same fixed polyline retimed at 2x and 4x duration; feasibility diagnostic only",
        parameter_search_scope="All four existing smoothing parameter combinations on all three validation paths; no test use",
        selection="Prefer full-success coverage; if none, report scalar-only feasibility separately and do not select a full-constraint winner",
        no_training=True, machine_execution_authorized=False)
    write(output/"protocol.json", protocol)
    # Gate the case study on a complete straight-line correctness check.
    straight = evaluate_case(np.array([[0., 0.], [16., 0.]]), SmoothingParams(), output/"straight_probe")
    assert straight["audit"]["success"], "Straight-line scalar/rest correctness failed"
    validation_root = ROOT/"论文项目/revision_20260909_event_preview/study/two_step_event_v1/validation_nominal"
    source_protocol = json.loads((validation_root/"protocol.json").read_text(encoding="utf-8"))
    assert source_protocol["dt_s"] == DT and source_protocol["one_sided_tolerance_mm"] == TOLERANCE
    refs = {}
    for family in ("mixed_lengths", "dense_corners", "line_curve"):
        path_id = family+"_validation"
        old_record = json.loads((validation_root/path_id/"result.json").read_text(encoding="utf-8"))
        reference_file = validation_root/path_id/"reference.csv"
        assert sha(reference_file) == old_record["artifact_sha256"]["reference.csv"]
        refs[path_id] = (reference_file, np.loadtxt(reference_file, delimiter=",", skiprows=1))
    records = []
    for candidate in candidates:
        params = SmoothingParams(transition_ratio=candidate["transition_ratio"], angle_threshold_deg=candidate["angle_threshold_deg"])
        for path_id, (_, reference) in refs.items():
            result = evaluate_case(reference, params, output/candidate["name"]/path_id)
            records.append(dict(candidate=candidate["name"], path_id=path_id, **result))
            print(candidate["name"], path_id, result.get("full_success"), result.get("audit", {}).get("failure_reasons"), flush=True)
    assert source == {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}
    groups = []
    for candidate in candidates:
        cases = [r for r in records if r["candidate"] == candidate["name"]]
        groups.append(dict(candidate=candidate["name"], full_successes=sum(bool(r.get("full_success")) for r in cases),
            scalar_successes=sum(bool(r.get("scalar_motion_satisfied")) for r in cases),
            geometry_successes=sum(bool(r.get("audit", {}).get("geometry_feasible")) for r in cases),
            stationary_successes=sum(bool(r.get("audit", {}).get("full_rest_satisfied")) for r in cases),
            all_elapsed_sum_s=sum(r.get("audit", {}).get("elapsed_time_s", 0.) for r in cases),
            time_scope="All attempted scalar-retimed cases; failures are not success times"))
    summary = dict(protocol_sha256=sha(output/"protocol.json"), source_sha256=source,
        reference_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p, _ in refs.values()},
        straight_probe=straight, rows=records, groups=groups,
        total_planning_and_audit_wall_s=straight["planning_and_audit_wall_s"]+sum(r["planning_and_audit_wall_s"] for r in records),
        total_duration_candidates=len(straight["attempts"])+sum(len(r["attempts"]) for r in records),
        total_new_training_steps=0,
        scope="One repaired timing family on original fixed smoothing geometry. Feasibility boundary only until all full constraints pass; no PPO-superiority claim.")
    write(output/"summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().output)
