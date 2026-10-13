"""Export registered path geometry tables only; no figures or model execution."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "论文项目"
DATA = ROOT / "PPO_project/data/preview_event_v1"
OUT = PAPER / "revision_20260909_event_preview/path_geometry"
FAMILIES = ("mixed_lengths", "dense_corners", "line_curve")
SPLITS = ("train", "validation", "test")
FAMILY_LABEL = dict(mixed_lengths="Mixed lengths", dense_corners="Dense corners", line_curve="Lines/curves")
SPLIT_LABEL = dict(train="Train", validation="Validation", test="New holdout")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def measure(entry, points):
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Reference must be finite Nx2 points")
    delta = np.diff(points, axis=0); lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("Repeated adjacent points are invalid")
    directions = delta / lengths[:, None]
    turns = np.degrees(np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1)))
    quantiles = np.quantile(lengths, [0, .25, .5, .75, 1])
    stats = entry["statistics"]
    if (stats["point_count"] != len(points) or stats["segment_count"] != len(lengths)
        or not np.isclose(stats["path_length_mm"], lengths.sum(), rtol=0, atol=1e-7)
        or not np.allclose(stats["segment_lengths_mm"], lengths, rtol=0, atol=1e-8)
        or not np.allclose(stats["all_turn_angles_deg"], turns, rtol=0, atol=1e-7)):
        raise ValueError("Saved manifest statistics disagree with reference points: " + entry["id"])
    has_bezier = any("cubic" in p for p in entry["parameters"].get("primitives", []))
    source_bound = float(entry["source"]["analytic_to_polyline_deviation_bound_mm"])
    if source_bound != (.005 if has_bezier else 0.):
        raise ValueError("Unexpected discretization contract: " + entry["id"])
    source_geometry = "lines_and_cubic_bezier" if has_bezier else "defining_polyline"
    row = dict(path_id=entry["id"], family=entry["family"], split=entry["split"], source_kind="synthetic",
               coordinate_unit="mm", plane="XY", source_geometry=source_geometry,
               reference_point_count=len(points), reference_segment_count=len(lengths),
               path_length_mm=float(lengths.sum()),
               segment_min_mm=float(quantiles[0]), segment_q25_mm=float(quantiles[1]),
               segment_median_mm=float(quantiles[2]), segment_q75_mm=float(quantiles[3]), segment_max_mm=float(quantiles[4]),
               maximum_turn_deg=float(turns.max()), major_corner_threshold_deg=stats["major_corners_threshold_deg"],
               major_corner_count=int(np.count_nonzero(turns >= stats["major_corners_threshold_deg"])),
               closed=bool(np.linalg.norm(points[0]-points[-1]) < 1e-9),
               start_x_mm=float(points[0, 0]), start_y_mm=float(points[0, 1]),
               end_x_mm=float(points[-1, 0]), end_y_mm=float(points[-1, 1]),
               xmin_mm=float(points[:, 0].min()), xmax_mm=float(points[:, 0].max()),
               ymin_mm=float(points[:, 1].min()), ymax_mm=float(points[:, 1].max()),
               source_discretization_bound_mm=source_bound,
               discretization_scope="Bezier subcurve-to-chord bound; straight primitives exact" if has_bezier else "Exact defining polyline",
               max_spacing_parameter_mm=entry["parameters"].get("max_spacing_mm"),
               reference_file=entry["reference_file"], reference_sha256=entry["reference_sha256"])
    arc = np.r_[0., np.cumsum(lengths)]
    segment_rows = [dict(path_id=entry["id"], split=entry["split"], segment_index=i, start_arc_mm=float(arc[i]),
                         end_arc_mm=float(arc[i+1]), length_mm=float(length),
                         start_x_mm=float(points[i, 0]), start_y_mm=float(points[i, 1]),
                         end_x_mm=float(points[i+1, 0]), end_y_mm=float(points[i+1, 1]),
                         turn_at_end_deg=float(turns[i]) if i < len(turns) else None)
                    for i, length in enumerate(lengths)]
    return row, segment_rows


def table(rows):
    lines = ["% Data table only. Generated from registered reference CSVs; no figure is inserted.",
             r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Geometry of the nine synthetic planar paths registered for the event study. $S$ is the reference polyline length and $N$ its linear segment count. Segment lengths are reported as minimum/median/maximum; $\theta_{\max}$ is the largest angle between adjacent reference segments. $\delta_{\mathrm{disc}}$ is the declared source discretization bound: $0.005$ mm applies to cubic B\'ezier subcurves, whereas defining polylines have zero approximation error. This source bound is separate from the full planning tolerance width and its half-width one-sided deviation. These paths are synthetic, with no industrial CAM provenance.}",
             r"\label{tab:event_path_geometry}", r"\begin{tabular}{llrrlrr}", r"\toprule",
             r"Family & Split & $S$ (mm) & $N$ & Segment length (mm) & $\theta_{\max}$ ($^\circ$) & $\delta_{\mathrm{disc}}$ (mm) \\",
             r"\midrule"]
    for index, row in enumerate(rows):
        if index and index % 3 == 0:
            lines.append(r"\addlinespace")
        lengths = "/".join(f"{row[key]:.3f}" for key in ("segment_min_mm", "segment_median_mm", "segment_max_mm"))
        lines.append(f"{FAMILY_LABEL[row['family']]} & {SPLIT_LABEL[row['split']]} & {row['path_length_mm']:.2f} & {row['reference_segment_count']} & {lengths} & {row['maximum_turn_deg']:.1f} & {row['source_discretization_bound_mm']:.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main():
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol_path = PAPER / "revision_20260909_event_preview/study/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["manifest_sha256"] != sha(manifest_path):
        raise ValueError("Registered manifest changed")
    selected = [p for p in manifest["paths"] if p["split"] in SPLITS]
    expected = {(f, s) for f in FAMILIES for s in SPLITS}
    if len(selected) != 9 or {(p["family"], p["split"]) for p in selected} != expected:
        raise ValueError("Require one registered path per family and split")
    selected.sort(key=lambda p: (FAMILIES.index(p["family"]), SPLITS.index(p["split"])))
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in (manifest_path, protocol_path)}
    rows, segments = [], []
    for entry in selected:
        if entry["source"]["kind"] != "synthetic" or entry["plane"] != "XY" or entry["coordinate_unit"] != "mm":
            raise ValueError("This table is restricted to the registered synthetic XY paths")
        path = (DATA / entry["reference_file"]).resolve()
        if not path.is_relative_to(DATA.resolve()) or sha(path) != entry["reference_sha256"]:
            raise ValueError("Reference path or hash differs")
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        row, segment_rows = measure(entry, np.loadtxt(path, delimiter=",", skiprows=1))
        rows.append(row); segments.extend(segment_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "geometry_summary.csv", rows)
    write_csv(OUT / "reference_segments.csv", segments)
    write_json(OUT / "path_parameters.json", selected)
    tex_path = PAPER / "generated/preview_event_paths.tex"
    tex_path.write_text(table(rows), encoding="utf-8")
    products = [OUT / "geometry_summary.csv", OUT / "reference_segments.csv", OUT / "path_parameters.json", tex_path]
    write_json(OUT / "source_hashes.json", dict(
        command="python " + " ".join(sys.argv), script_sha256=sha(__file__), source_sha256=sources,
        output_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in products},
        displayed_path_count=len(rows), reference_segment_count=len(segments),
        excluded_development=[dict(id=p["id"], split=p["split"], previous_split=p.get("previous_split"),
                                   reference_sha256=p["reference_sha256"]) for p in manifest["paths"] if p["split"] == "development"],
        scope="Registered geometry description only; no performance evaluation, model execution, or parameter selection",
        figures_generated=False, data_modified=False,
        discretization="0.005 mm is the declared Bezier subcurve/chord bound; defining polylines have zero approximation error; separate from motion tolerance"))
    print(json.dumps(dict(paths=len(rows), reference_segments=len(segments), figures=0, output=str(OUT), tex=str(tex_path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
