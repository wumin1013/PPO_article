"""Reserve twelve seeded synthetic holdouts using geometry-only checks.

No policy, checkpoint, reward or candidate result is imported or inspected.
Generation is one fixed draw, not a search for geometries a planner can pass.
The output directory is immutable; ``verify`` is read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import generate_preview_continuation_paths as geometry

ROOT, PROJECT = geometry.ROOT, geometry.PROJECT
TARGET = PROJECT / "data/preview_frontier_holdout_v1"
SEED = 2026091001
FAMILIES = ("mixed_lengths", "dense_corners", "line_curve")
PRIOR_DATASETS = tuple(PROJECT / "data" / name for name in
                       ("revision_20260908", "preview_event_v1", "preview_continuation_v1"))
SOURCE_BOUND_MM = .005
SERIALIZATION_MARGIN_MM = 1e-8
TOLERANCE = dict(planning_bandwidth_mm=2.5, one_sided_tolerance_mm=1.25,
                 source_discretization_bound_mm=SOURCE_BOUND_MM,
                 source_bound_added_to_planning_tolerance=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _polyline(headings, lengths):
    radians = np.radians(headings)
    delta = np.column_stack((np.cos(radians), np.sin(radians))) * np.asarray(lengths)[:, None]
    return np.vstack(([0., 0.], np.cumsum(delta, axis=0))).tolist()


def definitions(seed=SEED):
    """Twelve independent child streams, with fixed counts and parameter ranges."""
    if seed != SEED:
        raise ValueError("This registered version has one fixed seed; create another version for another seed")
    result = []
    children = np.random.SeedSequence(seed).spawn(12)
    for index, child in enumerate(children):
        family, case = FAMILIES[index // 4], index % 4
        rng = np.random.default_rng(child)
        parameters = dict(seed=seed, rng="numpy.random.default_rng/PCG64",
                          spawn_key=list(child.spawn_key), max_spacing_mm=.6)
        if family == "mixed_lengths":
            blocks = (5, 6, 6, 7)[case]
            headings, lengths = [], []
            # Each long traverse has two independently drawn short connectors.
            # Positive x motion avoids self-intersections without rejection sampling.
            for block in range(blocks):
                headings.append(0. if block == 0 else float(rng.uniform(-28., 28.)))
                lengths.append(float(rng.uniform(7., 10.5)))
                sign = float(rng.choice([-1., 1.]))
                for direction in (sign, -sign):
                    headings.append(direction * float(rng.uniform(35., 78.)))
                    lengths.append(float(rng.uniform(.65, 1.5)))
            parameters.update(long_block_count=blocks, long_length_range_mm=[7., 10.5],
                              short_length_range_mm=[.65, 1.5],
                              long_heading_range_deg=[-28., 28.],
                              short_absolute_heading_range_deg=[35., 78.],
                              heading_deg=headings, defining_segment_lengths_mm=lengths,
                              vertices_mm=_polyline(headings, lengths))
            challenge = "Unequal long traverses and short reentrant connectors with independently drawn headings and lengths."
        elif family == "dense_corners":
            count = (48, 56, 60, 64)[case]
            signs = np.ones(count)
            signs[0] = rng.choice([-1., 1.])
            for turn in range(1, count):
                signs[turn] = signs[turn - 1] * (-1. if rng.random() < .8 else 1.)
            headings = signs * rng.uniform(18., 78., count)
            lengths = rng.uniform(.65, 1.55, count)
            parameters.update(defining_segment_count=count, sign_reversal_probability=.8,
                              absolute_heading_range_deg=[18., 78.],
                              length_range_mm=[.65, 1.55], heading_deg=headings.tolist(),
                              defining_segment_lengths_mm=lengths.tolist(),
                              vertices_mm=_polyline(headings, lengths))
            challenge = "Dense unequal short blocks, mostly alternating turns plus short same-sign turn groups, without a repeated motif."
        else:
            count = (4, 4, 5, 5)[case]
            position = np.zeros(2)
            primitives = []
            sign = float(rng.choice([-1., 1.]))
            for turn in range(count):
                position = position + [float(rng.uniform(3., 6.5)), 0.]
                primitives.append(dict(line_to=position.tolist()))
                width = float(rng.uniform(6., 8.5))
                rise = sign * float(rng.uniform(3., 7.))
                start = position.copy()
                position = start + [width, rise]
                primitives.append(dict(cubic=[
                    (start + [width * float(rng.uniform(.25, .4)), 0.]).tolist(),
                    (position - [width * float(rng.uniform(.25, .4)), 0.]).tolist(),
                    position.tolist()]))
                sign *= -1.
            position = position + [float(rng.uniform(4., 8.)), 0.]
            primitives.append(dict(line_to=position.tolist()))
            parameters.update(start_mm=[0., 0.], primitives=primitives, cubic_count=count,
                              line_length_range_mm=[3., 6.5], final_line_length_range_mm=[4., 8.],
                              cubic_x_span_range_mm=[6., 8.5], cubic_y_span_absolute_range_mm=[3., 7.],
                              tangent_handle_fraction_range=[.25, .4],
                              chord_error_bound_mm=SOURCE_BOUND_MM-SERIALIZATION_MARGIN_MM)
            challenge = "Unequal straight runs joined by four or five tangent-continuous cubic S-transitions with alternating curvature."
        result.append(dict(id=f"{family}_frontier_v1_{case + 1:02d}_test", family=family,
                           split="test", exposure_status="unevaluated_holdout", challenge=challenge,
                           parameters=parameters))
    return result


def serialized(points):
    return np.asarray([[float(format(x, ".12g")), float(format(y, ".12g"))] for x, y in points])


def prior_inputs():
    priors, hashes = [], {}
    for directory in PRIOR_DATASETS:
        manifest_path = directory / "manifest.json"
        manifest = geometry.load_manifest(directory)
        hashes[manifest_path.relative_to(ROOT).as_posix()] = sha(manifest_path)
        for item in manifest["paths"]:
            path = directory / item["reference_file"]
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            priors.append((f"{directory.name}/{item['id']}", geometry.load_path(item["id"], directory)))
    return priors, hashes


def source_state():
    files = [Path(__file__), Path(geometry.__file__), PROJECT / "src/utils/revision_paths.py"]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in files}


def check_geometry(points, definition, priors):
    statistics = geometry.path_statistics(points)
    if statistics["closed"] or not 40. <= statistics["path_length_mm"] <= 100.:
        raise ValueError("Expected an open 40..100 mm path")
    if statistics["segment_length_quantiles_mm"]["max"] > .6 + 1e-9:
        raise ValueError("Reference spacing exceeds the fixed 0.6 mm limit")
    return geometry.geometry_checks(points, definition, priors)


def _summary(target, manifest):
    return dict(dataset=str(target), manifest_sha256=sha(target / "manifest.json"),
                reservation_sha256=sha(target / "reservation.json"),
                generator_sha256=manifest["source"]["generator_sha256"], seed=SEED,
                path_count=len(manifest["paths"]),
                family_counts=dict(Counter(item["family"] for item in manifest["paths"])),
                path_ids=[item["id"] for item in manifest["paths"]],
                geometry_and_regeneration_verified=True,
                policy_evaluation_performed=False, model_artifact_access_performed=False)


def generate(target=TARGET):
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError("Preserve registered geometry; use verify or a new version directory")
    priors, prior_hashes = prior_inputs()
    prepared = []
    for definition in definitions():
        raw = geometry.materialize(definition)
        points = serialized(raw)
        error = float(np.linalg.norm(points - raw, axis=1).max())
        if error > SERIALIZATION_MARGIN_MM:
            raise ValueError("Decimal serialization exceeded the reserved source-error margin")
        checks = check_geometry(points, definition, priors)
        checks["serialization_max_perturbation_mm"] = error
        prepared.append((definition, raw, checks))
        priors.append((definition["id"], points))
    # No geometry is redrawn or dropped when a geometry check fails.
    created = datetime.now(timezone.utc).isoformat()
    source_files = source_state()
    source = dict(kind="synthetic", cam_export=False,
                  generator=Path(__file__).relative_to(ROOT).as_posix(),
                  generator_sha256=sha(__file__), source_files=source_files,
                  seed=SEED, rng="numpy.random.default_rng/PCG64",
                  construction="One SeedSequence child stream per registered path; one draw per parameter, no result-informed rejection or resampling")
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    for definition, points, checks in prepared:
        definition.update(created_utc=created, geometry_checks=checks,
                          evaluation_gate="Freeze final methods, common settings, selected distances, seeds, checkpoints and evaluator before any policy evaluation",
                          model_output_inspected_during_design=False)
        item_source = dict(source,
                           analytic_to_polyline_deviation_bound_mm=SOURCE_BOUND_MM if definition["family"] == "line_curve" else 0.,
                           coordinate_serialization_significant_digits=12,
                           discretization_definition="Reference polylines define straight geometry; cubic convex-hull subdivision reserves 1e-8 mm for decimal serialization inside the 0.005 mm source bound. Source error is separate from motion tolerance.")
        entries.append(geometry._save_path(target, definition, points, item_source))
    manifest = dict(schema_version="preview-frontier-holdout-v1", created_utc=created,
                    split_frozen_date=created[:10], source=source, paths=entries,
                    units=dict(distance="mm", time="s", angle="rad"), plane="XY",
                    tolerance_definition=TOLERANCE,
                    dataset_role="Twelve new synthetic holdouts reserved for the next frozen comparison; no training or model selection",
                    exposure_status="unevaluated_holdout", evaluation_count=0,
                    test_usage="Coordinate access here is geometry-only reservation. Candidate development must not load these paths. Test-informed redesign requires another holdout.",
                    design_budget=dict(seed=SEED, families=3, test_instances_per_family=4,
                                       path_length_range_mm=[40., 100.], maximum_interpolation_segment_mm=.6,
                                       bezier_source_chord_bound_mm=SOURCE_BOUND_MM, policy_calls=0),
                    prior_geometry_sha256=prior_hashes,
                    policy_evaluation_performed=False, model_artifact_access_performed=False)
    write_new(target / "manifest.json", manifest)
    files = {path.relative_to(target).as_posix(): sha(path) for path in sorted(target.rglob("*")) if path.is_file()}
    write_new(target / "reservation.json", dict(version="geometry-reservation-v1", created_utc=created,
               manifest_sha256=sha(target / "manifest.json"), source_files=source_files,
               files=files, prior_geometry_sha256=prior_hashes,
               policy_evaluation_performed=False, model_artifact_access_performed=False))
    return verify(target)


def verify(target=TARGET):
    target = Path(target).resolve()
    reservation = read_json(target / "reservation.json")
    if sha(target / "manifest.json") != reservation["manifest_sha256"]:
        raise ValueError("Reserved manifest changed")
    if source_state() != reservation["source_files"]:
        raise ValueError("Reserved generation source changed")
    for name, digest in reservation["files"].items():
        if sha(target / name) != digest:
            raise ValueError(f"Reserved geometry artifact changed: {name}")
    manifest = geometry.load_manifest(target)
    expected_definitions = definitions()
    if [item["id"] for item in manifest["paths"]] != [item["id"] for item in expected_definitions]:
        raise ValueError("The twelve registered path IDs or order changed")
    if manifest["tolerance_definition"] != TOLERANCE:
        raise ValueError("Reserved geometric tolerance metadata changed")
    priors, prior_hashes = prior_inputs()
    if prior_hashes != reservation["prior_geometry_sha256"]:
        raise ValueError("Prior geometry changed after reservation")
    for item, definition in zip(manifest["paths"], expected_definitions):
        for key in ("family", "split", "parameters"):
            if item[key] != definition[key]:
                raise ValueError(f"Registered construction changed: {item['id']}")
        points = geometry.load_path(item["id"], target)
        raw = geometry.materialize(definition)
        np.testing.assert_array_equal(points, serialized(raw))
        checks = check_geometry(points, definition, priors)
        checks["serialization_max_perturbation_mm"] = float(np.linalg.norm(points - raw, axis=1).max())
        if checks != item["geometry_checks"]:
            raise ValueError(f"Geometry audit changed: {item['id']}")
        priors.append((item["id"], points))
    return _summary(target, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "verify"))
    parser.add_argument("--output-dir", type=Path, default=TARGET)
    args = parser.parse_args()
    result = generate(args.output_dir) if args.command == "generate" else verify(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
