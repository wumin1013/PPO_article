"""Geometry-only reservation tests; no policy/checkpoint evaluation."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import generate_preview_frontier_holdouts as module


@pytest.fixture(scope="module")
def reserved(tmp_path_factory):
    target = tmp_path_factory.mktemp("frontier-reservation") / "new-data"
    summary = module.generate(target)
    return target, summary


def test_fixed_seed_gives_twelve_independent_reproducible_definitions():
    first = module.definitions()
    assert first == module.definitions()
    assert len(first) == len({item["id"] for item in first}) == 12
    assert Counter(item["family"] for item in first) == dict.fromkeys(module.FAMILIES, 4)
    assert len({tuple(item["parameters"]["spawn_key"]) for item in first}) == 12
    assert all(item["split"] == "test" for item in first)
    with pytest.raises(ValueError, match="fixed seed"):
        module.definitions(module.SEED + 1)


def test_generation_and_readonly_regeneration_preserve_all_hashes(reserved):
    target, summary = reserved
    before = {path: module.sha(path) for path in target.rglob("*") if path.is_file()}
    assert module.verify(target) == summary
    assert before == {path: module.sha(path) for path in before}
    assert summary["path_count"] == 12
    assert not summary["policy_evaluation_performed"]
    assert not summary["model_artifact_access_performed"]


def test_saved_geometry_is_valid_distinct_and_uses_separate_source_tolerance(reserved):
    target, _ = reserved
    manifest = module.read_json(target / "manifest.json")
    assert manifest["tolerance_definition"] == module.TOLERANCE
    assert len(manifest["source"]["source_files"]) == 3
    for item in manifest["paths"]:
        points = module.geometry.load_path(item["id"], target)
        assert np.isfinite(points).all()
        assert np.all(np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10)
        checks = item["geometry_checks"]
        assert checks["open_path"] and checks["x_nondecreasing"]
        assert checks["nonadjacent_intersection_pairs"] == []
        comparison = checks["shape_comparison"]
        assert comparison["minimum_max_point_distance"] > comparison["near_copy_rejection_threshold"]
        assert item["source"]["cam_export"] is False
        bound = item["source"]["analytic_to_polyline_deviation_bound_mm"]
        assert bound == (.005 if item["family"] == "line_curve" else 0.)


@pytest.mark.parametrize("reflection,reverse", [(1, False), (-1, False), (1, True)])
def test_shape_duplicate_detection_rejects_transformed_reused_geometry(reflection, reverse):
    definition = module.definitions()[0]
    points = module.serialized(module.geometry.materialize(definition))
    transformed = points[::-1] if reverse else points.copy()
    transformed = transformed * [3., 3. * reflection] + [20., -11.]
    with pytest.raises(ValueError, match="duplicates"):
        module.check_geometry(points, definition, [("transformed_old_path", transformed)])


def test_existing_output_is_never_overwritten(reserved, tmp_path):
    target, _ = reserved
    digest = module.sha(target / "manifest.json")
    with pytest.raises(FileExistsError, match="Preserve"):
        module.generate(target)
    assert module.sha(target / "manifest.json") == digest
    empty = tmp_path / "existing-empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        module.generate(empty)


@pytest.mark.parametrize("artifact", ["manifest", "csv", "source", "prior"])
def test_verifier_rejects_changed_frozen_inputs(reserved, monkeypatch, artifact):
    target, _ = reserved
    original_sha = module.sha
    reservation = module.read_json(target / "reservation.json")
    if artifact == "manifest":
        changed = target / "manifest.json"
    elif artifact == "csv":
        changed = target / next(name for name in reservation["files"] if name.endswith(".csv"))
    elif artifact == "source":
        changed = Path(module.__file__)
    else:
        changed = module.ROOT / next(iter(reservation["prior_geometry_sha256"]))
    monkeypatch.setattr(module, "sha", lambda path: "changed" if Path(path) == changed else original_sha(path))
    with pytest.raises(ValueError, match="changed"):
        module.verify(target)


def test_invalid_path_is_rejected_without_sampling_replacement():
    definition = module.definitions()[0]
    with pytest.raises(ValueError, match="Repeated|repeated"):
        module.check_geometry(np.array([[0., 0.], [0., 0.], [50., 0.]]), definition, [])
