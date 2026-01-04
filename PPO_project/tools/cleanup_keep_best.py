from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return (PROJECT_ROOT.parent / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _read_table(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if isinstance(row, dict)]


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isfinite(v):
        return v
    return None


def _safe_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _bundle_tag(bundle_path: str) -> Optional[str]:
    parts = Path(bundle_path).parts
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return parts[0] if parts else None


def _read_baseline_ref(baseline_dir: Path) -> Optional[Path]:
    ref_path = baseline_dir / "BASELINE_REF.txt"
    if not ref_path.exists():
        return None
    text = ref_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return _resolve(text)


def _f_max(row: Dict[str, str], key: str, *, default: float = -1e12) -> float:
    val = _safe_float(row.get(key))
    return val if val is not None else default


def _f_min(row: Dict[str, str], key: str, *, default: float = -1e12) -> float:
    val = _safe_float(row.get(key))
    return -val if val is not None else default


def _score_row(row: Dict[str, str], phase: str) -> tuple:
    verdict_pass = 1.0 if _safe_bool(row.get("verdict_pass")) else 0.0
    eval_pass = 1.0 if _safe_bool(row.get("eval_pass")) else 0.0
    base = (
        verdict_pass,
        eval_pass,
        _f_max(row, "success_rate"),
        _f_min(row, "stall_rate"),
        _f_max(row, "mean_progress_final"),
        _f_min(row, "max_abs_contour_error"),
    )

    phase_lower = phase.lower()
    if phase_lower == "b2a":
        extra = (
            _f_min(row, "trace_corner_peak_abs_omega"),
            _f_min(row, "trace_corner_mean_abs_domega"),
            _f_max(row, "trace_corner_min_velocity"),
            _f_max(row, "trace_corner_v_drop_ratio"),
        )
    elif phase_lower == "b2b":
        extra = (
            _f_min(row, "trace_recovery_mean_abs_error"),
            _f_min(row, "trace_recovery_outside_rate"),
        )
    elif phase_lower == "b2c":
        extra = (
            _f_max(row, "trace_mean_velocity"),
            _f_min(row, "mean_steps"),
        )
    else:
        extra = ()

    return base + extra


def _select_best(rows: List[Dict[str, str]], phase: str) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    return max(rows, key=lambda r: _score_row(r, phase))


def _find_bundle_dirs(artifacts_root: Path) -> List[Path]:
    return [p.parent.resolve() for p in artifacts_root.rglob("manifest.json")]


def _delete_dir(path: Path, *, apply: bool) -> None:
    if not apply:
        print(f"[dry-run] delete {path}")
        return
    shutil.rmtree(path, ignore_errors=True)
    print(f"[deleted] {path}")


def _remove_empty_dirs(root: Path, keep: Iterable[Path]) -> None:
    keep_set = {p.resolve() for p in keep}
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.is_dir():
            continue
        if path.resolve() in keep_set:
            continue
        try:
            if not any(path.iterdir()):
                path.rmdir()
        except Exception:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep only baseline + aggregation + best bundles per phase.")
    parser.add_argument("--artifacts_root", default="artifacts", help="Artifacts root (default: artifacts)")
    parser.add_argument(
        "--aggregation",
        default="artifacts/aggregation/main_table.csv",
        help="Aggregation CSV path (default: artifacts/aggregation/main_table.csv)",
    )
    parser.add_argument(
        "--phases",
        default="B2a,B2b,B2c",
        help="Comma-separated phases to keep (default: B2a,B2b,B2c)",
    )
    parser.add_argument("--baseline_dir", default="P0_L2", help="Baseline bundle dir name (default: P0_L2)")
    parser.add_argument("--apply", action="store_true", help="Delete files when set; otherwise dry-run")
    args = parser.parse_args()

    artifacts_root = _resolve(args.artifacts_root)
    aggregation_path = _resolve(args.aggregation)
    phases = [p.strip() for p in str(args.phases).split(",") if p.strip()]

    if not artifacts_root.exists():
        print(f"[error] artifacts_root not found: {artifacts_root}")
        return 1

    rows = _read_table(aggregation_path)
    if not rows:
        print(f"[error] aggregation table not found or empty: {aggregation_path}")
        return 1

    keep_bundle_dirs: List[Path] = []
    keep_top_dirs: List[Path] = []

    baseline_dir = artifacts_root / str(args.baseline_dir)
    aggregation_dir = artifacts_root / "aggregation"
    if baseline_dir.exists():
        keep_top_dirs.append(baseline_dir.resolve())
        baseline_ref = _read_baseline_ref(baseline_dir)
        if baseline_ref and baseline_ref.exists():
            keep_bundle_dirs.append(baseline_ref.resolve())
    if aggregation_dir.exists():
        keep_top_dirs.append(aggregation_dir.resolve())

    for phase in phases:
        phase_rows = [r for r in rows if _bundle_tag(r.get("bundle_path", "")) and _bundle_tag(r.get("bundle_path", "")).lower() == phase.lower()]
        best = _select_best(phase_rows, phase)
        if not best:
            print(f"[warn] no bundles found for phase {phase}")
            continue
        bundle_path = best.get("bundle_path")
        if not bundle_path:
            print(f"[warn] phase {phase} best row missing bundle_path")
            continue
        bundle_dir = _resolve(bundle_path)
        if not bundle_dir.exists():
            print(f"[warn] phase {phase} bundle_dir not found: {bundle_dir}")
            continue
        keep_bundle_dirs.append(bundle_dir.resolve())
        print(f"[keep] {phase}: {bundle_dir}")

    keep_dirs = set(keep_top_dirs + keep_bundle_dirs)

    bundle_dirs = _find_bundle_dirs(artifacts_root)
    for bundle_dir in bundle_dirs:
        if baseline_dir.exists() and _is_within(bundle_dir, baseline_dir):
            continue
        if bundle_dir.resolve() in keep_dirs:
            continue
        _delete_dir(bundle_dir, apply=bool(args.apply))

    _remove_empty_dirs(artifacts_root, keep_dirs)
    print("[done] cleanup complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
