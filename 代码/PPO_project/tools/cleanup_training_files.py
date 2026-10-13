from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def _is_keep_dir(name: str) -> bool:
    text = str(name).strip().lower()
    keep_tokens = ("p0", "p0_gold", "p0-l2-gold", "p0_l2_gold")
    return any(token in text for token in keep_tokens)


def _iter_targets(workspace_root: Path, project_root: Path) -> Iterable[Path]:
    for base in (workspace_root, project_root):
        yield base / "out"
        yield base / "save_models"
        yield base / "saved_models"


def _remove_dir(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"[dry-run] remove: {path}")
        return
    shutil.rmtree(path, ignore_errors=True)
    print(f"[removed] {path}")


def _cleanup_saved_models(path: Path, *, dry_run: bool) -> None:
    if not path.exists() or not path.is_dir():
        return

    for child in path.iterdir():
        if not child.is_dir():
            continue
        if _is_keep_dir(child.name):
            print(f"[keep] {child}")
            continue
        _remove_dir(child, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean training artifacts and keep only P0 / P0_gold baselines in saved_models."
    )
    parser.add_argument("--workspace_root", type=str, default=None, help="Workspace root. Default: auto infer")
    parser.add_argument("--dry_run", action="store_true", help="Only print actions.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else project_root.parent

    for target in _iter_targets(workspace_root, project_root):
        name = target.name.lower()
        if name == "saved_models":
            _cleanup_saved_models(target, dry_run=args.dry_run)
        else:
            _remove_dir(target, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

