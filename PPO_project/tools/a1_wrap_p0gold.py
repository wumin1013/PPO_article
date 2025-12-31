"""
tools/a1_wrap_p0gold.py (starter implementation)

Goal:
- Read existing P0_gold evidence stack (checkpoint/config/summary/trace/plots/manifest)
- Produce Level-2 Run Bundle with:
  - eval/run1/summary_raw.json (copy from P0)
  - eval/run2/summary_raw.json (re-eval using acceptance_suite)
  - eval/run*/summary.json (normalized schema for aggregation)
  - a1_verdict.json (stability check)
  - manifest.json (bind commands + hashes)

Reality check (current repo):
- tools/acceptance_suite.py supports --seed / --episode_set.
- We still patch config_eval.yaml for seed determinism if PyYAML is available.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def json_load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def json_dump(obj: dict, p: Path):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def try_git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
        return out.decode().strip()
    except Exception:
        return "unknown"

def write_config_eval(src_cfg: Path, dst_cfg: Path, seed_eval: int) -> dict:
    """
    Writes a patched YAML config that forces seed. Uses PyYAML if available.
    Returns a small dict with seed info.
    """
    try:
        import yaml  # type: ignore
    except Exception:
        # Fallback: copy as-is (seed comes from original config)
        shutil.copy2(src_cfg, dst_cfg)
        return {"seed_eval": None, "seed_source": "config (unpatched, PyYAML missing)"}

    cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        cfg = {}

    cfg["seed"] = int(seed_eval)
    exp = cfg.get("experiment", {})
    if isinstance(exp, dict):
        exp["seed"] = int(seed_eval)
        cfg["experiment"] = exp

    dst_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"seed_eval": int(seed_eval), "seed_source": "patched_config_eval.yaml"}

def normalize_summary(raw_payload: dict, meta: dict) -> dict:
    """
    acceptance_suite writes: {"summary": {EvalSummary fields...}, "episodes": [...]}
    We keep raw, and create a normalized schema that is stable for aggregation.
    """
    raw_sum = raw_payload.get("summary", {}) if isinstance(raw_payload, dict) else {}

    def get(k, default=None):
        return raw_sum.get(k, default) if isinstance(raw_sum, dict) else default

    return {
        "meta": meta,
        "global": {
            "phase": get("phase"),
            "passed": get("passed"),
            "episodes": get("episodes"),
            "deterministic": get("deterministic"),
            "model_path": get("model_path"),
            "seed_eval": get("seed_eval"),
            "episode_set": get("episode_set"),
            "success_rate": get("success_rate"),
            "stall_rate": get("stall_rate"),
            "mean_progress_final": get("mean_progress_final"),
            "max_abs_contour_error": get("max_abs_contour_error"),
            "mean_return": get("mean_return"),
            "has_non_finite": get("has_non_finite"),
            "half_epsilon": get("half_epsilon"),
            "thresholds": get("thresholds"),
            "timestamp": get("timestamp"),
            "config_path": get("config_path"),
        },
        "segment": {},      # optional: fill if you compute from trace later
        "smoothness": {},   # optional: fill if you compute from trace later
        "raw_keys": sorted(list(raw_sum.keys())) if isinstance(raw_sum, dict) else [],
    }

def verdict_from_norm(s1: dict, s2: dict, smoke: dict | None = None) -> dict:
    """
    Simple stability verdict for p0_eval plus A0 smoke/eval pass gate.
    """

    def g(s, path, default=None):
        cur = s
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    checks = []

    def add_num(path, rel_thr=0.01, abs_thr=1e-6):
        a = g(s1, path)
        b = g(s2, path)
        ok = True
        delta = None
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = abs(a - b)
            denom = max(abs(a), 1e-12)
            ok = (delta <= abs_thr) or (delta / denom <= rel_thr)
        elif isinstance(a, bool) and isinstance(b, bool):
            ok = (a == b)
        checks.append(
            {
                "key": ".".join(path),
                "run1": a,
                "run2": b,
                "delta": delta,
                "pass": ok,
                "rel_thr": rel_thr,
                "abs_thr": abs_thr,
            }
        )

    # Key checks (aligned with acceptance_suite outputs)
    add_num(["global", "passed"], rel_thr=0.0, abs_thr=0.0)
    add_num(["global", "success_rate"], rel_thr=0.05, abs_thr=1e-6)
    add_num(["global", "max_abs_contour_error"], rel_thr=0.01, abs_thr=1e-4)
    add_num(["global", "mean_progress_final"], rel_thr=0.01, abs_thr=1e-4)

    eval_pass = bool(g(s2, ["global", "passed"], False))
    smoke_pass = None
    if smoke is not None:
        smoke_pass = bool(g(smoke, ["global", "passed"], False))

    reasons = []
    if smoke_pass is False:
        reasons.append("smoke_failed")
    if not eval_pass:
        reasons.append("eval_failed")
    if not all(c["pass"] for c in checks):
        reasons.append("stability_failed")

    return {
        "pass": bool(eval_pass and (smoke_pass is not False)),
        "eval_pass": eval_pass,
        "smoke_pass": smoke_pass,
        "stability_pass": all(c["pass"] for c in checks),
        "checks": checks,
        "reasons": reasons,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0_gold_dir", required=True)
    ap.add_argument("--out_root", default="artifacts/P0_gold_L2")
    ap.add_argument("--seed_eval", type=int, default=43)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--smoke_episodes", type=int, default=5)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--episode_set", default=None)
    ap.add_argument("--eval_cmd", required=True,
                    help="Template with {config_eval} {ckpt} {episodes} {out_dir} {det_flag} {seed_flag} {episode_set_flag}")
    ap.add_argument("--smoke_cmd", default=None,
                    help="Optional template with {config_eval} {episodes} {out_dir} {det_flag} {seed_flag} {episode_set_flag}")
    ap.add_argument("--rollout_cmd", default=None,
                    help="Optional template to generate trace/plots into {out_dir}")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    p0 = Path(args.p0_gold_dir).resolve()
    assert p0.exists()

    ckpt = p0 / "checkpoint.pth"
    cfg = p0 / "config.yaml"
    raw1 = p0 / "summary.json"
    man1 = p0 / "manifest.json"

    for req in [ckpt, cfg, raw1, man1]:
        if not req.exists():
            raise FileNotFoundError(f"Missing required file: {req}")

    run_id = p0.name
    out = (project_root / args.out_root / run_id).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Copy baseline artifacts
    shutil.copy2(ckpt, out / "checkpoint.pth")
    shutil.copy2(cfg, out / "config.yaml")
    shutil.copy2(man1, out / "manifest.json")

    # rollout_det: reuse if exists
    (out / "rollout_det" / "plots").mkdir(parents=True, exist_ok=True)
    if (p0 / "trace.csv").exists():
        shutil.copy2(p0 / "trace.csv", out / "rollout_det" / "trace.csv")
    for png in ["overlay.png", "v_t.png", "e_n_t.png"]:
        if (p0 / png).exists():
            shutil.copy2(p0 / png, out / "rollout_det" / "plots" / png)

    # Eval run1: keep raw
    (out / "eval" / "run1").mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw1, out / "eval" / "run1" / "summary_raw.json")

    # Prepare patched config_eval.yaml for stable seeding
    cfg_eval = out / "config_eval.yaml"
    seed_info = write_config_eval(out / "config.yaml", cfg_eval, args.seed_eval)

    det_flag = "--deterministic" if args.deterministic else ""
    seed_flag = f"--seed {int(args.seed_eval)}" if args.seed_eval is not None else ""
    episode_set_flag = f'--episode_set "{args.episode_set}"' if args.episode_set else ""
    seed_eval_value = seed_info.get("seed_eval")
    seed_source = seed_info.get("seed_source")
    if seed_eval_value is None and args.seed_eval is not None:
        seed_eval_value = int(args.seed_eval)
        seed_source = "cli_seed"

    # Smoke run
    (out / "smoke").mkdir(parents=True, exist_ok=True)
    if args.smoke_cmd:
        smoke_cmd = args.smoke_cmd.format(
            config_eval=str(cfg_eval),
            episodes=args.smoke_episodes,
            out_dir=str(out / "smoke"),
            det_flag=det_flag,
            seed_flag=seed_flag,
            episode_set_flag=episode_set_flag,
        )
    else:
        acceptance_script = project_root / "tools" / "acceptance_suite.py"
        smoke_cmd = (
            f"\"{sys.executable}\" \"{acceptance_script}\" --phase p0_smoke "
            f"--config \"{cfg_eval}\" --episodes {int(args.smoke_episodes)} --out \"{out / 'smoke'}\" "
            f"{seed_flag} {episode_set_flag} {det_flag}"
        ).strip()
    subprocess.check_call(smoke_cmd, shell=True)
    smoke_raw_src = out / "smoke" / "summary.json"
    if not smoke_raw_src.exists():
        raise FileNotFoundError(f"Expected smoke output summary.json at {smoke_raw_src}")
    smoke_raw_dst = out / "smoke" / "summary_raw.json"
    if smoke_raw_dst.exists():
        smoke_raw_dst.unlink()
    smoke_raw_src.rename(smoke_raw_dst)

    # Eval run2: re-run acceptance/eval
    (out / "eval" / "run2").mkdir(parents=True, exist_ok=True)
    cmd = args.eval_cmd.format(
        config_eval=str(cfg_eval),
        ckpt=str(out / "checkpoint.pth"),
        episodes=args.episodes,
        out_dir=str(out / "eval" / "run2"),
        det_flag=det_flag,
        seed_flag=seed_flag,
        episode_set_flag=episode_set_flag,
    )
    subprocess.check_call(cmd, shell=True)

    # acceptance_suite writes summary.json to out_dir -> rename to summary_raw.json
    raw2_src = out / "eval" / "run2" / "summary.json"
    if not raw2_src.exists():
        raise FileNotFoundError(f"Expected eval output summary.json at {raw2_src}")
    raw2_dst = out / "eval" / "run2" / "summary_raw.json"
    if raw2_dst.exists():
        raw2_dst.unlink()
    raw2_src.rename(raw2_dst)

    # Optionally generate rollout_det if missing and rollout_cmd provided
    if args.rollout_cmd and not (out / "rollout_det" / "trace.csv").exists():
        rc = args.rollout_cmd.format(
            config_eval=str(cfg_eval),
            ckpt=str(out / "checkpoint.pth"),
            out_dir=str(out / "rollout_det"),
            det_flag="--deterministic",
            seed_flag=seed_flag,
            episode_set_flag=episode_set_flag,
        )
        subprocess.check_call(rc, shell=True)

    # Normalize summaries
    git_hash = try_git_hash(project_root)
    config_hash = sha256_file(out / "config.yaml")
    ckpt_hash = sha256_file(out / "checkpoint.pth")
    trace_hash = sha256_file(out / "rollout_det" / "trace.csv") if (out / "rollout_det" / "trace.csv").exists() else None

    meta = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "config_hash": config_hash,
        "checkpoint_hash": ckpt_hash,
        "seed_eval": seed_eval_value,
        "seed_source": seed_source,
        "env_hash": "unknown",
        "baseline_ref": str(out),
        "episode_set": args.episode_set,
        "episode_set_source": "arg" if args.episode_set else "config",
    }

    s1_raw = json_load(out / "eval" / "run1" / "summary_raw.json")
    s2_raw = json_load(out / "eval" / "run2" / "summary_raw.json")
    smoke_raw = json_load(out / "smoke" / "summary_raw.json")
    json_dump(normalize_summary(smoke_raw, meta), out / "smoke" / "summary.json")
    json_dump(normalize_summary(s1_raw, meta), out / "eval" / "run1" / "summary.json")
    json_dump(normalize_summary(s2_raw, meta), out / "eval" / "run2" / "summary.json")

    # Verdict
    s1 = json_load(out / "eval" / "run1" / "summary.json")
    s2 = json_load(out / "eval" / "run2" / "summary.json")
    s_smoke = json_load(out / "smoke" / "summary.json")
    verdict = verdict_from_norm(s1, s2, smoke=s_smoke)
    verdict["paths"] = {
        "smoke_norm": "smoke/summary.json",
        "smoke_raw": "smoke/summary_raw.json",
        "run1_norm": "eval/run1/summary.json",
        "run2_norm": "eval/run2/summary.json",
        "run1_raw": "eval/run1/summary_raw.json",
        "run2_raw": "eval/run2/summary_raw.json",
        "trace": "rollout_det/trace.csv" if trace_hash else None,
    }
    json_dump(verdict, out / "a1_verdict.json")

    # Manifest bind
    m = json_load(out / "manifest.json") if (out / "manifest.json").exists() else {}
    m["run_id"] = run_id
    m["git_hash"] = git_hash
    m["config_hash"] = config_hash
    m["checkpoint_hash"] = ckpt_hash
    m["trace_hash"] = trace_hash
    m["episode_set"] = args.episode_set
    m["episode_set_source"] = "arg" if args.episode_set else "config"
    m["p0_gold_source"] = str(p0)
    m["eval"] = {
        "phase": "p0_eval",
        "command": cmd,
        "episodes": args.episodes,
        "deterministic": bool(args.deterministic),
        "seed_eval": seed_eval_value,
        "seed_source": seed_source,
        "config_eval_path": str(cfg_eval),
        "episode_set": args.episode_set,
    }
    m["smoke"] = {
        "phase": "p0_smoke",
        "command": smoke_cmd,
        "episodes": args.smoke_episodes,
        "seed_eval": seed_eval_value,
        "seed_source": seed_source,
        "episode_set": args.episode_set,
    }
    if args.rollout_cmd:
        m["rollout_det"] = {"command_template": args.rollout_cmd}
    json_dump(m, out / "manifest.json")

    print(f"[OK] Level-2 bundle written to: {out}")
    print(f"[VERDICT] pass={verdict['pass']}")

if __name__ == "__main__":
    main()
