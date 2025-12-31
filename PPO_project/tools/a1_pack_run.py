"""
tools/a1_pack_run.py (starter implementation)

Purpose:
- Standardize any run (PhaseB, PhaseC, etc.) into a Run Bundle:
  - eval/run1 + eval/run2 (acceptance_suite)
  - a1_verdict.json
  - manifest.json binding
  - optional rollout_det via rollout_cmd template

This is intended to be called at the end of each PhaseB run.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, shutil, subprocess
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
    try:
        import yaml  # type: ignore
    except Exception:
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
            "success_rate": get("success_rate"),
            "stall_rate": get("stall_rate"),
            "mean_progress_final": get("mean_progress_final"),
            "max_abs_contour_error": get("max_abs_contour_error"),
            "has_non_finite": get("has_non_finite"),
            "half_epsilon": get("half_epsilon"),
        },
        "segment": {},
        "smoothness": {},
        "raw_keys": sorted(list(raw_sum.keys())) if isinstance(raw_sum, dict) else [],
    }

def verdict_from_norm(s1: dict, s2: dict) -> dict:
    def g(s, path, default=None):
        cur = s
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
    checks=[]
    def add(path, rel_thr=0.01, abs_thr=1e-6):
        a=g(s1,path); b=g(s2,path)
        ok=True; delta=None
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            delta=abs(a-b); denom=max(abs(a),1e-12)
            ok=(delta<=abs_thr) or (delta/denom<=rel_thr)
        elif isinstance(a,bool) and isinstance(b,bool):
            ok=(a==b)
        checks.append({"key":".".join(path),"run1":a,"run2":b,"delta":delta,"pass":ok,"rel_thr":rel_thr,"abs_thr":abs_thr})
    add(["global","passed"], rel_thr=0.0, abs_thr=0.0)
    add(["global","success_rate"], rel_thr=0.05, abs_thr=1e-6)
    add(["global","max_abs_contour_error"], rel_thr=0.01, abs_thr=1e-4)
    add(["global","mean_progress_final"], rel_thr=0.01, abs_thr=1e-4)
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "reasons": []}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tag", default="phaseB")
    ap.add_argument("--run_id", default=None, help="default: timestamp-based")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_root", default="artifacts")
    ap.add_argument("--seed_eval", type=int, default=43)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--eval_cmd", required=True,
                    help="Template with {config_eval} {ckpt} {episodes} {out_dir} {det_flag}")
    ap.add_argument("--rollout_cmd", default=None,
                    help="Optional template to generate trace/plots into {out_dir}")
    ap.add_argument("--baseline_ref", default=None)
    args=ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (project_root / args.out_root / args.tag / run_id).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Copy inputs
    ckpt = Path(args.model).resolve()
    cfg = Path(args.config).resolve()
    shutil.copy2(ckpt, out/"checkpoint.pth")
    shutil.copy2(cfg, out/"config.yaml")

    cfg_eval = out/"config_eval.yaml"
    seed_info = write_config_eval(out/"config.yaml", cfg_eval, args.seed_eval)

    # Eval run1
    (out/"eval"/"run1").mkdir(parents=True, exist_ok=True)
    det_flag = "--deterministic" if args.deterministic else ""
    cmd1 = args.eval_cmd.format(config_eval=str(cfg_eval), ckpt=str(out/"checkpoint.pth"),
                                episodes=args.episodes, out_dir=str(out/"eval"/"run1"), det_flag=det_flag)
    subprocess.check_call(cmd1, shell=True)
    (out/"eval"/"run1"/"summary.json").rename(out/"eval"/"run1"/"summary_raw.json")

    # Eval run2
    (out/"eval"/"run2").mkdir(parents=True, exist_ok=True)
    cmd2 = args.eval_cmd.format(config_eval=str(cfg_eval), ckpt=str(out/"checkpoint.pth"),
                                episodes=args.episodes, out_dir=str(out/"eval"/"run2"), det_flag=det_flag)
    subprocess.check_call(cmd2, shell=True)
    (out/"eval"/"run2"/"summary.json").rename(out/"eval"/"run2"/"summary_raw.json")

    # Optional rollout_det
    (out/"rollout_det"/"plots").mkdir(parents=True, exist_ok=True)
    if args.rollout_cmd:
        rc = args.rollout_cmd.format(config_eval=str(cfg_eval), ckpt=str(out/"checkpoint.pth"),
                                     out_dir=str(out/"rollout_det"), det_flag="--deterministic")
        subprocess.check_call(rc, shell=True)

    git_hash = try_git_hash(project_root)
    config_hash = sha256_file(out/"config.yaml")
    ckpt_hash = sha256_file(out/"checkpoint.pth")
    trace_hash = sha256_file(out/"rollout_det"/"trace.csv") if (out/"rollout_det"/"trace.csv").exists() else None

    meta = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash,
        "config_hash": config_hash,
        "checkpoint_hash": ckpt_hash,
        "seed_eval": seed_info.get("seed_eval"),
        "seed_source": seed_info.get("seed_source"),
        "env_hash": "unknown",
        "baseline_ref": args.baseline_ref,
        "episode_set_source": "config",
    }

    s1_raw = json_load(out/"eval"/"run1"/"summary_raw.json")
    s2_raw = json_load(out/"eval"/"run2"/"summary_raw.json")
    json_dump(normalize_summary(s1_raw, meta), out/"eval"/"run1"/"summary.json")
    json_dump(normalize_summary(s2_raw, meta), out/"eval"/"run2"/"summary.json")

    verdict = verdict_from_norm(json_load(out/"eval"/"run1"/"summary.json"),
                                json_load(out/"eval"/"run2"/"summary.json"))
    verdict["paths"] = {
        "run1_norm": "eval/run1/summary.json",
        "run2_norm": "eval/run2/summary.json",
        "run1_raw": "eval/run1/summary_raw.json",
        "run2_raw": "eval/run2/summary_raw.json",
        "trace": "rollout_det/trace.csv" if trace_hash else None,
    }
    json_dump(verdict, out/"a1_verdict.json")

    manifest = {
        "run_id": run_id,
        "git_hash": git_hash,
        "config_hash": config_hash,
        "checkpoint_hash": ckpt_hash,
        "trace_hash": trace_hash,
        "episode_set_source": "config",
        "baseline_ref": args.baseline_ref,
        "eval": {
            "phase": "p0_eval",
            "command_run1": cmd1,
            "command_run2": cmd2,
            "episodes": args.episodes,
            "deterministic": bool(args.deterministic),
            "seed_eval": seed_info.get("seed_eval"),
            "seed_source": seed_info.get("seed_source"),
            "config_eval_path": str(cfg_eval),
        },
        "rollout_det": {"command_template": args.rollout_cmd} if args.rollout_cmd else {"command_template": None},
    }
    json_dump(manifest, out/"manifest.json")

    print(f"[OK] Run Bundle written to: {out}")
    print(f"[VERDICT] pass={verdict['pass']}")

if __name__ == "__main__":
    main()
