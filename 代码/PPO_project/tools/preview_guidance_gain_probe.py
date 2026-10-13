"""Validation-only constant-action guidance-gain audit; no training or test use."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
import preview_research_v2 as v2
from src.utils.revision_metrics import evaluate_trajectory

DEFAULT_OUT=v2.OUT/"guidance_gain_probe"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    source=v2.source_state()
    source["probe_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol=dict(source=source,constant_action=[0.0,.05,.5],command_feed_mm_s=5.0,
        tangent_blend_values=[.005,.015],distances_mm=[2.3,4.0],split="validation",
        command="python PPO_project/tools/preview_guidance_gain_probe.py",
        scope="Common guidance gain diagnostic, applies to fixed and adaptive methods; not adaptive-preview contribution.",
        actual_cfg_key="reward_weights.control_authority.tangent_blend",schedule="none",training_steps=0)
    protocol_path=out/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8"))!=protocol:
        raise ValueError("Existing probe source/protocol differs; choose a new output directory")
    v2.backend.write_json(protocol_path,protocol)
    paths=[r["id"] for r in v2.manifest()["paths"] if r["split"]=="validation"]
    results=[]
    for blend in [.005,.015]:
        for distance in [2.3,4.0]:
            for path_id in paths:
                name=f"blend_{blend:g}_D{distance:g}_{path_id}"
                target=out/f"{name}.json"
                if target.exists():
                    old=json.loads(target.read_text(encoding="utf-8"));results.append(old)
                    print("CACHED",name,old["success"],flush=True);continue
                cfg=v2.config_for("pretrain_mid",101)
                cfg["reward_weights"]["lookahead"]["distance"]=distance
                cfg["reward_weights"]["lookahead_control"].update(enabled=False,policy_action=False)
                cfg["reward_weights"]["control_authority"].update(mode="blend",tangent_blend=blend,
                    blend_schedule="none",warmup_episodes=0,warmup_blend=blend)
                cfg["revision"].update(variant="constant_guidance_probe",fixed_distance_mm=distance)
                config_path=out/f"{name}.yaml"
                config_path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")
                reference=v2.path_points(path_id)
                env,observation=v2.make_env(cfg,path_id,reference=reference)
                if not np.isclose(env.control_tangent_blend_active,blend):
                    raise AssertionError("Requested tangent blend did not become active")
                action=np.array([0.0,5.0/float(env.MAX_VEL),.5],dtype=np.float64)
                rows=[v2.backend.sample_row(env,{},0.0,action)]
                done=False;observed=[];start=time.perf_counter()
                while not done:
                    observation,reward,done,info=env.step(action[:env.action_space_dim])
                    row=v2.backend.sample_row(env,info,reward,action)
                    rows.append(row)
                    observed.append(float(row["control_status__blend_active"]))
                assert np.allclose(observed,blend,rtol=0,atol=1e-12),"Trace shows unintended blend schedule"
                trace_path=out/f"{name}.npz"
                v2.backend.save_trace(trace_path,rows)
                metrics=evaluate_trajectory(reference,[[r["x_mm"],r["y_mm"]] for r in rows],
                    [r["timestamp_s"] for r in rows],tolerance_mm=cfg["environment"]["epsilon"]/2,
                    limits=cfg["kinematic_constraints"],termination_reason=info["done_reason"],
                    claimed_success=info["done_reason"]=="success",initial_velocity=[0.,0.],
                    initial_acceleration=[0.,0.],initial_heading=rows[0]["heading_rad"],
                    initial_angular_velocity=0.,initial_angular_acceleration=0.)
                metrics.update(path_id=path_id,split="validation",tangent_blend=blend,distance_mm=distance,
                    gain_per_s=blend/cfg["environment"]["interpolation_period"],constant_feed_intent_mm_s=5.0,
                    actual_blend_min=min(observed),actual_blend_max=max(observed),
                    runtime_s=time.perf_counter()-start,training_steps=0,
                    trace_file=trace_path.name,trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),source_sha256=source["source_sha256"],
                    comparison_role="Shared guidance diagnostic; not learned-preview advantage")
                v2.backend.write_json(target,metrics);results.append(metrics)
                v2.backend.write_json(out/"summary.json",{"rows":results,"completed_rollouts":len(results),"planned_rollouts":12})
                print(name,"success",metrics["success"],"progress",round(metrics["ordered_progress"],5),
                    "time",metrics["elapsed_time_s"],"error",round(metrics["contour_error_max_mm"],5),
                    "blend_actual",metrics["actual_blend_min"],metrics["actual_blend_max"],metrics["failure_reasons"],flush=True)
    v2.backend.write_json(out/"summary.json",{"rows":results,"completed_rollouts":len(results),"planned_rollouts":12})
    print(out/"summary.json",flush=True)


if __name__=="__main__":
    main()
