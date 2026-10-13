"""Read-only PPO cause analysis of frozen exploration runs (train/validation only).

Never calls update(), optimizer.step(), training entry points, or test evaluation.
The sole output tree is revision_20260909_cause_analysis/ppo. Replay uses stored
latent actions to reconstruct the first historical rollout without updating.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project"))
FORMAL = ROOT / "论文项目/revision_20260908/preview_direct_explore/formal"
OUT = ROOT / "论文项目/revision_20260909_cause_analysis/ppo"
INPUTS = {}


def track(path):
    path = Path(path)
    key = path.relative_to(ROOT).as_posix()
    if key not in INPUTS:
        INPUTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in track(path).read_text(encoding="utf-8").splitlines() if line]


def read_trace(path):
    with np.load(track(path), allow_pickle=False) as data:
        matrix, columns = data["data"], data["columns"]
        return {str(k): matrix[:, i] for i, k in enumerate(columns)}


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def save_batch(name, batch):
    directory = OUT / "probe_batches"
    directory.mkdir(exist_ok=True)
    path = directory / (name + ".npz")
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in batch.items()})
    return path.relative_to(OUT).as_posix()


def moments(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {k: None for k in ("mean", "std", "min", "p05", "median", "p95", "max")}
    return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()),
                p05=float(np.quantile(a, .05)), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(a.max()))


def flatten_stats(row, prefix, values):
    row.update({prefix + "_" + key: value for key, value in moments(values).items()})


def aggregate():
    run_rows, episode_rows, update_rows, validation_rows, reward_rows = [], [], [], [], []
    source_audits = []
    for run in sorted(FORMAL.iterdir()):
        if not run.is_dir():
            continue
        cfg = yaml.safe_load(track(run / "config.yaml").read_text(encoding="utf-8"))
        summary = read_json(run / "training_summary.json")
        provenance = read_json(run / "provenance.json")
        track(run / "checkpoint.pt")
        track(run / "source.zip")
        archived_checks, changed = [], []
        with zipfile.ZipFile(run / "source.zip") as archive:
            for name, expected in provenance["source_files"].items():
                actual = hashlib.sha256(archive.read(name)).hexdigest()
                assert actual == expected, name
                current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                archived_checks.append(name)
                if current != actual:
                    changed.append(dict(path=name, archived=actual, current=current))
        source_audits.append(dict(run=run.name, source_sha256=provenance["source_sha256"],
                                  archived_members_verified=len(archived_checks), current_differences=changed))
        diagnostics = read_lines(run / "actor_update_diagnostics.jsonl")
        losses = read_lines(run / "updates.jsonl")
        episodes = read_lines(run / "episodes.jsonl")
        assert len({(d["step"], d["transition_count"]) for d in diagnostics}) == len(diagnostics)
        assert sum(d["transition_count"] for d in diagnostics) == summary["steps"]
        assert [d["step"] for d in diagnostics] == [d["step"] for d in losses]
        assert len({d["invocation_id"] for d in diagnostics}) == 1
        assert {d["checkpoint_start_steps"] for d in diagnostics} == {0}
        identity = dict(run=run.name, variant=summary["variant"], seed=summary["seed"])
        epochs = [epoch for d in diagnostics for epoch in d["epochs"]]
        row = dict(**identity, steps=summary["steps"], interaction_s=summary["steps"] * .001,
                   wall_s=summary["training_wall_s"], complete_episodes=sum(e["done"] for e in episodes),
                   successful_episodes=sum(e["done_reason"] == "success" for e in episodes),
                   failed_episodes=sum(e["done"] and e["done_reason"] != "success" for e in episodes),
                   budget_partial_episodes=sum(e["budget_truncation"] for e in episodes),
                   update_count=len(diagnostics), optimizer_step_count=len(epochs),
                   full_512_updates=sum(d["transition_count"] == 512 for d in diagnostics),
                   shorter_updates=sum(d["transition_count"] < 512 for d in diagnostics),
                   gamma=cfg["ppo"]["gamma"], lmbda=cfg["ppo"]["lmbda"],
                   discount_time_constant_s=-.001 / math.log(cfg["ppo"]["gamma"]),
                   max_rollout_s=.512, anchor_beta=cfg["revision"]["motion_anchor_beta"])
        for e in episodes:
            episode_rows.append(dict(**identity, **e))
        for diagnostic in diagnostics:
            for epoch in diagnostic["epochs"]:
                update_rows.append(dict(**identity, step=diagnostic["step"], transitions=diagnostic["transition_count"],
                                        clip_scale=min(1., .5 / (epoch["actor_gradient_norm_before_clip"] + 1e-6)),
                                        **{k: v for k, v in epoch.items() if not isinstance(v, list)}))
        scales = np.minimum(1., .5 / (np.array([e["actor_gradient_norm_before_clip"] for e in epochs]) + 1e-6))
        row["global_clip_epoch_fraction"] = float(np.mean(scales < 1))
        row["ratio_outside_eps_epoch_fraction"] = float(np.mean([e["ratio_min"] < .9 or e["ratio_max"] > 1.1 for e in epochs]))
        row["ratio_outside_eps_noninitial_epoch_fraction"] = float(np.mean([e["ratio_min"] < .9 or e["ratio_max"] > 1.1 for e in epochs if e["epoch"] > 0]))
        row["latent_boundary_clipped_fraction"] = float(sum(d["preview_latent_clipped_fraction"] * d["transition_count"] for d in diagnostics) / summary["steps"])
        flatten_stats(row, "global_clip_scale", scales)
        for field in ("actor_gradient_norm_before_clip", "preview_mu_weight_gradient_norm",
                      "physics_mu_weight_gradient_norm", "preview_std_mean", "motion_anchor_kl",
                      "policy_loss", "anchor_loss", "critic_loss"):
            flatten_stats(row, field, [e[field] for e in epochs])
        flatten_stats(row, "preview_to_physics_mu_weight_norm_ratio",
                      [e["preview_mu_weight_gradient_norm"] / max(e["physics_mu_weight_gradient_norm"], 1e-30) for e in epochs])
        row["logged_preview_mu_min_mm"] = .8 + 3.8 * min(e["preview_mu_min"] for e in epochs)
        row["logged_preview_mu_max_mm"] = .8 + 3.8 * max(e["preview_mu_max"] for e in epochs)
        row["last_update_preview_mu_low_mm"] = .8 + 3.8 * diagnostics[-1]["epochs"][-1]["preview_mu_min"]
        row["last_update_preview_mu_high_mm"] = .8 + 3.8 * diagnostics[-1]["epochs"][-1]["preview_mu_max"]
        row["last_update_preview_std_mm"] = 3.8 * diagnostics[-1]["epochs"][-1]["preview_std_mean"]
        row["physics_std_min"] = min(min(e["physics_std_mean"]) for e in epochs)
        row["physics_std_max"] = max(max(e["physics_std_mean"]) for e in epochs)
        arrays = {}
        trace_paths = sorted(run.glob("train_trace_*.npz"))
        for trace_path in trace_paths:
            trace = read_trace(trace_path)
            for key, values in trace.items():
                if key in ("lookahead_mm", "reward", "latent_2", "v_mm_s", "kcm_intervention") or key.startswith("reward_components__"):
                    arrays.setdefault(key, []).append(values[1:])
        arrays = {k: np.concatenate(v) for k, v in arrays.items()}
        assert len(arrays["lookahead_mm"]) == summary["steps"]
        for key in ("lookahead_mm", "reward", "v_mm_s", "kcm_intervention"):
            flatten_stats(row, "train_" + key, arrays[key])
        row["train_D_first_quarter_mm"] = float(arrays["lookahead_mm"][:32768].mean())
        row["train_D_last_quarter_mm"] = float(arrays["lookahead_mm"][-32768:].mean())
        row["train_D_below_2_3_fraction"] = float(np.mean(arrays["lookahead_mm"] < 2.3))
        row["train_D_above_4_fraction"] = float(np.mean(arrays["lookahead_mm"] > 4.))
        row["train_D_outside_3_plusminus_0_5_fraction"] = float(np.mean(np.abs(arrays["lookahead_mm"] - 3) > .5))
        block = arrays["lookahead_mm"][:len(arrays["lookahead_mm"]) // 100 * 100].reshape(-1, 100).mean(1)
        row["train_D_100ms_block_mean_std_mm"] = float(block.std())
        row["train_D_lag1_correlation"] = float(np.corrcoef(arrays["lookahead_mm"][:-1], arrays["lookahead_mm"][1:])[0, 1]) if np.std(arrays["lookahead_mm"]) > 1e-12 else 0.
        row["train_reward_sum"] = float(arrays["reward"].sum())
        for key, values in arrays.items():
            if key.startswith("reward_components__"):
                reward_rows.append(dict(**identity, component=key, total=float(np.nansum(values)), **moments(values)))
        for path in sorted((run / "evaluation_validation_nominal").glob("*.npz")):
            trace = read_trace(path)
            result = read_json(path.with_suffix(".json"))
            vrow = dict(**identity, path_id=path.stem)
            for key in ("lookahead_mm", "v_mm_s", "kcm_intervention", "control_status__direct_observation_q", "control_status__direct_observation_range_over_pi"):
                flatten_stats(vrow, key, trace[key][1:])
            vrow["D_range_mm"] = float(np.ptp(trace["lookahead_mm"][1:]))
            validation_rows.append(vrow)
        run_rows.append(row)
        print("aggregated", run.name, flush=True)
    write_csv("run_summary.csv", run_rows)
    write_csv("episodes.csv", episode_rows)
    write_csv("epochs.csv", update_rows)
    write_csv("validation_distance.csv", validation_rows)
    write_csv("reward_components.csv", reward_rows)
    write_json("source_audit.json", source_audits)
    write_json("run_summary.json", run_rows)
    return run_rows


def gradient_group(parameters, gradients):
    accum = dict(trunk=0., physics_mu=0., preview_mu=0., physics_std=0., preview_std=0.)
    for (name, parameter), gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if name.startswith("mu_head") or name.startswith("std_head"):
            suffix = "mu" if name.startswith("mu") else "std"
            accum["physics_" + suffix] += float(gradient[:2].square().sum())
            accum["preview_" + suffix] += float(gradient[2:].square().sum())
        else:
            accum["trunk"] += float(gradient.square().sum())
    out = {key: math.sqrt(value) for key, value in accum.items()}
    out["global"] = math.sqrt(sum(accum.values()))
    for dim, label in ((0, "angular"), (1, "feed"), (2, "preview")):
        for head in ("mu", "std"):
            out[label + "_" + head + "_head"] = math.sqrt(sum(float(g[dim].square().sum()) for (name, _), g in zip(parameters, gradients) if name.startswith(head + "_head") and g is not None))
    return out


def gradient_decomposition(agent, batch):
    from src.utils import rl_utils
    agent.critic.eval()
    states = torch.tensor(np.array(batch["states"]), dtype=torch.float32)
    actions = torch.tensor(np.array(batch["actions"]), dtype=torch.float32)
    next_states = torch.tensor(np.array(batch["next_states"]), dtype=torch.float32)
    rewards = torch.tensor(batch["rewards"], dtype=torch.float32).view(-1, 1)
    dones = torch.tensor(batch["dones"], dtype=torch.float32).view(-1, 1)
    terminated = torch.tensor(batch["terminated"], dtype=torch.float32).view(-1, 1)
    td_target = rewards + agent.gamma * agent.critic(next_states) * (1 - terminated)
    td_delta = td_target - agent.critic(states)
    raw_adv = rl_utils.compute_advantage(agent.gamma, agent.lmbda, td_delta.cpu(), dones.cpu())
    adv = raw_adv
    if adv.std(unbiased=False) > 1e-6:
        adv = torch.clamp((adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8), -10, 10)
    mu, std = agent.actor(states)
    logp = agent.latent_log_prob(mu, std, actions)
    ratio = torch.exp(torch.clamp(logp - logp.detach(), -5, 5))
    policy = -torch.min(ratio * adv, torch.clamp(ratio, 1-agent.eps, 1+agent.eps) * adv).mean()
    neg_entropy = -torch.distributions.Normal(mu, std).entropy()[:, agent.active_action_dims].sum(1).mean() * agent.ent_coef
    anchor_raw = agent.motion_anchor_kl(mu, std)
    anchor = agent.anchor_beta * anchor_raw
    parameters = list(agent.actor.named_parameters())
    gradients, components = {}, {}
    for name, loss in (("policy", policy), ("negative_entropy", neg_entropy), ("anchor_unweighted", anchor_raw),
                       ("anchor_weighted", anchor), ("total", policy + neg_entropy + anchor)):
        grad = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
        gradients[name] = grad
        components[name] = dict(loss=float(loss.detach()), **gradient_group(parameters, grad))
    total_norm = components["total"]["global"]
    scale = min(1., .5 / (total_norm + 1e-6))
    grad = gradients["total"][dict((name, i) for i, (name, _) in enumerate(parameters))["mu_head.weight"]]
    bias = gradients["total"][dict((name, i) for i, (name, _) in enumerate(parameters))["mu_head.bias"]]
    result = dict(transition_count=len(states), phase="before_any_optimizer_step", ratio_sample_clip_fraction=0.,
                  ratio_explanation="old and new distributions are identical; historical later epochs cannot be recovered from extrema",
                  reward=moments(rewards.detach().numpy()), td_delta=moments(td_delta.detach().numpy()),
                  raw_advantage=moments(raw_adv.numpy()), normalized_advantage=moments(adv.numpy()),
                  mean_distance_mm=moments((.8+3.8*mu[:, 2]).detach().numpy()),
                  preview_std_mm=moments((3.8*std[:, 2]).detach().numpy()),
                  physics_std_min=float(std[:, :2].detach().min()), physics_std_max=float(std[:, :2].detach().max()),
                  global_clip_scale=scale, components=components,
                  preview_mu_weight_gradient_norm=float(grad[2].norm()),
                  physics_mu_weight_gradient_norm=float(grad[:2].norm()),
                  preview_mu_bias_gradient=float(bias[2]))
    # Score correlations are within a rollout, descriptive only; not independent replicates.
    epsilon = ((actions - mu.detach()) / std.detach()).numpy()
    result["sampled_noise_normalized_advantage_correlations"] = [float(np.corrcoef(epsilon[:, i], adv.numpy().ravel())[0, 1]) for i in range(3)]
    derivative = torch.cat((1 - mu[:, :1].detach().square(), mu[:, 1:].detach() * (1 - mu[:, 1:].detach())), dim=1)
    score = torch.tensor(epsilon) / std.detach() * derivative
    result["raw_mu_score_rms_by_action"] = torch.sqrt(score.square().mean(0)).tolist()
    return result


def probe():
    import preview_direct_explore as explore
    study = explore.install()
    path_manifest = read_json(study.DATA / "manifest.json")
    for entry in path_manifest["paths"]:
        if entry["split"] == "train":
            path = track(study.DATA / entry["reference_file"])
            assert INPUTS[path.relative_to(ROOT).as_posix()] == entry["reference_sha256"]
    torch.set_num_threads(1)
    outputs = []
    optimizer_rows = []
    variants = ("selected_fixed", "direct", "selected_fixed_no_anchor", "direct_no_anchor")
    for variant in variants:
        for seed in (11, 22, 33, 44, 55):
            run = FORMAL / f"{variant}_seed{seed}"
            cfg = yaml.safe_load(track(run / "config.yaml").read_text(encoding="utf-8"))
            path_id = cfg["revision"]["training_path_ids"][0]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            env, obs = study.make_env(cfg, path_id)
            agent = explore.make_agent(cfg, env)
            initial_parameters = {k: v.clone() for k, v in agent.actor.state_dict().items()}
            first = read_trace(sorted(run.glob("train_trace_*.npz"))[0])
            expected = read_lines(run / "actor_update_diagnostics.jsonl")[0]
            batch = {k: [] for k in ("states", "actions", "next_states", "rewards", "dones", "terminated")}
            errors = dict(reward=0., x_mm=0., y_mm=0.)
            for i in range(1, expected["transition_count"] + 1):
                latent = np.array([first[f"latent_{j}"][i] for j in range(3)])
                action = np.clip(latent[:env.action_space_dim], env.action_space.low, env.action_space.high)
                nxt, reward, done, info = env.step(action)
                terminal = done and info.get("done_reason") not in ("max_steps", "timeout", "time_limit")
                for key, value in zip(batch, (obs, latent, nxt, reward, done, terminal)):
                    batch[key].append(value)
                errors["reward"] = max(errors["reward"], abs(reward-first["reward"][i]))
                errors["x_mm"] = max(errors["x_mm"], abs(env.current_position[0]-first["x_mm"][i]))
                errors["y_mm"] = max(errors["y_mm"], abs(env.current_position[1]-first["y_mm"][i]))
                obs = nxt
            assert max(errors.values()) < 1e-9, errors
            result = gradient_decomposition(agent, batch)
            batch_file = save_batch(run.name + "_historical_first", batch)
            reference = expected["epochs"][0]
            match = {name: abs(result[name]-reference[name]) for name in
                     ("preview_mu_weight_gradient_norm", "physics_mu_weight_gradient_norm", "preview_mu_bias_gradient")}
            assert max(match.values()) < 1e-5, match
            outputs.append(dict(run=run.name, variant=variant, seed=seed, batch_kind="historical_first_rollout_reconstructed",
                                path_id=path_id, batch_file=batch_file, replay_max_absolute_error=errors,
                                logged_epoch0_gradient_absolute_error=match, **result))
            # Final model, new on-policy train batch; deterministic seed is recorded.
            checkpoint = torch.load(track(run / "checkpoint.pt"), map_location="cpu", weights_only=False)
            agent.actor.load_state_dict(checkpoint["actor"])
            agent.critic.load_state_dict(checkpoint["critic"])
            actor_before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
            params = list(agent.actor.named_parameters())
            shifts = [actor_before[k] - initial_parameters[k] for k, _ in params]
            opt = checkpoint["actor_optimizer"]
            ordered_states = [opt["state"][index] for group in opt["param_groups"] for index in group["params"]]
            optimizer_rows.append(dict(run=run.name, variant=variant, seed=seed,
                adam_steps=sorted(set(int(s["step"]) for s in ordered_states)),
                initial_to_final_parameter_change=gradient_group(params, shifts),
                exp_avg_norms=gradient_group(params, [s["exp_avg"] for s in ordered_states]),
                exp_avg_sq_sum_sqrt=gradient_group(params, [torch.sqrt(s["exp_avg_sq"]) for s in ordered_states]),
                normalized_moment_direction=gradient_group(params, [s["exp_avg"] / (torch.sqrt(s["exp_avg_sq"]) + opt["param_groups"][0]["eps"]) for s in ordered_states]),
                note="Saved Adam moments only; normalized_moment_direction omits bias correction and is not an actual update or training step"))
            for train_id in cfg["revision"]["training_path_ids"]:
                diagnostic_seed = 909000 + seed
                random.seed(diagnostic_seed); np.random.seed(diagnostic_seed); torch.manual_seed(diagnostic_seed)
                env, obs = study.make_env(cfg, train_id)
                batch = {k: [] for k in batch}
                for _ in range(512):
                    latent = agent.take_action(obs)
                    action = np.clip(latent[:env.action_space_dim], env.action_space.low, env.action_space.high)
                    nxt, reward, done, info = env.step(action)
                    terminal = done and info.get("done_reason") not in ("max_steps", "timeout", "time_limit")
                    for key, value in zip(batch, (obs, latent, nxt, reward, done, terminal)):
                        batch[key].append(value)
                    obs = nxt
                    if done:
                        break
                result = gradient_decomposition(agent, batch)
                batch_file = save_batch(run.name + "_frozen_final_" + train_id, batch)
                outputs.append(dict(run=run.name, variant=variant, seed=seed, diagnostic_seed=diagnostic_seed,
                                    batch_kind="frozen_final_model_new_train_rollout", path_id=train_id, batch_file=batch_file, **result))
            assert all(torch.equal(actor_before[k], v) for k, v in agent.actor.state_dict().items())
            print("probed without optimizer steps", run.name, flush=True)
    write_json("gradient_decomposition.json", outputs)
    write_json("optimizer_snapshots.json", optimizer_rows)
    rows = []
    for item in outputs:
        for component, values in item["components"].items():
            rows.append(dict(run=item["run"], variant=item["variant"], seed=item["seed"],
                             batch_kind=item["batch_kind"], path_id=item["path_id"],
                             global_clip_scale=item["global_clip_scale"], component=component, **values))
    write_csv("gradient_decomposition.csv", rows)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("aggregate", "probe", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("aggregate", "all"):
        aggregate()
    if args.phase in ("probe", "all"):
        probe()
    track(Path(__file__))
    write_json("inputs_" + args.phase + ".json", dict(command="python " + " ".join(sys.argv),
               scope="train and already saved validation only; no test data; no optimization steps", files=INPUTS))


if __name__ == "__main__":
    main()
