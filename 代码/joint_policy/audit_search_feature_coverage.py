"""Recover saved pre-action observations without repeating physical dynamics."""
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from reflection_policy import ReflectionPolicy, reflect_observation
from joint_runtime_v5 import environment_factory
from study_io import BASE, register, receipt, sha, write_json


def saved_observations(reference, rows, actions):
    env = environment_factory(reference)
    observations = []
    for tick, row in enumerate(rows):
        if row[12] > .5:
            break
        if tick % 20 == 0:
            observations.append(env.observation())
            env.held = actions[tick//20].copy()
            env.decisions += 1
        env.position = row[:2].copy()
        env.velocity = row[2:4].copy()
        env.acceleration = row[4:6].copy()
        # Stored requested vector encodes the already executed guide heading.
        env.heading = math.atan2(row[9], row[8])
        env.tracker.update(env.position)
        env.ticks += 1
        rest = np.linalg.norm(env.velocity)<=.001 and np.linalg.norm(env.acceleration)<=.01
        env.rest_ticks = env.rest_ticks+1 if rest else 0
    return np.asarray(observations)


def main():
    parent=BASE/'broader_shape_learning_20260916'
    model=BASE/'reflection_shape_search_20260916/best_generation_5.pt'
    files=[parent/f'training_traces/baseline_{i}.npz' for i in range(16)]
    stage=register('search_feature_coverage_audit_20260916',
        [__file__, model, parent/'basis.npz', BASE/'reflection_policy.py', BASE/'joint_runtime.py']+files,
        dict(traces=16, observation_rows=4000, physics_ticks=0, training_updates=0, wall_s=90),
        'Check the restricted search basis on actual broad-curriculum observations. Recover state from saved rows and verify all deterministic actions; variance loss is descriptive, not proof of policy insufficiency.')
    started=time.perf_counter();torch.set_num_threads(1)
    incoming=torch.load(model,map_location='cpu',weights_only=False)
    policy=ReflectionPolicy(72);policy.load_state_dict(incoming['policy'])
    observations=[];indices=[];errors=[]
    for i,path in enumerate(files):
        with np.load(path) as d:
            obs=saved_observations(d['reference'],d['rows'],d['actions'])
            with torch.no_grad():
                latent=policy.distribution(torch.tensor(obs)).mean.numpy()
            held=np.array([policy_held(z) for z in latent])
            error=float(np.max(np.abs(held-d['actions'][:len(obs)])))
            assert error<1e-4,(i,error)
        observations.append(obs);indices.extend([i]*len(obs));errors.append(error)
    obs=torch.tensor(np.vstack(observations));assert len(obs)<=4000
    with torch.no_grad():
        raw=policy.actor[:4](obs).numpy().astype(float)
        mirrored=policy.actor[:4](reflect_observation(obs)).numpy().astype(float)
    basis=np.load(parent/'basis.npz')['basis'];axes=basis[:8,:96]
    orthogonal=np.linalg.qr(axes.T)[0]
    centered=raw-raw.mean(0)
    retained=float(np.sum((centered@orthogonal)**2)/np.sum(centered**2))
    singular=np.linalg.svd(centered,compute_uv=False)
    fractions=np.cumsum(singular**2)/np.sum(singular**2)
    projected=np.c_[raw,np.ones(len(raw))]@basis.T
    np.savez_compressed(stage/'saved_states.npz',observation=obs.numpy(),hidden=raw,mirrored_hidden=mirrored,
        training_trace_index=np.asarray(indices),basis=basis)
    result=dict(observation_rows=len(obs),maximum_action_reconstruction_error=max(errors),
        existing_rank8_variance_fraction=retained,optimal_rank8_variance_fraction=float(fractions[7]),
        optimal_rank16_variance_fraction=float(fractions[15]),
        search_feature_abs_quantiles=np.quantile(np.abs(projected[:,:8]),[.5,.9,1.],axis=0).tolist(),
        boundary='Hidden feature variance is not action relevance or a sufficiency proof; no new physical rollout or policy training.',
        model_sha256=sha(model),physics_ticks=0,training_updates=0)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_saved_feature_coverage_audit',observation_rows=len(obs),
        physics_ticks=0,training_updates=0,wall_s=time.perf_counter()-started))
    print(json.dumps(result,indent=2))


def policy_held(latent):
    from joint_runtime_v2 import JointRuntimeV2
    return JointRuntimeV2.decode_action(latent)


if __name__=='__main__':main()
