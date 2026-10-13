"""Independent deterministic physical evaluation for one policy candidate.

Workers never update weights from rewards or select candidates. The parent
orders all results, enforces aggregate budgets and performs CEM updates.
"""
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from reflection_policy import ReflectionPolicy
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from validate_actions import difference_audit
from study_io import write_json


def initialize(config):
    global CONFIG, POLICY, WEIGHT, BIAS, BASIS, HEAD
    CONFIG = config
    torch.set_num_threads(1)
    incoming = torch.load(config['model'], map_location='cpu', weights_only=False)
    if config.get('preview_residual'):
        from dynamic_residual_policy import PreviewResidualPolicy
        POLICY=PreviewResidualPolicy(incoming['observation_dim'])
        missing=POLICY.load_state_dict(incoming['policy'],strict=False)
        assert set(missing.missing_keys)=={'preview_head.weight','preview_head.bias'} and not missing.unexpected_keys
        HEAD=POLICY.preview_head
    elif config.get('dynamic_residual'):
        from dynamic_residual_policy import DynamicResidualPolicy
        POLICY=DynamicResidualPolicy(incoming['observation_dim'])
        missing=POLICY.load_state_dict(incoming['policy'],strict=False)
        assert set(missing.missing_keys)=={'dynamic_head.weight','dynamic_head.bias'} and not missing.unexpected_keys
        HEAD=POLICY.dynamic_head
    else:
        POLICY=ReflectionPolicy(incoming['observation_dim']);POLICY.load_state_dict(incoming['policy'])
        HEAD=POLICY.actor[-1]
    WEIGHT=HEAD.weight.detach().numpy().copy()
    BIAS=HEAD.bias.detach().numpy().copy()
    BASIS = np.asarray(config['basis'], float)


def install(coefficients):
    delta = np.asarray(coefficients)@BASIS
    with torch.no_grad():
        HEAD.weight.copy_(torch.as_tensor(WEIGHT+delta[:, :WEIGHT.shape[1]], dtype=torch.float32))
        HEAD.bias.copy_(torch.as_tensor(BIAS+delta[:, -1], dtype=torch.float32))


def action_check(coefficients, observations):
    """No physics: cross-process equality of the actual deterministic action API."""
    install(coefficients)
    return np.asarray([POLICY.act(obs, deterministic=True)[0] for obs in observations])


def evaluate_candidate(coefficients, tag, actor_budget, tick_budget):
    install(coefficients)
    directory = Path(CONFIG['stage'])/'training_traces'
    records = []; actor_calls = 0; physics_ticks = 0; chunks = []; actions = []
    def progress(status):
        write_json(directory/f'{tag}_progress.json', dict(status=status,actor_calls=actor_calls,
            physics_ticks=physics_ticks,completed_paths=len(records)))
    try:
        for index, points in enumerate(CONFIG['paths']):
            ref = np.asarray(points, float); env = environment_factory(ref)
            chunks = []; actions = []; reward_sum = 0.
            while not env.done:
                if time.time() >= CONFIG['deadline_unix']:
                    raise TimeoutError('Registered worker wall limit')
                if actor_calls >= actor_budget or physics_ticks >= tick_budget-2200:
                    raise RuntimeError('Reserved candidate computation budget reached')
                latent, _, _ = POLICY.act(env.observation(), deterministic=True)
                actor_calls += 1
                _, reward, _, info = env.step(latent)
                physics_ticks += len(info['rows'])
                reward_sum += reward; chunks.append(info['rows']); actions.append(info['held_action'])
                if actor_calls % 100 == 0: progress('running')
            rows = np.vstack(chunks); held = np.asarray(actions)
            audit = difference_audit(ref[0], rows)
            success = bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            trace = f'{tag}_{index}.npz'
            np.savez_compressed(directory/trace, reference=ref, rows=rows, actions=held)
            records.append(dict(path=CONFIG['indices'][index], success=success, reason=env.reason,
                elapsed_s=env.ticks*.001, complete_time_s=env.ticks*.001 if success else None,
                return_sum=reward_sum, audit=audit, geometry=env.tracker.snapshot(1.25), trace=trace,
                shape=corner_shape(ref, rows, held) if len(ref)==3 and len(rows)>21 else None))
        result = dict(tag=tag, coefficients=np.asarray(coefficients).tolist(), results=records,
            successes=sum(r['success'] for r in records), return_sum=sum(r['return_sum'] for r in records),
            candidate_complete=True, actor_calls=actor_calls, physics_ticks=physics_ticks)
        if result['successes']==len(records):
            result['complete_time_sum_s']=sum(r['complete_time_s'] for r in records)
            length=sum(np.linalg.norm(np.diff(np.asarray(p),axis=0),axis=1).sum() for p in CONFIG['paths'])
            assert abs(result['return_sum']+result['complete_time_sum_s']-length/100.)<1e-9
        progress('complete')
        write_json(directory/f'{tag}_result.json', result)
        return result
    except Exception:
        if chunks:
            np.savez_compressed(directory/f'{tag}_interrupted_path.npz',reference=ref,
                rows=np.vstack(chunks),actions=np.asarray(actions))
        result=dict(tag=tag,coefficients=np.asarray(coefficients).tolist(),results=records,
            candidate_complete=False,actor_calls=actor_calls,physics_ticks=physics_ticks,
            exception=traceback.format_exc(),incomplete_not_ranked=True)
        progress('preserved_exception')
        write_json(directory/f'{tag}_failure.json',result)
        return result
