"""Check recorded likelihoods and compare updates without collecting new data."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from historical_two_action_residual import TwoActionResidualPolicy
from learning_update_compact_mc import update as legacy_update, optimizers
from learning_update_guarded_mc import update as guarded_update, PostStepGuard
from evaluate_historical_dense import sha, write


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    source = root / 'paired_two_action_ppo_seed922302_pilot'
    stage = root / 'paired_update_guard_recorded_audit'
    stage.mkdir(exist_ok=False)
    started = time.perf_counter()
    checkpoint = source / 'initial.pt'
    batch_path = source / 'batch_0001.npz'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    data = np.load(batch_path)
    batch = {key: data[key] for key in ['observation', 'latent', 'log_probability',
        'value', 'reward', 'next_value', 'terminal']}
    write(stage/'registration.json', dict(kind='Recorded-batch optimizer diagnostic; no new simulation',
        initial_checkpoint=str(checkpoint), initial_sha256=sha(checkpoint),
        batch=str(batch_path), batch_sha256=sha(batch_path),
        fixed_shuffle_seed=922302, source_sha256={str(p): sha(p) for p in [
            Path(__file__), root/'learning_update_guarded_mc.py', root/'learning_update_compact_mc.py']},
        adoption='Likelihood consistency and post-step KL bound, including optimizer rollback, must pass before live PPO',
        wall_budget_s=120, goal_complete=False))
    results = {}
    for name, update in [('legacy', legacy_update), ('guarded', guarded_update)]:
        policy = TwoActionResidualPolicy()
        policy.load_state_dict(saved['policy'])
        optimizer = optimizers(policy)
        for key in optimizer:
            optimizer[key].load_state_dict(copy.deepcopy(saved['optimizers'][key]))
        obs = torch.as_tensor(batch['observation'], dtype=torch.float32)
        latent = torch.as_tensor(batch['latent'], dtype=torch.float32)
        old_log = torch.as_tensor(batch['log_probability'], dtype=torch.float32)
        meter = PostStepGuard(policy, optimizer['actor'], obs, latent, old_log)
        with torch.no_grad():
            difference = (policy.distribution(obs).log_prob(latent).sum(-1)-old_log).abs().numpy()
        assert difference.max() < 1e-3
        torch.manual_seed(922302)
        statistics = update(policy, optimizer, batch)
        exact, sampled = meter.measure()
        results[name] = dict(log_probability_max_error=float(difference.max()),
            log_probability_error_quantiles=np.quantile(difference,[.5,.9,.99]).tolist(),
            post_update_exact_kl=exact, post_update_sampled_kl=sampled, statistics=statistics)
    # Force rejection on a controlled differentiable input; ensure Adam moments
    # and parameters are unchanged, not merely the public learning rate.
    policy = TwoActionResidualPolicy()
    policy.load_state_dict(saved['policy'])
    optimizer = optimizers(policy)['actor']
    optimizer.load_state_dict(copy.deepcopy(saved['optimizers']['actor']))
    with torch.no_grad():
        old_log = policy.distribution(obs).log_prob(latent).sum(-1)
    guard = PostStepGuard(policy, optimizer, obs, latent, old_log, limit=0., max_backtracks=0)
    policy.actor(obs).sum().backward()
    parameters = [p.detach().clone() for p in guard.parameters]
    state = copy.deepcopy(optimizer.state_dict())
    guard.step()
    def equal(a,b):
        if torch.is_tensor(a): return torch.equal(a,b)
        if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    rollback_pass = (guard.rejected==1 and
        all(torch.equal(a,b) for a,b in zip(parameters,guard.parameters)) and equal(state,optimizer.state_dict()))
    assert rollback_pass
    assert max(results['guarded']['post_update_exact_kl'],results['guarded']['post_update_sampled_kl']) <= .03
    result = dict(results=results, optimizer_and_parameter_rollback_pass=rollback_pass,
        new_simulation_steps=0, new_policy_deployed=False, goal_complete=False)
    write(stage/'result.json', result)
    write(stage/'receipt.json', dict(status='finished',exit_code=0,wall_s=time.perf_counter()-started,goal_complete=False))
    print(json.dumps({name:{k:v for k,v in record.items() if k!='statistics'} for name,record in results.items()},indent=2))
    print('rollback_pass',rollback_pass)


if __name__ == '__main__':
    main()
