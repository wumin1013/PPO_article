"""Value-informed offline joint fitting and independently measured full-path checks."""
import copy
import time
import numpy as np
import torch
from nonlinear_joint_adapter import NonlinearJointAdapter
from nonlinear_feed_adapter import NonlinearFeedAdapter
from canonical_gate_state import checkpoint_environment_factory
from analyze_nonlinear_joint_response import measure, AMPLITUDE
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def shape_cost(s):
    return s['full_physical_reverse_deg']/5.+s['out_edge_outward_mm']/.05+s['full_normalized_energy']


def rollout(policy, factory, ref, costs, deadline):
    env = factory(ref)
    chunks, actions, observations = [], [], []
    while not env.done:
        if time.perf_counter() >= deadline:
            raise TimeoutError('Registered joint-learning wall budget')
        x = env.observation()
        latent = policy.act(x, True)[0]
        _, _, _, info = env.step(latent)
        observations.append(x)
        actions.append(info['held_action'])
        chunks.append(info['rows'])
        costs['actor_calls'] += 1
        costs['physics_ticks'] += len(info['rows'])
    rows = np.vstack(chunks)
    return measure(ref, rows, env), rows, np.asarray(actions), np.asarray(observations)


def aggregate(records):
    successful = all(r['success'] for r in records)
    shapes = [r for r in records if 'shape' in r]
    return dict(paths=len(records), successes=sum(r['success'] for r in records),
        complete_time_s=sum(r['complete_time_s'] for r in records) if successful else None,
        curved_cruise99_s=sum(r['motion']['cruise_ge99_s'] for r in records
                            if r['path'] not in (48, 57)) if successful else None,
        reverse_sum_deg=sum(r['shape']['full_physical_reverse_deg'] for r in shapes) if successful else None,
        exit_sum_mm=sum(r['shape']['out_edge_outward_mm'] for r in shapes) if successful else None,
        inward_first_turns=sum(r['target_motion']['first_turn_inside'] is True for r in shapes),
        energy_sum=sum(r['shape']['full_normalized_energy'] for r in shapes) if successful else None)


def promotion(records, retained):
    corners = [r for r in records if 'shape' in r]
    old = {r['path']: r['baseline_shape'] for r in retained if 'baseline_shape' in r}
    # Screening, never final scientific acceptance. No shape tradeoff hidden in cost.
    return dict(all_complete=all(r['success'] for r in records),
        inward_first_turns=all(r['target_motion']['first_turn_inside'] for r in corners),
        entry_nonworse=all(r['shape']['in_edge_outward_mm'] <= old[r['path']]['in_edge_outward_mm']+1e-6 for r in corners),
        exit_nonworse=all(r['shape']['out_edge_outward_mm'] <= old[r['path']]['out_edge_outward_mm']+1e-6 for r in corners),
        reverse_nonworse=all(r['shape']['full_physical_reverse_deg'] <= old[r['path']]['full_physical_reverse_deg']+1e-5 for r in corners),
        energy_nonworse=all(r['shape']['full_normalized_energy'] <= old[r['path']]['full_normalized_energy']+1e-6 for r in corners),
        apex_speed_nonworse=all(r['shape']['apex_speed_mm_s'] >= old[r['path']]['apex_speed_mm_s']-1e-4 for r in corners),
        curved_cruise_present=sum(r['motion']['cruise_ge99_s'] for r in records if r['success'] and r['path'] not in (48,57)) > 0.)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260919)
    source = BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919'
    probe = BASE/'canonical_gate_band_nonlinear_joint_response_20260919'
    model = BASE/'canonical_gate_band_nonlinear_cruise_capacity_20260918/diagnostic_policy.pt'
    baseline = read(source/'results.json')['records']
    evidence = read(probe/'results.json')
    stage = register('canonical_gate_band_nonlinear_joint_feedback_20260919',
        [__file__, BASE/'nonlinear_joint_adapter.py', BASE/'analyze_nonlinear_joint_response.py',
         model, probe/'results.json', probe/'queries.npz', probe/'receipt.json',
         source/'results.json', source/'receipt.json'] + [source/r['trace'] for r in baseline],
        dict(arms=['joint', 'feed_only'], fitting_updates_per_arm=600,
             evaluation_scales=[.5, 1., 2.], paths=[r['path'] for r in baseline],
             wall_s=240, actor_calls=15000, physics_ticks=400000),
        'Offline value-informed regression, NOT PPO: jointly fit three latent-action '
        'residual outputs to the best physically measured complete-consequence offset '
        'at each of12 current states. All12 zero interventions remain feasible choices. '
        'Use existing continuous observable inputs and reflection equivariance, no '
        'path/angle/tick template, online label, switch or limiter. Anchor correction '
        'on all20 saved complete trajectories and compare a matched feed-only output '
        'mask with identical evidence, optimizer, architecture and three scales. '
        'Record actual closed-loop trajectories, not just fitting errors. Promotions '
        'must preserve each retained86cd corner entry/exit/reversal/energy/apex speed '
        'and completion plus actual curved cruise; otherwise retain86cd and switch '
        'from sparse local regression to measured complete-policy credit. No adoption '
        'from offline fit, no repeated completed experiments or sealed tests.')
    start = time.perf_counter()
    costs = dict(actor_calls=0, physics_ticks=0, fitting_updates=0)
    md = torch.load(model, map_location='cpu', weights_only=False)
    original = NonlinearFeedAdapter()
    original.load_state_dict(md['policy'])
    factory = checkpoint_environment_factory(md['environment_architecture'])
    with np.load(probe/'queries.npz') as z:
        qobs = torch.tensor(z['observations'])
    targets, evidence_rows = [], []
    for q in evidence['queries']:
        options = [r for r in evidence['records'] if r['query'] == q['query'] and r['success']]
        best = min(options, key=lambda r: shape_cost(r['shape']))
        improvement = shape_cost(q['baseline_shape'])-shape_cost(best['shape'])
        delta = np.zeros(3)
        if improvement > 0:
            if 'head' in best:
                delta[best['head']] = AMPLITUDE[best['head']]*best['direction']
            else:
                delta = np.asarray(best['offset'])
        targets.append(delta/AMPLITUDE)
        evidence_rows.append(dict(query=q['query'], selected_trace=best['trace'],
                                  improvement=improvement, normalized_target=(delta/AMPLITUDE).tolist()))
    supports, refs = [], {}
    for r in baseline:
        with np.load(source/r['trace']) as z:
            supports.append(z['observations'].copy())
            refs[r['path']] = z['reference'].copy()
    support = torch.tensor(np.vstack(supports))
    targets = torch.tensor(np.asarray(targets), dtype=torch.float32)
    write_json(stage/'training_evidence.json', dict(targets=evidence_rows,
        query_states=len(qobs), anchor_states=len(support), source_model_sha256=sha(model)))
    outcomes, histories = [], {}
    try:
        for arm in ('joint', 'feed_only'):
            torch.manual_seed(20260919)
            p = NonlinearJointAdapter()
            p.initialize_from_feed(md['policy'])
            mask = torch.tensor([1., 1., 1.] if arm == 'joint' else [0., 1., 0.])
            with torch.no_grad():
                assert torch.equal(p.distribution(support).mean, original.distribution(support).mean)
            for parameter in p.parameters():
                parameter.requires_grad_(False)
            for parameter in p.joint_adapter.parameters():
                parameter.requires_grad_(True)
            optimizer = torch.optim.Adam(p.joint_adapter.parameters(), lr=.001)
            history = []
            for step in range(600):
                pred = p.correction(qobs)/p.joint_scale
                anchor = p.correction(support)/p.joint_scale
                fit_loss = (pred-targets*mask).square().mean()
                anchor_loss = anchor.square().mean()
                loss = fit_loss+.2*anchor_loss+2.*torch.relu(abs(anchor)-1.).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if arm == 'feed_only':
                    with torch.no_grad():
                        p.joint_adapter[-1].weight[[0, 2]] = 0
                        p.joint_adapter[-1].bias[[0, 2]] = 0
                costs['fitting_updates'] += 1
                if step % 100 == 0 or step == 599:
                    history.append(dict(step=step, fit=float(fit_loss.detach()), anchor=float(anchor_loss.detach())))
            histories[arm] = history
            learned_state = copy.deepcopy(p.state_dict())
            for scale in (.5, 1., 2.):
                p.load_state_dict(learned_state)
                with torch.no_grad():
                    p.joint_scale.mul_(scale)
                tag = f'{arm}_{scale}'
                torch.save(dict(policy=p.state_dict(), architecture='nonlinear_joint_adapter',
                    environment_architecture=md['environment_architecture'], parent_sha256=sha(model),
                    method='offline_complete_consequence_regression', arm=arm, scale=scale), stage/f'{tag}.pt')
                records = []
                folder = stage/tag
                folder.mkdir()
                for old in baseline:
                    path = old['path']
                    r, rows, actions, observations = rollout(p, factory, refs[path], costs, start+235)
                    np.savez_compressed(folder/f'path{path}.npz', reference=refs[path], rows=rows,
                                        actions=actions, observations=observations)
                    r.update(path=path, trace=f'{tag}/path{path}.npz')
                    records.append(r)
                outcome = dict(arm=arm, scale=scale, model=f'{tag}.pt', records=records,
                               summary=aggregate(records), promotion=promotion(records, baseline))
                outcomes.append(outcome)
                write_json(stage/'results.json', dict(candidates=outcomes, fitting=histories,
                    costs=costs, adopted=False, goal_complete=False))
                print('JOINT_FEEDBACK', arm, scale, outcome['summary'], outcome['promotion'], flush=True)
        receipt(stage, dict(status='completed_offline_joint_counterfactual_learning', costs=costs,
            candidates=len(outcomes), eligible=sum(all(r['promotion'].values()) for r in outcomes),
            wall_s=time.perf_counter()-start, adopted=False, goal_complete=False))
    except Exception as exc:
        receipt(stage, dict(status='failed_offline_joint_counterfactual_learning', error=str(exc),
                           costs=costs, wall_s=time.perf_counter()-start, adopted=False, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
