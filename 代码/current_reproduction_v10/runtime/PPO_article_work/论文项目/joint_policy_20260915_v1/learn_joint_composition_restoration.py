"""Learn joint phase/action amplitudes using responses at the measured joint point."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from duration_joint_metrics import guards, penalty
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from audit_context_restoration_scaling import solve_correction
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source = BASE / 'canonical_gate_band_coupled_context_composition_learning_20260920'
    assert not read(source / 'receipt.json')['research_acceptable']
    parent_file = BASE / 'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    md = torch.load(source / 'state_joint/candidate.pt', map_location='cpu', weights_only=False)
    history = copy.deepcopy(read(source / 'state_joint/results.json')['candidates'])
    anchor = next(c for c in history if c['tag'] == 'context_joint_46')
    joint = np.asarray(anchor['coefficients']).reshape(3, 3, 2, 3)
    vectors = []
    for kernel in (0, 1):
        for head in range(3):
            v = np.zeros_like(joint)
            v[:, :, kernel, head] = joint[:, :, kernel, head]
            vectors.append(v.reshape(-1))
    basis = np.asarray(vectors).T
    assert np.linalg.matrix_rank(basis) == 6
    stage = register('canonical_gate_band_measured_joint_amplitude_restoration_20260920',
        [__file__, BASE/'duration_joint_metrics.py', BASE/'observable_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py', BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py', BASE/'audit_context_restoration_scaling.py',
         parent_file, source/'receipt.json', source/'state_joint/results.json',
         source/'state_joint/candidate.pt', source/'actual_guard_attribution.json'],
        dict(wall_s=240, workers=4, axis_responses=12, maximum_joint_corrections=6,
             learned_amplitudes=6, axis_amplitude=.25, original_coefficient_bounds=[-1., 1.]),
        'Six same-parent component controls show a useful actual loss decrease, but '
        'small reverse/energy/speed regressions remain. Learn six phase-by-action '
        'amplitudes from central complete-loop responses around the actual JOINT '
        'candidate, rather than extrapolating zero-centered single-phase credit. '
        'Reuse all50 vectors. Measure12 symmetric responses at amplitude0.25; solve '
        'row-equilibrated nonnegative-guard corrections with at most6 validations. '
        'Stop this representation if its local feasible correction is absent or '
        'actual violation does not improve. Keep all real acceptance bounds, '
        'original64, physical limits and duration guards. No runtime phase switches, '
        'no feed labels, no PPO update and no generalization claim.')
    started = time.perf_counter(); deadline = started + 220
    folder = stage/'state_joint'; folder.mkdir()
    data = read(source/'baseline.json'); baseline = data['baseline']['records']
    support = read(source/'support_audit.json'); choice = support['selected']; paths = choice['paths']
    selected = [r for r in baseline if r['path'] in paths]
    write_json(stage/'baseline.json', data); write_json(stage/'support_audit.json', support)
    with np.load(source/'replayed_states.npz') as z:
        np.savez_compressed(stage/'replayed_states.npz', **{k: z[k] for k in z.files})
    _, old_guards = metric_functions(data, selected, stage, md['parent'], deadline)
    retained = [next(r['baseline_shape'] for r in data['retained'] if r['path'] == b['path']) for b in selected]
    def score(records):
        return float(sum(teacher_score(r, b, o, barrier=False)+penalty(r)
                         for r, b, o in zip(records, selected, retained)))
    def constraints(records):
        return np.concatenate([old_guards(records)]+[guards(r, b) for r, b in zip(records, selected)])
    for c in history:
        assert abs(c['score']-score(c['records'])) < 1e-10
        assert np.array_equal(c['measured_inequalities'], constraints(c['records']))
    zero = history[0]; md['reused_candidates'] = len(history)
    cache = {tuple(np.round(c['coefficients'], 12)): c for c in history}
    def feasible(c):
        return parameters_in_bounds(c['coefficients'], -1., 1.) and min(c['measured_inequalities']) >= -1e-8
    def best():
        return min(filter(feasible, history), key=lambda c: c['score'])
    blocks = []; pool = None
    print('MEASURED_JOINT_START', anchor['tag'], len(history), flush=True)
    try:
        pool = mp.get_context('spawn').Pool(4, worker_initialize, (md, selected, folder, deadline))
        def evaluate(u):
            key = tuple(np.round(u, 12))
            if key not in cache:
                assert parameters_in_bounds(u, -1., 1.)
                tag = f'joint_recenter_{len(history)}'
                records = pool.map(worker_evaluate, [(p, u, tag) for p in paths])
                c = dict(tag=tag, coefficients=np.asarray(u).tolist(), records=records,
                         score=score(records), measured_inequalities=constraints(records).tolist(),
                         actual_rerun_paths=paths, exact_zero_support_paths=[])
                history.append(c); cache[key] = c
                write_json(folder/'results.json', dict(candidates=history, best=best()['tag']))
                print('MEASURED_JOINT', tag, c['score'], min(c['measured_inequalities']), flush=True)
            return cache[key]
        u0 = np.asarray(anchor['coefficients']); jac = []; gradient = []; pairs = []
        for i in range(6):
            plus = evaluate(u0+.25*basis[:, i]); minus = evaluate(u0-.25*basis[:, i])
            jac.append((np.asarray(plus['measured_inequalities'])-minus['measured_inequalities'])/.5)
            gradient.append((plus['score']-minus['score'])/.5)
            pairs.append(dict(index=i, plus=plus['tag'], minus=minus['tag']))
        jac = np.asarray(jac).T; gradient = np.asarray(gradient)
        write_json(stage/'measured_joint_credit.json', dict(anchor=anchor['tag'], basis=basis.tolist(),
            pairs=pairs, amplitude_step=.25, central_constraints=jac.tolist(), central_loss=gradient.tolist()))
        current = anchor
        for iteration in range(6):
            values = np.asarray(current['measured_inequalities'])
            correction = solve_correction(jac, gradient, values, zero['score']-.001-current['score'], radius=1.)
            block = dict(iteration=iteration, anchor=current['tag'], **correction)
            if not correction['success']:
                blocks.append(block); break
            delta = np.asarray(correction['scaled_parameter_step'])
            proposal = evaluate(np.asarray(current['coefficients'])+basis@delta)
            block.update(tag=proposal['tag'], actual_minimum=min(proposal['measured_inequalities']),
                         actual_loss=proposal['score'])
            blocks.append(block); write_json(stage/'optimization_blocks.json', blocks)
            if feasible(proposal):
                break
            old_violation = np.linalg.norm(np.minimum(values, 0.))
            new_violation = np.linalg.norm(np.minimum(proposal['measured_inequalities'], 0.))
            if new_violation >= old_violation:
                break
            current = proposal
        pool.close(); pool.join(); pool = None
        finish(stage, md, baseline, data, best(), zero, history, blocks, choice, parent_file, started)
    except TimeoutError:
        if pool is not None: pool.terminate(); pool.join(); pool = None
        finish(stage, md, baseline, data, best(), zero, history, blocks, choice, parent_file, started, budget_exhausted=True)
    except Exception as exc:
        if pool is not None: pool.terminate(); pool.join()
        receipt(stage, dict(status='failed_measured_joint_amplitude_restoration', error=str(exc),
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False)); raise


if __name__ == '__main__':
    mp.freeze_support(); main()
