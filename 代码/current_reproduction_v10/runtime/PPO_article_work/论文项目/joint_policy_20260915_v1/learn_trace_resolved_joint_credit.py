"""Learn joint parameters with signed physical traces before shape aggregation."""
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from geometric_turn_metrics import spatial_turn_profile
from complete_local_credit import complete_metric_response, state_function_features
from raw_state_joint_policy import RawStateJointPolicy
from optimize_complete_joint_function import initialize_worker, evaluate_job, residual
from learn_nonlinear_joint_feedback import promotion
from learn_coupled_feasibility_credit import build_geometric_response
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def signed_channels(reference, rows, nominal_grid=None, ticks=None):
    incoming, outgoing = np.diff(reference, axis=0)
    incoming /= np.linalg.norm(incoming)
    outgoing /= np.linalg.norm(outgoing)
    angle = np.arctan2(incoming[0]*outgoing[1]-incoming[1]*outgoing[0], incoming@outgoing)
    ni = np.sign(angle)*np.array([-incoming[1], incoming[0]])
    no = np.sign(angle)*np.array([-outgoing[1], outgoing[0]])
    profile = spatial_turn_profile(np.vstack([reference[0], rows[:, :2]]), .1)
    grid = profile['grid'][1:-1]
    turn = np.sign(angle)*profile['turn']
    if nominal_grid is not None:
        turn = np.interp(nominal_grid, grid, turn, left=0., right=0.)
    else:
        nominal_grid = grid
    if ticks is not None:
        # Complete trajectories only. A shorter completed path remains at its
        # actual final stopped state; no failed prefix is extended.
        rows = rows[np.minimum(ticks, len(rows)-1)]
    xy = rows[:, :2]-reference[1]
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    omega = np.sign(angle)*(rows[:, 2]*rows[:, 5]-rows[:, 3]*rows[:, 4])/np.maximum(speed, 1e-9)**2
    omega[speed <= 5.] = 0.
    return dict(incoming=xy@ni, outgoing=xy@no, omega=omega,
                turn=turn, grid=nominal_grid, angle=angle)


def read_rows(source, record):
    with np.load(source/record['trace']) as z:
        return z['reference'].copy(), z['rows'].copy()


def build_trace_response(source, current, paired):
    models = {}
    for index, r in enumerate(current['records']):
        if 'shape' not in r:
            continue
        ref, rows = read_rows(source, r)
        base = signed_channels(ref, rows)
        jac = {k: [] for k in ('incoming', 'outgoing', 'omega', 'turn')}
        for low, high in paired:
            low_ok = all(r['success'] for r in low['records'])
            high_ok = all(r['success'] for r in high['records'])
            lc = signed_channels(ref, read_rows(source, low['records'][index])[1],
                                 base['grid'], np.arange(len(rows))) if low_ok else None
            hc = signed_channels(ref, read_rows(source, high['records'][index])[1],
                                 base['grid'], np.arange(len(rows))) if high_ok else None
            for key in jac:
                response = ((hc[key]-lc[key])/2. if low_ok and high_ok else
                            hc[key]-base[key] if high_ok else
                            base[key]-lc[key] if low_ok else np.zeros_like(base[key]))
                jac[key].append(response)
        models[r['path']] = dict(base=base, jac={k: np.column_stack(v) for k, v in jac.items()})
    return models


def trace_deficits(u, base_deficits, metric_jac, models, current, retained):
    values = base_deficits+metric_jac@u
    gradient = metric_jac.copy()
    old = {r['path']: r for r in retained}
    offset = 0
    for record in current['records']:
        offset += 1  # success placeholder, identical to signed_deficits
        if 'shape' in record:
            m = models[record['path']]
            b, j = m['base'], m['jac']
            target = old[record['path']]['baseline_shape']
            for key, name, scale in [('incoming', 'in_edge_outward_mm', .002),
                                     ('outgoing', 'out_edge_outward_mm', .005)]:
                signal = b[key]+j[key]@u
                idx = int(signal.argmin())
                values[offset] = (max(-signal[idx], 0.)-target[name])/scale
                gradient[offset] = -j[key][idx]/scale if signal[idx] < 0. else 0.
                offset += 1
            signal = b['omega']+j['omega']@u
            factor = .001*180/np.pi
            values[offset] = (np.maximum(-signal, 0.).sum()*factor-target['full_physical_reverse_deg'])/.5
            gradient[offset] = -j['omega'][signal < 0.].sum(0)*factor/.5
            offset += 1
            turn = b['turn']+j['turn']@u
            factor = 1.25/(.1*b['angle']**2)
            values[offset] = ((turn@turn)*factor-target['full_normalized_energy'])/.02
            gradient[offset] = 2*turn@j['turn']*factor/.02
            offset += 1
            idx = int(abs(turn).argmax())
            values[offset] = (abs(turn[idx])/.1-target['full_spatial_peak_curvature_mm_inv'])/.2
            gradient[offset] = np.sign(turn[idx])*j['turn'][idx]/.02
            offset += 3  # peak curvature plus existing apex speed and first turn
        offset += 1  # complete time
    assert offset == len(values)-1
    return values, gradient


def main():
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_raw_state_joint_learning_20260919'
    data = read(source/'results.json')
    iteration = data['iterations'][-1]
    current = next(r for r in data['candidates'] if r['tag'] == iteration['previous'])
    paired = [(next(r for r in data['candidates'] if r['tag'] == f'iter1_axis{axis}_-1'),
               next(r for r in data['candidates'] if r['tag'] == f'iter1_axis{axis}_+1')) for axis in range(48)]
    assert all(all(r['success'] for r in c['records']) for pair in paired for c in pair)
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    spec = torch.load(source/'policy_spec.pt', map_location='cpu', weights_only=False)
    stage = register('canonical_gate_band_trace_resolved_joint_credit_20260919',
        [__file__, BASE/'complete_local_credit.py', BASE/'raw_state_joint_policy.py',
         BASE/'geometric_turn_metrics.py', BASE/'learn_coupled_feasibility_credit.py',
         BASE/'optimize_complete_joint_function.py', source/'results.json', source/'receipt.json',
         source/'policy_spec.pt'],
        dict(reused_complete_candidates=96, new_candidates=3, paths=20, workers=3, wall_s=240),
        'The coupled-corridor proposal was effectively identical to the axis-exclusion '
        'proposal because all96 local branches succeeded and corridor cuts were inactive. '
        'Do not interpret that result as evidence about failed-axis compensation. '
        'Test a different credit estimator at exactly the same center: central differences '
        'of signed actual XY half-plane positions, physical angular velocity at1ms, '
        'and spatial0.1mm turning profiles, followed by the original nonlinear max, '
        'hinge integral, energy and peak metrics. Preserve the original shape scales, '
        'time/cruise response, finite amplitude, function trust and physical executor. '
        'All source traces complete; stopped-state extension is permitted only after '
        'actual completion. New success is determined only by exact full closed-loop '
        'rollouts. This is complete-policy parameter optimization, not PPO, a safety '
        'proof, trajectory-template learning or deployed action switching. Compare '
        'against the already completed aggregate-metric proposals. Three fresh scales '
        'only; if the new model does not predict useful primary-preserving improvement, '
        'change representation/credit rather than expanding the same sweep. Adoption '
        'still requires every original retained shape condition and full64 validation.')
    started = time.perf_counter()
    subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
        '--active', stage.name, '--kind', 'complete_closed_loop_parameter_optimization',
        '--next', '逐时刻带符号运动响应建模后执行三个完整闭环候选；不采用探针或预测结果。'],
        cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    pool = None
    try:
        models = build_trace_response(source, current, paired)
        base, matrix, bounds, modes, _ = complete_metric_response(current, paired, retained)
        refs, support = {}, []
        for r in current['records']:
            with np.load(source/r['trace']) as z:
                refs[r['path']] = z['reference'].copy()
                support.append(z['observations'].copy())
        p = RawStateJointPolicy(spec['parent'], spec['basis'])
        features = state_function_features(p, np.vstack(support))
        steps = np.full(48, .125)
        trust = (features*steps[None, None, :]/np.array([.0025, .0125, .0125])[None, :, None]).reshape(-1, 48)
        margin, geometry, _ = build_geometric_response(source, current, paired)
        identity, _ = trace_deficits(np.zeros(48), base, matrix, models, current, retained)
        identity_error = float(abs(identity-base).max())
        assert identity_error < 1e-8, identity_error
        def objective(u):
            signed, jac = trace_deficits(u, base, matrix, models, current, retained)
            value = np.maximum(signed, 0.)
            return float(value@value+.01*(u@u)), 2*jac.T@value+.02*u
        # Analytic gradient check uses an interior generic direction, away from
        # zero constraints; it verifies the implemented optimization objective.
        rng = np.random.default_rng(20260919)
        q, direction = rng.normal(0, .02, 48), rng.normal(size=48)
        eps = 1e-7
        finite = (objective(q+eps*direction)[0]-objective(q-eps*direction)[0])/(2*eps)
        analytic = float(objective(q)[1]@direction)
        assert abs(finite-analytic) < 1e-4*max(1., abs(finite)), (finite, analytic)
        sol = minimize(objective, np.zeros(48), jac=True, method='SLSQP', bounds=bounds,
            constraints=[dict(type='ineq', fun=lambda u: np.r_[1-trust@u, 1+trust@u],
                              jac=lambda u: np.vstack([-trust, trust])),
                         dict(type='ineq', fun=lambda u: (margin-geometry@u)/.01,
                              jac=lambda u: -geometry/.01)],
            options=dict(maxiter=250, ftol=1e-8))
        step = sol.x
        assert abs(trust@step).max() <= 1.+1e-6 and (margin-geometry@step).min() >= -1e-7
        predicted = {}
        for scale in (.25, .5, 1.):
            v, _ = trace_deficits(scale*step, base, matrix, models, current, retained)
            predicted[str(scale)] = float(np.maximum(v, 0.)@np.maximum(v, 0.))
        audit = dict(current=current['tag'], baseline_objective=current['objective'],
            original_best_objective=73.72009482483934, solver_success=bool(sol.success),
            solver_message=sol.message, initial_metric_identity_error=identity_error,
            gradient_check=dict(finite=finite, analytic=analytic), predicted=predicted,
            normalized_step=step.tolist(), maximum_function_change=float(abs(trust@step).max()),
            minimum_linear_geometry_margin_mm=float((margin-geometry@step).min()))
        write_json(stage/'proposal.json', audit)
        np.savez_compressed(stage/'parameter_response.npz', signed_deficit=base, metric_jacobian=matrix,
                            step=step, function_trust=trust)
        print('TRACE_PROPOSAL', {k:v for k,v in audit.items() if k != 'normalized_step'}, flush=True)
        if not sol.success:
            raise RuntimeError('Signed-trace proposal did not converge: '+sol.message)
        pool = mp.get_context('spawn').Pool(3, initialize_worker,
            (stage, spec, refs, 'gate_band_canonical_kernel', started+235))
        results = []
        for result in pool.imap_unordered(evaluate_job,
                [(f'trace_{s}', np.asarray(current['coefficients'])+s*steps*step) for s in (.25, .5, 1.)]):
            result['objective'] = float(np.linalg.norm(residual(result, retained))**2)
            result['promotion'] = promotion(result['records'], retained)
            result['primary_preserved'] = all(result['promotion'][k]
                for k in ('all_complete', 'inward_first_turns', 'entry_nonworse'))
            result['predicted_objective'] = predicted[result['tag'].split('_')[-1]]
            results.append(result)
            write_json(stage/'results.json', dict(candidates=results, baseline=current['tag'],
                prior_best_objective=73.72009482483934, adopted=False, goal_complete=False))
            print('TRACE_CREDIT', result['tag'], result['objective'], result['summary'],
                  result['primary_preserved'], flush=True)
        pool.close(); pool.join(); pool = None
        usable = [r for r in results if r['primary_preserved'] and r['objective'] < 73.72009482483934]
        best = min(usable, key=lambda r:r['objective']) if usable else None
        if best:
            torch.save(dict(architecture='raw_state_joint_policy', policy_spec=spec,
                coefficients=np.asarray(best['coefficients']), environment_architecture='gate_band_canonical_kernel',
                learning_method='signed_trace_complete_policy_parameter_optimization'), stage/'research_candidate.pt')
            write_json(stage/'results.json', dict(candidates=results, best=best['tag'],
                baseline=current['tag'], prior_best_objective=73.72009482483934,
                adopted=False, goal_complete=False))
        receipt(stage, dict(status='completed_trace_resolved_parameter_learning',
            candidates=len(results), best=None if best is None else best['tag'],
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
    except Exception as exc:
        if pool is not None:
            pool.terminate(); pool.join()
        receipt(stage, dict(status='failed_trace_resolved_parameter_learning', error=str(exc),
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
            '--kind', 'complete_closed_loop_parameter_optimization_finished',
            '--next', '核对逐时刻响应与真实形状收益，继续学习三动作协调；保留模型及全部验收约束。'],
            cwd=BASE, check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    mp.freeze_support()
    main()
