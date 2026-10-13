"""Reuse full-policy evidence to test joint compensation versus sign exclusion."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from learn_full_pool_joint_credit import initialize, evaluate, candidate_metadata
from learn_trace_resolved_joint_credit import signed_channels, read_rows, trace_deficits
from learn_retained_joint_growth import growth_deficits
from learn_absolute_shape_joint_credit import absolute_shape_response
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def consistent_response(source, baseline, retained, paired):
    """Use exactly the same eligible sides for metrics, margins and traces."""
    base = growth_deficits(baseline, retained)
    bm = np.array([r['corridor']['margin_lower_mm'] for r in baseline['records']])
    modes, columns, margins, bounds = [], [], [], []
    for lo, hi in paired:
        lok, hik = lo['primary_preserved'], hi['primary_preserved']
        mode = 'central' if lok and hik else 'positive' if hik else 'negative' if lok else 'frozen'
        modes.append(mode)
        def diff(a, b, c):
            return (c-b)/2 if mode == 'central' else c-a if mode == 'positive' else a-b if mode == 'negative' else np.zeros_like(a)
        lv = growth_deficits(lo, retained) if lok else base
        hv = growth_deficits(hi, retained) if hik else base
        columns.append(diff(base, lv, hv))
        lm = np.array([r['corridor']['margin_lower_mm'] for r in lo['records']])
        hm = np.array([r['corridor']['margin_lower_mm'] for r in hi['records']])
        margins.append(diff(bm, lm, hm))
        bounds.append((-1., 1.) if mode == 'central' else (0., 1.) if mode == 'positive' else (-1., 0.) if mode == 'negative' else (0., 0.))
    models = {}
    for index, r in enumerate(baseline['records']):
        if 'shape' not in r:
            continue
        ref, rows = read_rows(source, r)
        b = signed_channels(ref, rows)
        jac = {k: [] for k in ('incoming', 'outgoing', 'omega', 'turn')}
        for mode, (lo, hi) in zip(modes, paired):
            lc = signed_channels(ref, read_rows(source, lo['records'][index])[1], b['grid'], np.arange(len(rows))) if mode in ('central', 'negative') else b
            hc = signed_channels(ref, read_rows(source, hi['records'][index])[1], b['grid'], np.arange(len(rows))) if mode in ('central', 'positive') else b
            for k in jac:
                jac[k].append((hc[k]-lc[k])/2 if mode == 'central' else hc[k]-b[k] if mode == 'positive' else b[k]-lc[k] if mode == 'negative' else np.zeros_like(b[k]))
        models[r['path']] = dict(base=b, jac={k: np.column_stack(v) for k, v in jac.items()})
    return base, np.column_stack(columns), bm, np.column_stack(margins), bounds, modes, models


def solve_response(baseline, retained, response_data, trust, compensate, deadline, aggregate_guards=False,
                   additional_constraints=(), scalar_response=False):
    base, matrix, bm, margin_jac, bounds, modes, traces = response_data
    n = matrix.shape[1]
    if compensate:
        bounds = [(-1., 1.) if mode != 'frozen' else (0., 0.) for mode in modes]
    guards, time_indices, offset = [], [], 0
    for r in baseline['records']:
        if 'shape' in r:
            guards.extend(offset+k for k in range(1, 8))
            time_indices.append(offset+8)
            offset += 9
        else:
            time_indices.append(offset+1)
            offset += 2
    guards = np.array(guards)
    upper = np.maximum(base[guards], 0.)
    normalization = max(baseline['absolute_objective'], 1.)
    def response(u):
        if scalar_response:
            return base+matrix@u,matrix
        return trace_deficits(u, base, matrix, traces, baseline, retained)
    def fun(u):
        v, j = response(u)
        v, j = absolute_shape_response(v, j, baseline['records'], retained)
        pos = np.maximum(v, 0.)
        return float(pos@pos/normalization+.001*(u@u)), 2*j.T@pos/normalization+.002*u
    time_jac = .02*matrix[time_indices].sum(0)
    constraints = [
        dict(type='ineq', fun=lambda u: np.r_[2.-trust@u, 2.+trust@u], jac=lambda u: np.vstack([-trust, trust])),
        dict(type='ineq', fun=lambda u: (bm+margin_jac@u-np.minimum(bm, .02))/.01, jac=lambda u: margin_jac/.01),
        dict(type='ineq', fun=lambda u: upper-response(u)[0][guards], jac=lambda u: -response(u)[1][guards]),
        dict(type='ineq', fun=lambda u: np.array([.02-time_jac@u, .1-matrix[-1]@u]), jac=lambda u: np.vstack([-time_jac, -matrix[-1]]))]
    if aggregate_guards:
        groups = [np.array([9*p+k for p in range(16)]) for k in (2, 3, 4)]
        constraints.append(dict(type='ineq',
            fun=lambda u: np.array([base[g].sum()-response(u)[0][g].sum() for g in groups]),
            jac=lambda u: np.vstack([-response(u)[1][g].sum(0) for g in groups])))
    constraints.extend(additional_constraints)
    identity_error = float(abs(response(np.zeros(n))[0]-base).max())
    assert identity_error < 1e-7, identity_error
    rng = np.random.default_rng(20260919)
    q, d = rng.normal(0, .01, n), rng.normal(size=n)
    eps = 1e-7
    finite = (fun(q+eps*d)[0]-fun(q-eps*d)[0])/(2*eps)
    analytic = fun(q)[1]@d
    assert abs(finite-analytic) < 1e-4*max(abs(finite), 1.)
    best = np.zeros(n)
    best_value = fun(best)[0]
    def constraint_margin(c, u):
        residual = c['fun'](u)
        return -float(np.max(abs(residual))) if c['type'] == 'eq' else float(np.min(residual))
    def remember(u):
        nonlocal best, best_value
        if all(constraint_margin(c, u) >= -1e-8 for c in constraints) and fun(u)[0] < best_value:
            best, best_value = u.copy(), fun(u)[0]
        if time.perf_counter() > deadline:
            raise TimeoutError('Bounded optimizer budget')
    try:
        sol = minimize(fun, best, jac=True, method='SLSQP', bounds=bounds, constraints=constraints,
                       callback=remember, options=dict(maxiter=250, ftol=1e-10))
        remember(sol.x)
        converged, message = bool(sol.success), str(sol.message)
    except TimeoutError as exc:
        converged, message = False, str(exc)
    minima = [constraint_margin(c, best) for c in constraints]
    return best, dict(compensate=compensate, aggregate_shape_guards=aggregate_guards,
        scalar_response=scalar_response, converged=converged, message=message,
        response_modes=modes, parameters=best.tolist(), base_objective=normalization,
        predicted_objective=fun(best)[0]*normalization,
        maximum_function_change=float(abs(trust@best).max()), constraint_minima=minima,
        identity_error=identity_error, gradient_check=dict(finite=finite, analytic=float(analytic)),
        predicted_half_objective=fun(.5*best)[0]*normalization,
        predicted_margin_by_path=(bm+margin_jac@best).tolist(),
        complete_time_change_s=float(time_jac@best),
        predicted_signed_deficits=response(best)[0].tolist())


def progress(stage, active, next_action):
    subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
        '--active', stage.name if active else '', '--kind',
        'complete_closed_loop_parameter_optimization' if active else 'complete_closed_loop_parameter_optimization_finished',
        '--next', next_action], check=True, stdout=subprocess.DEVNULL)


def main():
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_full_pool_joint_credit_learning_20260919'
    parent = BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    assert sha(parent) == '5ab47111bfcec3a511903e20f1249c91bbfb65e5ab4cd1a3d2b4a493a979c26a'
    assert read(source/'receipt.json')['status'] == 'completed_full_pool_joint_credit_learning'
    data = read(source/'baseline.json')
    baseline, retained = data['baseline'], data['retained']
    candidates = {c['tag']: c for c in read(source/'results.json')['candidates']}
    with np.load(source/'parameter_space.npz') as z:
        directions, trust = z['directions'].copy(), z['trust'].copy()
    paired = [(candidates[f'axis{i}_-1'], candidates[f'axis{i}_+1']) for i in range(len(directions))]
    metadata = torch.load(parent, map_location='cpu', weights_only=False)
    stage = register('canonical_gate_band_full_pool_compensation_learning_20260919',
        [__file__, BASE/'learn_full_pool_joint_credit.py', BASE/'learn_trace_resolved_joint_credit.py',
         BASE/'learn_absolute_shape_joint_credit.py', BASE/'learn_nonlinear_joint_feedback.py',
         source/'results.json', source/'receipt.json', source/'baseline.json', source/'parameter_space.npz', parent],
        dict(wall_s=360, optimizer_s_per_arm=90, reused_full64_candidates=16,
             new_full64_candidates=2, workers=2, centers=114, new_network_layers=0),
        'Test whether excluding one sign after an individual full-policy failure prevents joint compensation. '
        'Reuse all16 actual full64 response candidates, without rerunning. Use primary-preserving complete '
        'sides consistently for scalar, margin and trace credit; axis5 positive completed but entry failed, '
        'so it is excluded consistently from these response estimates. Match two offline solvers differing '
        'only in sign bounds. The compensation solver can extrapolate a complete one-sided derivative in '
        'both signs, never inventing failed-prefix complete metrics. Freeze dimensions without a usable '
        'side. Keep path margins, all corner entry/exit/reversal/energy/curvature/apex/first-turn guards, '
        'training-state action trust, time and cruise guards. At most half/full actual full64 candidates. '
        'Extrapolated predictions require full physical validation; no PPO, runtime rule or final adoption. '
        'If no actual acceptable gain, use active constraint evidence to change the parameter subspace.')
    started = time.perf_counter()
    pool, outcomes = None, []
    progress(stage, True, '复用16个完整64路径对照，检验联合补偿与单边排除；最多两条参数候选。')
    try:
        audit = []
        for c in candidates.values():
            audit.append(dict(tag=c['tag'], promotion=c['promotion'],
                failures=[dict(path=r['path'], margin=r['corridor']['margin_lower_mm']) for r in c['records'] if not r['success']],
                entry_regressions=[dict(path=r['path'], entry_mm=r['shape']['in_edge_outward_mm']) for r in c['records']
                    if 'shape' in r and r['shape']['in_edge_outward_mm'] > next(x for x in retained if x['path']==r['path'])['baseline_shape']['in_edge_outward_mm']+1e-6]))
        write_json(stage/'reused_evidence_audit.json', audit)
        response_data = consistent_response(source, baseline, retained, paired)
        proposals = []
        for arm in (False, True):
            step, report = solve_response(baseline, retained, response_data, trust, arm, min(started+230, time.perf_counter()+90))
            proposals.append(step)
            write_json(stage/('compensation_optimizer.json' if arm else 'matched_exclusion_optimizer.json'), report)
            print('COMPENSATION_SOLVER', arm, report['message'], report['predicted_objective'], report['maximum_function_change'], flush=True)
        step = proposals[-1]
        refs, curved = {}, {}
        for r in baseline['records']:
            with np.load(r['trace']) as z:
                refs[r['path']] = z['reference'].copy()
            curved[r['path']] = r['curved_path']
        jobs = [(f'compensation_joint_{s}', s*step) for s in (.5, 1.)] if np.max(abs(step)) > 1e-6 else []
        for tag, u in jobs:
            torch.save(candidate_metadata(metadata, directions, u), stage/(tag+'.pt'))
        if jobs:
            pool = mp.get_context('spawn').Pool(2, initialize, (stage, metadata, directions, refs, curved, retained, started+350))
            for c in pool.imap_unordered(evaluate, jobs):
                outcomes.append(c)
                write_json(stage/'results.json', dict(candidates=outcomes, adopted=False, goal_complete=False))
                print('COMPENSATION_ACTUAL', c['tag'], c['summary'], c['promotion'], flush=True)
            pool.close(); pool.join(); pool = None
        # Research selection preserves each previously attained shape guard.
        base = response_data[0]
        good = []
        for c in outcomes:
            if not c['primary_preserved']:
                continue
            v = growth_deficits(c, retained)
            guard = np.array([9*p+k for p in range(16) for k in range(1, 8)])
            c['research_corner_guards_pass'] = bool(np.all(v[guard] <= np.maximum(base[guard], 0.)+1e-5))
            b, a = baseline['summary'], c['summary']
            c['research_acceptable'] = bool(c['research_corner_guards_pass'] and c['absolute_objective'] < baseline['absolute_objective']
                and a['complete_time_s'] <= b['complete_time_s']+.02 and a['curved_cruise99_s'] >= b['curved_cruise99_s']-.05
                and a['reverse_sum_deg'] < b['reverse_sum_deg'] and a['exit_sum_mm'] <= b['exit_sum_mm']
                and a['energy_sum'] <= b['energy_sum'])
            if c['research_acceptable']:
                good.append(c)
        selected = min(good, key=lambda c:c['absolute_objective']) if good else None
        result = dict(candidates=outcomes, adopted=False, goal_complete=False)
        if selected:
            result['best'] = selected['tag']
            torch.save(candidate_metadata(metadata, directions, selected['parameters']), stage/'research_candidate.pt')
        write_json(stage/'results.json', result)
        receipt(stage, dict(status='completed_full_pool_compensation_learning', full64_candidates=len(outcomes),
            best=selected['tag'] if selected else None, reused_full64_candidates=16,
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
    except Exception as exc:
        if pool is not None:
            pool.terminate(); pool.join()
        receipt(stage, dict(status='failed_full_pool_compensation_learning', error=str(exc),
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
        raise
    finally:
        progress(stage, False, '依据联合补偿实测结果和活跃约束继续形状学习；保留原模型，不进入封存测试或论文。')


if __name__ == '__main__':
    mp.freeze_support()
    main()
