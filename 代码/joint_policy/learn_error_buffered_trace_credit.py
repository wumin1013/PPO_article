"""Epigraph joint-policy learning with empirically buffered trajectory credit."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
import learn_stratified_suffix_joint_credit as suffix
from dense_trace_credit_model import build_model, values, objective, coefficients, coordinates
from learn_absolute_shape_joint_credit import absolute_shape_response, absolute_actual
from learn_retained_joint_growth import growth_deficits
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def layout(model):
    indices, paths, offset = [], [], 0
    for row in model['baseline']['records']:
        if 'shape' in row:
            paths.append((row['path'], offset))
            indices.extend(offset + k for k in range(1, 6))
            offset += 9
        else:
            offset += 2
    return np.asarray(indices), paths


def actual_guard_audit(model, candidate):
    v = growth_deficits(candidate, model['retained'])
    _, paths = layout(model)
    names = ['out_edge_outward_mm', 'full_physical_reverse_deg',
             'full_normalized_energy', 'full_spatial_peak_curvature_mm_inv', 'apex_speed_mm_s']
    scale = [.005, .5, .02, .2, 2.]
    violations = []
    for path, offset in paths:
        for k, name, s in zip(range(2, 7), names, scale):
            excess = (v[offset+k] - max(model['base'][offset+k], 0.)) * s
            if excess > 1e-7:
                violations.append(dict(path=path, metric=name, signed_excess=float(excess)))
    return dict(violations=violations, guard_pass=not violations)


def solve_epigraph(model, start, scale, buffer, deadline):
    """Use linear sampled cuts for max metrics; verify the full grid afterwards."""
    idx, paths = layout(model)
    n = len(start)
    count = len(idx)
    anchor = model['anchors']
    normalization = max(objective(model, np.zeros(n))[0], 1.)
    diagonal = np.maximum(np.sqrt((model['metric_jac']**2).sum(0)/normalization
                                  + .04*(anchor**2).mean(0)), 1e-4)
    a = anchor / diagonal
    active = model['margins'] < .05
    mg = model['margin_jac'][active]/diagonal
    mb = model['margins'][active]
    guard_idx = np.array([off+k for _, off in paths for k in range(2, 7)])
    upper = np.maximum(model['base'][guard_idx], 0.) - buffer[guard_idx]
    old = {r['path']: r['baseline_shape'] for r in model['retained'] if 'baseline_shape' in r}
    aux = {int(k): n+j for j, k in enumerate(idx)}
    surrogate_jac = np.zeros((len(model['base']), n+count))
    surrogate_jac[:, :n] = model['metric_jac']/diagonal
    surrogate_jac[idx] = 0.
    surrogate_jac[idx, np.arange(n, n+count)] = 1.
    upper_jac = -surrogate_jac[guard_idx]
    linear_jac = np.vstack([np.c_[-a, np.zeros((len(a), count))],
                           np.c_[a, np.zeros((len(a), count))],
                           np.c_[mg/.01, np.zeros((len(mg), count))], upper_jac])

    def surrogate(y):
        v = model['base'] + model['metric_jac']@(y[:n]/diagonal)
        v[idx] = y[n:]
        return v

    def fun(y):
        v, j = absolute_shape_response(surrogate(y), surrogate_jac,
                                      model['baseline']['records'], model['retained'])
        positive = np.maximum(v, 0.)
        az = a@y[:n]
        gradient = 2*j.T@positive/normalization
        gradient[:n] += .04*a.T@az/len(az)
        return float(positive@positive/normalization+.02*np.mean(az*az)), gradient

    def linear(y):
        return np.r_[scale-a@y[:n], scale+a@y[:n],
                     (mb+mg@y[:n]-np.minimum(mb, .02))/.01,
                     upper-surrogate(y)[guard_idx]]

    def exact_audit(z):
        v, _ = values(model, z)
        return min(float(scale-abs(anchor@z).max()),
                   float(((mb+model['margin_jac'][active]@z-np.minimum(mb, .02))/.01).min()),
                   float((upper-v[guard_idx]).min()))

    # The cuts carry time/space indices ONLY in this training response model.
    # The deployed actor uses continuous state features, without indices/tables.
    cuts = {}
    for path, offset in paths:
        m = model['models'][path]
        for channel in ('incoming', 'outgoing', 'turn'):
            signal = m['base'][channel] + m['jac'][channel]@start
            rank = np.argsort(-abs(signal) if channel == 'turn' else signal)
            cuts[path, channel] = set(map(int, rank[:8]))

    def make_cuts():
        rows, constants = [], []
        for path, offset in paths:
            m, target = model['models'][path], old[path]
            for channel, k, key, factor in [
                ('incoming', 1, 'in_edge_outward_mm', .002),
                ('outgoing', 2, 'out_edge_outward_mm', .005),
                ('turn', 5, 'full_spatial_peak_curvature_mm_inv', .02)]:
                for tick in sorted(cuts[path, channel]):
                    for sign in (-1., 1.) if channel == 'turn' else (-1.,):
                        row = np.zeros(n+count)
                        row[:n] = -sign*m['jac'][channel][tick]/diagonal/factor
                        row[aux[offset+k]] = 1.
                        rows.append(row)
                        target_scaled = target[key]*(.1 if channel == 'turn' else 1.)
                        constants.append((target_scaled-sign*m['base'][channel][tick])/factor)
                row = np.zeros(n+count)
                row[aux[offset+k]] = 1.
                rows.append(row)
                constants.append(target[key]/(.2 if channel == 'turn' else factor))
        return np.asarray(rows), np.asarray(constants)

    def nonlinear(y, return_jac=False):
        z = y[:n]/diagonal
        out, jac = [], []
        for path, offset in paths:
            m, target = model['models'][path], old[path]
            omega = m['base']['omega'] + m['jac']['omega']@z
            f = .001*180/np.pi/.5
            row = np.zeros(n+count)
            row[:n] = f*m['jac']['omega'][omega < 0.].sum(0)/diagonal
            row[aux[offset+3]] = 1.
            out.append(y[aux[offset+3]] + target['full_physical_reverse_deg']/.5
                       - f*np.maximum(-omega, 0.).sum())
            jac.append(row)
            turn = m['base']['turn'] + m['jac']['turn']@z
            f = 1.25/(.1*m['base']['angle']**2)/.02
            row = np.zeros(n+count)
            row[:n] = -2*f*turn@m['jac']['turn']/diagonal
            row[aux[offset+4]] = 1.
            out.append(y[aux[offset+4]] + target['full_normalized_energy']/.02 - f*(turn@turn))
            jac.append(row)
        return np.asarray(jac if return_jac else out)

    start_values = values(model, start)[0]
    y = np.r_[start*diagonal, start_values[idx]]
    best = None
    best_value = np.inf
    histories = []

    def remember(y):
        nonlocal best, best_value
        if time.perf_counter() > deadline:
            raise TimeoutError('Registered optimization wall budget')
        z = y[:n]/diagonal
        if exact_audit(z) >= -1e-7:
            value = objective(model, z)[0]
            if value < best_value:
                best, best_value = z.copy(), value

    remember(y)
    for sweep in range(4):
        cut_jac, cut_base = make_cuts()
        result = minimize(fun, y, jac=True, method='SLSQP',
            constraints=[dict(type='ineq', fun=linear, jac=lambda y: linear_jac),
                         dict(type='ineq', fun=lambda y: cut_base+cut_jac@y, jac=lambda y: cut_jac),
                         dict(type='ineq', fun=nonlinear, jac=lambda y: nonlinear(y, True))],
            callback=remember, options=dict(maxiter=160, ftol=1e-9))
        y = result.x
        remember(y)
        z = y[:n]/diagonal
        exact_values = values(model, z)[0]
        added = 0
        for path, offset in paths:
            m, target = model['models'][path], old[path]
            for channel, k, key, divisor in [('incoming', 1, 'in_edge_outward_mm', .002),
                ('outgoing', 2, 'out_edge_outward_mm', .005),
                ('turn', 5, 'full_spatial_peak_curvature_mm_inv', .2)]:
                signal = m['base'][channel] + m['jac'][channel]@z
                bound = y[aux[offset+k]]*divisor + target[key]
                violation = (abs(signal)/.1 if channel == 'turn' else -signal)-bound
                bad = np.flatnonzero(violation > 1e-9)
                for tick in bad[np.argsort(-violation[bad])[:16]]:
                    if int(tick) not in cuts[path, channel]:
                        cuts[path, channel].add(int(tick)); added += 1
        history = dict(sweep=sweep, converged=bool(result.success), message=result.message,
                       iterations=int(result.nit), cut_rows=len(cut_base), added_cuts=added,
                       full_grid_margin=exact_audit(z), predicted_objective=objective(model,z)[0],
                       max_epigraph_gap=float((exact_values[idx]-y[n:]).max()))
        histories.append(history)
        print('EPIGRAPH_SOLVE', scale, bool(np.any(buffer)), history, flush=True)
        if added == 0:
            break
    return best, dict(sweeps=histories, selected_predicted_objective=None if best is None else best_value,
                      selected_full_grid_margin=None if best is None else exact_audit(best),
                      convergence_claim=bool(result.success and added == 0 and best is not None
                                             and np.max(abs(best-z)) < 1e-9))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete-control-binding', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_dense_margin_suffix_learning_20260919'
    sampled = BASE/'canonical_gate_band_complete_current_suffix_credit_20260919'
    previous = BASE/'canonical_gate_band_per_corner_dense_credit_20260919'
    latest = BASE/'canonical_gate_band_dense_signed_trace_completion_20260919'
    plan = read(sampled/'query_plan.json')
    baseline = plan['baseline']
    parent = torch.load(source/'research_candidate.pt', map_location='cpu', weights_only=False)
    assert sha(source/'research_candidate.pt') == plan['parent_sha256']
    metadata = torch.load(previous/'per_corner_constrained_0.125.pt', map_location='cpu', weights_only=False)
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    stage = register('canonical_gate_band_error_buffered_trace_completion_20260919' if args.complete_control_binding
                     else 'canonical_gate_band_error_buffered_trace_credit_20260919',
        [__file__, BASE/'dense_trace_credit_model.py', BASE/'learn_trace_resolved_joint_credit.py',
         BASE/'local_suffix_credit_policy.py', BASE/'learn_stratified_suffix_joint_credit.py',
         source/'research_candidate.pt', sampled/'query_plan.json', sampled/'query_results.json',
         previous/'results.json', latest/'results.json', latest/'receipt.json'],
        dict(wall_s=480, reused_queries=638, new_queries=0, maximum_new_candidates=2,
             paths=20, trust_scale=.125, workers=2),
        'The measured signed-trace half-step preserves all local guards except peak '
        'curvature on paths8 and14. Separate nonlinear max aggregation with training-only '
        'epigraph variables and linear sample cuts, checking the entire saved grid before '
        'any physical evaluation. Compare unbuffered versus empirically buffered guards '
        'at the same parent, data,108-center three-action actor and0.125 trust. Calibrate '
        'a quadratic-in-trust positive prediction-error envelope from two already measured '
        'per-corner actors and the signed half-step. This empirical cushion is not a '
        'safety proof or a changed physical tolerance. Save solver status, including feasible '
        'nonconverged iterates without claiming optimality. Reject duplicate/unchanged '
        'actors. Retain86cd and original full64,shape,cruise,fairness/generalization gates. '
        'No PPO interaction training, action label fitting, trajectory/speed rule or sealed test.')
    started = time.perf_counter()
    outcomes, jobs, audits = [], [], []
    pool = None
    subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'), '--active', stage.name,
        '--kind', 'complete_closed_loop_parameter_optimization', '--next',
        '用逐采样点峰值约束和实测预测误差余量完成两个联合候选；按真实轨迹筛选。'], check=True, stdout=subprocess.DEVNULL)
    try:
        queries = sorted(read(sampled/'query_results.json')['queries'], key=lambda q:q['query'])
        model = build_model(baseline, metadata, source, sampled, queries, retained, np.array(plan['amplitude']))
        torch.save(model, stage/'response_model.pt')
        # Results also contain a reused control whose checkpoint belongs to a
        # different source. The registered error controls are these two actors.
        controls = [(previous, c) for c in read(previous/'results.json')['candidates']
                    if c['tag'] in ('per_corner_constrained_0.0625', 'per_corner_constrained_0.125')]
        assert len(controls) == 2
        controls += [(latest, c) for c in read(latest/'results.json')['candidates']]
        error_envelope = np.zeros_like(model['base'])
        observed = []
        known = []
        for folder, candidate in controls:
            md = torch.load(folder/(candidate['tag']+'.pt'), map_location='cpu', weights_only=False)
            z = coordinates(model, md)
            radius = float(abs(model['anchors']@z).max())
            actual = growth_deficits(candidate, retained)
            predicted = values(model,z)[0]
            residual = actual-predicted
            error_envelope = np.maximum(error_envelope, np.maximum(residual,0.)/radius**2)
            observed.append(dict(tag=candidate['tag'], radius=radius,
                residual=residual.tolist(), actual_guards=actual_guard_audit(model,candidate)))
            known.append(z)
        write_json(stage/'model_error_audit.json', dict(controls=observed,
            positive_residual_per_radius_squared=error_envelope.tolist()))
        # Save the reproducible response model once so later mechanism checks need
        # not repeatedly parse3828 raw traces; source hashes are in registration.
        half = coordinates(model, torch.load(latest/'signed_trace_0.0625.pt', map_location='cpu', weights_only=False))
        for tag, buffer in [('epigraph_unbuffered', np.zeros_like(error_envelope)),
                            ('epigraph_error_buffered', error_envelope*.125**2*1.25)]:
            z, audit = solve_epigraph(model, half, .125, buffer, started+385)
            audit.update(tag=tag, buffer=buffer.tolist())
            audits.append(audit)
            write_json(stage/'optimizer.json', dict(candidates=audits))
            if z is None or objective(model,z)[0] >= objective(model,np.zeros_like(z))[0]-1e-5:
                continue
            if any(np.max(abs(model['anchors']@(z-v))) < 1e-7 for v in known):
                audit['skipped_duplicate'] = True
                continue
            candidate = copy.deepcopy(metadata)
            candidate.update(coefficients=coefficients(model,z), step_scale=1.,
                learning_method='error_buffered_dense_trace_joint_parameter_optimization')
            with torch.no_grad(): model['policy'].coefficients.copy_(torch.tensor(candidate['coefficients']))
            prefix_error = max(float(abs(model['policy'].act(x,True)[0]-model['policy'].base.act(x,True)[0]).max())
                               for x in model['protected'])
            assert prefix_error <= 1e-12, prefix_error
            audit['maximum_protected_latent_difference'] = prefix_error
            torch.save(candidate, stage/(tag+'.pt'))
            jobs.append((tag,candidate)); known.append(z)
        write_json(stage/'optimizer.json', dict(candidates=audits))
        if jobs:
            pool = mp.get_context('spawn').Pool(2,suffix.initialize,
                (stage,parent,baseline,retained,source,started+475))
            for result in pool.imap_unordered(suffix.evaluate_network,jobs):
                result['local_guard_audit'] = actual_guard_audit(model,result) if result['summary']['successes']==20 else None
                outcomes.append(result)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('BUFFERED_TRACE_ACTUAL',result['tag'],result['summary'],result['local_guard_audit'],flush=True)
            pool.close();pool.join();pool=None
        good = [c for c in outcomes if c['primary_preserved'] and c['local_guard_audit']['guard_pass']
                and c['absolute_objective'] < absolute_actual(baseline,retained)]
        best = min(good,key=lambda c:c['absolute_objective']) if good else None
        result = dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag']
            torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_error_buffered_trace_joint_learning',new_candidates=len(outcomes),
            best=None if best is None else best['tag'],wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_error_buffered_trace_joint_learning',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '核对经验误差余量及逐点约束的真实形状效果，继续闭环联合学习；目标未完成。'],check=True,stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    mp.freeze_support()
    main()
