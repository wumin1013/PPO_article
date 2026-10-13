"""Check action-difference scale, then learn head amplitudes from full feedback."""
import copy
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
import learn_stratified_suffix_joint_credit as suffix
from learn_absolute_shape_joint_credit import absolute_actual, absolute_shape_response
from learn_retained_joint_growth import growth_deficits
from analyze_nonlinear_joint_response import branch
from probe_coordinated_suffix import reconstruct
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def small_action_query(job):
    query, amplitude = job
    old = next(r for r in suffix.WORK['baseline']['records'] if r['path'] == query['path'])
    with np.load(suffix.WORK['source']/old['trace']) as z:
        ref, rows, actions = [z[k].copy() for k in ('reference', 'rows', 'actions')]
    costs = dict(actor_calls=0, physics_ticks=0, saved_prefix_geometry_ticks=0)
    env = reconstruct(ref, rows, actions, query['frame'], costs, env_factory=suffix.WORK['factory'])
    state = env.state()
    latent = suffix.WORK['policy'].act(env.observation(), True)[0]
    _, _, _, info = env.step(latent)
    assert np.array_equal(info['rows'], rows[query['frame']:query['frame']+len(info['rows'])])
    records = []
    for sign in (-1, 1):
        offset = np.array([sign*amplitude, 0., 0.])
        r, full, held, obs = branch(suffix.WORK['policy'], suffix.WORK['factory'], ref,
            rows, actions, state, offset, costs, suffix.WORK['deadline'])
        r.update(path=query['path'], sign=sign)
        r['absolute_loss'] = suffix.with_branch_loss(r, full)
        r['trace'] = f"small_q{query['query']}_{amplitude}_{sign:+d}.npz"
        np.savez_compressed(suffix.WORK['stage']/r['trace'], reference=ref, rows=full,
                            actions=held, suffix_observations=obs, offset=offset, frame=query['frame'])
        records.append(r)
    base = query['baseline_absolute_loss']
    values = [r['absolute_loss'] for r in records]
    complete = all(v is not None for v in values)
    return dict(query=query['query'], path=query['path'], frame=query['frame'], amplitude=amplitude,
        records=records, costs=costs, exact_next_decision_identity=True,
        derivative=(values[1]-values[0])/(2*amplitude) if complete else None,
        second_difference=(values[1]+values[0]-2*base) if complete else None)


def vector(candidate, retained):
    value = growth_deficits(candidate, retained)
    return absolute_shape_response(value, np.zeros((len(value), 1)), candidate['records'], retained)[0]


def constraint_vector(candidate):
    # Full command-segment bound, not just distances of selected points.
    values = [r['corridor']['margin_lower_mm'] for r in candidate['records']]
    values += [1e-6-r['shape']['in_edge_outward_mm'] for r in candidate['records'] if 'shape' in r]
    return np.asarray(values)


def proposal(current, paired, retained):
    origin = vector(current, retained)
    constraint = constraint_vector(current)
    g, h, cg, ch, bounds, modes = [], [], [], [], [], []
    for lo, hi in paired:
        lo_ok, hi_ok = [all(r['success'] for r in c['records']) for c in (lo, hi)]
        lv, hv = vector(lo, retained) if lo_ok else None, vector(hi, retained) if hi_ok else None
        if lo_ok and hi_ok:
            g.append((hv-lv)/2); h.append((hv+lv)/2-origin); bounds.append((-1., 1.)); modes.append('central_complete')
        elif lo_ok:
            g.append(origin-lv); h.append(np.zeros_like(origin)); bounds.append((-1., 0.)); modes.append('negative_complete')
        elif hi_ok:
            g.append(hv-origin); h.append(np.zeros_like(origin)); bounds.append((0., 1.)); modes.append('positive_complete')
        else:
            g.append(np.zeros_like(origin)); h.append(np.zeros_like(origin)); bounds.append((0., 0.)); modes.append('both_failed_fixed')
        low, high = constraint_vector(lo), constraint_vector(hi)
        cg.append((high-low)/2); ch.append((high+low)/2-constraint)
    g, h, cg, ch = [np.column_stack(a) for a in (g, h, cg, ch)]
    def objective(u):
        value = np.maximum(origin+g@u+h@(u*u), 0.)
        return float(value@value+.01*(u@u)), 2*(g+2*h*u).T@value+.02*u
    fun = lambda u: (constraint+cg@u+ch@(u*u))/.01
    jac = lambda u: (cg+2*ch*u)/.01
    starts = [np.zeros(3)]
    for axis, (lo, hi) in enumerate(paired):
        for sign, c in zip((-1, 1), (lo, hi)):
            if c['primary_preserved']: starts.append(sign*np.eye(3)[axis])
    solutions = [minimize(objective, start, jac=True, method='SLSQP', bounds=bounds,
        constraints=[dict(type='ineq', fun=fun, jac=jac)], options=dict(maxiter=250, ftol=1e-9)) for start in starts]
    good = [s for s in solutions if s.success and fun(s.x).min() >= -1e-6]
    if not good: return None, dict(modes=modes, messages=[s.message for s in solutions])
    best = min(good, key=lambda s: objective(s.x)[0])
    return best.x, dict(modes=modes, normalized_coefficients=best.x.tolist(),
        predicted_objective=objective(best.x)[0], predicted_minimum_constraint_mm=float(fun(best.x).min()*.01),
        objective_origin=absolute_actual(current, retained),
        scope='Diagonal complete-policy metric/constraint model; actual full rollouts remain decisive.')


def main():
    os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    learned = BASE/'canonical_gate_band_stratified_suffix_joint_credit_20260919'
    source = BASE/'canonical_gate_band_fast_absolute_shape_joint_credit_20260919'
    data = read(source/'results.json')
    baseline = copy.deepcopy(next(r for r in data['candidates'] if r['tag'] == data['best']))
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    parent = torch.load(source/'research_candidate.pt', map_location='cpu', weights_only=False)
    network = torch.load(learned/'joint_1.0.pt', map_location='cpu', weights_only=False)
    queries = read(learned/'query_results.json')['queries']
    picked = [next(q for q in queries if q['query'] == i) for i in (47, 83, 85, 94)]
    stage = register('canonical_gate_band_shared_network_credit_calibration_20260919',
        [__file__, BASE/'learn_stratified_suffix_joint_credit.py', BASE/'deep_joint_credit_policy.py',
         learned/'joint_1.0.pt', learned/'results.json', learned/'query_results.json', learned/'receipt.json',
         source/'research_candidate.pt', source/'results.json', source/'receipt.json'],
        dict(wall_s=360, workers=4, small_action_queries=8, small_action_branches=16,
             complete_head_axis_candidates=6, complete_proposal_candidates=3,
             paths=20, head_amplitude_step=.125, steering_amplitudes=[.001, .0002]),
        'The suffix-trained shared network predicted a cost decrease larger than the entire '
        'nonnegative current cost, dominated by isolated finite-difference queries, and every '
        'full network update failed. Check four influential states at two smaller steering '
        'amplitudes, preserving exact reconstruction and full policy feedback to rest. Reuse '
        'the original +/-0.005 evidence. Separately keep the learned state functions fixed '
        'and learn their three output amplitudes from COMPLETE policy perturbations, +/-0.125 '
        'per head. Fit a three-dimensional diagonal quadratic model of signed shape/time/'
        'cruise metrics and full segment corridor/entry margins. Failed prefixes supply only '
        'constraint evidence, never time/cruise. Test optimized full/half amplitudes plus a '
        'small equal-amplitude control. This is closed-loop parameter optimization, not PPO '
        'or runtime switching/action interpolation. Save only actual complete primary-preserving '
        'improvement as research evidence, never adopt it before all original gates. If full '
        'policy response contradicts local credit, do not continue the same surrogate training.')
    started = time.perf_counter()
    subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'), '--active', stage.name,
        '--kind', 'complete_closed_loop_parameter_optimization', '--next', '核实差分尺度与完整策略三动作响应，实测决定下一次学习。'],
        check=True, stdout=subprocess.DEVNULL)
    pool = mp.get_context('spawn').Pool(4, suffix.initialize, (stage, parent, baseline, retained, source, started+350))
    outcomes, small, cache = [], [], {}
    def metadata(coeff):
        md = copy.deepcopy(network); md['output_mask'] = np.asarray(coeff, np.float32); md['step_scale'] = 1.
        md['learning_method'] = 'complete_policy_calibrated_shared_head_amplitudes'
        return md
    def evaluate(jobs):
        submitted = []
        for tag, coeff in jobs:
            key = np.asarray(coeff, np.float32).tobytes()
            if key not in cache:
                md = metadata(coeff); torch.save(md, stage/(tag+'.pt'))
                submitted.append((tag, md)); cache[key] = tag
        for result in pool.imap_unordered(suffix.evaluate_network, submitted):
            md = next(m for t, m in submitted if t == result['tag'])
            result['head_amplitudes'] = md['output_mask'].tolist()
            outcomes.append(result)
            write_json(stage/'results.json', dict(candidates=outcomes, adopted=False, goal_complete=False))
            print('FULL_POLICY_CALIBRATION', result['tag'], result['absolute_objective'], result['summary'], result['promotion'], flush=True)
    try:
        for result in pool.imap_unordered(small_action_query, [(q, a) for q in picked for a in (.001, .0002)]):
            small.append(result); write_json(stage/'small_action_results.json', dict(queries=small))
            print('SMALL_ACTION_CREDIT', result['query'], result['amplitude'], result['derivative'], result['second_difference'], flush=True)
        evaluate([(f'head{h}_{s:+d}', .125*s*np.eye(3)[h]) for h in range(3) for s in (-1, 1)])
        paired = [(next(c for c in outcomes if c['tag'] == f'head{h}_-1'),
                   next(c for c in outcomes if c['tag'] == f'head{h}_+1')) for h in range(3)]
        u, audit = proposal(baseline, paired, retained)
        write_json(stage/'proposal.json', audit)
        jobs = [('equal_small', [.0625]*3)]
        if u is not None:
            jobs += [(f'calibrated_{scale}', .125*scale*u) for scale in (.5, 1.)]
        evaluate(jobs)
        pool.close(); pool.join(); pool = None
        usable = [c for c in outcomes if c['primary_preserved'] and c['absolute_objective'] < absolute_actual(baseline, retained)-1e-5]
        best = min(usable, key=lambda c: c['absolute_objective']) if usable else None
        result = dict(candidates=outcomes, baseline_objective=absolute_actual(baseline, retained), adopted=False, goal_complete=False)
        if best:
            result['best'] = best['tag']; torch.save(metadata(best['head_amplitudes']), stage/'research_candidate.pt')
        write_json(stage/'results.json', result)
        receipt(stage, dict(status='completed_shared_network_full_policy_credit_calibration', wall_s=time.perf_counter()-started,
            small_action_queries=len(small), new_candidates=len(outcomes), best=best['tag'] if best else None, adopted=False, goal_complete=False))
    except Exception as exc:
        if pool is not None: pool.terminate(); pool.join()
        receipt(stage, dict(status='failed_shared_network_full_policy_credit_calibration', error=str(exc),
            wall_s=time.perf_counter()-started, small_action_queries=len(small), new_candidates=len(outcomes), adopted=False, goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
            '--kind', 'complete_closed_loop_parameter_optimization_finished', '--next',
            '根据差分尺度与完整策略响应校准结果继续联合学习；所有研究候选尚未采用。'], check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    mp.freeze_support(); main()
