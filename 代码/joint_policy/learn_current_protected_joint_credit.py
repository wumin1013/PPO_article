"""Refresh complete suffix credit on current successful joint policies."""
import copy
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
import learn_stratified_suffix_joint_credit as suffix
from protected_suffix_optimizer import fit_actor
from learn_prefix_preserving_suffix_credit import prefix_indices
from learn_absolute_shape_joint_credit import absolute_actual
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


AMPLITUDE = np.array([.001, .015, .015])


def initialize(*args):
    suffix.AMPLITUDE = AMPLITUDE.copy()
    suffix.initialize(*args)


def query_plan(baseline, source, seed):
    rng = np.random.default_rng(seed); queries = []
    for r in baseline['records']:
        with np.load(source/r['trace']) as z: ref, rows, obs = [z[k].copy() for k in ('reference', 'rows', 'observations')]
        start = len(prefix_indices(ref, rows, obs))
        ordinary = np.flatnonzero(rows[np.minimum(np.arange(len(obs))*20, len(rows)-1), 12] == 0)
        ordinary = ordinary[ordinary >= start]
        for stratum, indices in enumerate(np.array_split(ordinary, 6)):
            if not len(indices): continue
            queries.append(dict(query=len(queries), path=r['path'], frame=int(rng.choice(indices))*20,
                stratum=stratum, stratum_size=len(indices), ordinary_decisions=len(ordinary), protected_prefix_decisions=start))
    return queries


def main():
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_numerical_prefix_suffix_credit_20260919'
    data = read(source/'results.json')
    baseline = next(c for c in data['candidates'] if c['tag'] == 'protected_0.125')
    parent_path = source/'protected_0.125.pt'
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    assert baseline['primary_preserved']
    stage = register('canonical_gate_band_current_protected_joint_credit_20260919',
        [__file__, BASE/'protected_suffix_optimizer.py', BASE/'local_suffix_credit_policy.py',
         BASE/'deep_joint_credit_policy.py', BASE/'learn_stratified_suffix_joint_credit.py',
         BASE/'learn_prefix_preserving_suffix_credit.py', parent_path, source/'results.json', source/'receipt.json'],
        dict(wall_s=900, workers=4, maximum_iterations=2, strata_per_unprotected_path=6,
             branches_per_query=6, amplitude=AMPLITUDE.tolist(), candidates_per_iteration=4,
             joint_scales=[.0625,.125,.25], feed_only_scale=.125, optimizer_steps=250),
        'Start from the specifically recorded protected0.125 research policy, not the '
        'scalar-best unprotected control or retained86cd. It completes20/20, preserves '
        'entry/first-turn, and reduces BOTH reversal and exit excursion while keeping '
        '3.677s curved99 cruise. Recollect action credit on its NEW actual closed-loop '
        'states with smaller steering/feed/preview perturbations[.001,.015,.015]. '
        'Sample six uniform temporal strata after the saved protected prefix and weight '
        'by stratum size. Excluded protected decisions have numerically zero proposed '
        'correction; this is not an unbiased-estimator claim. Each action intervention '
        'lasts20ms then follows the current actual policy to complete rest, checking '
        'exact nominal next-decision identity first. Optimize a smooth local joint actor '
        'in the numerical prefix nullspace, with diagonal curvature credit and original '
        'action trust limits. Matched feed-only uses the SAME queried states, consequences, '
        'optimizer and steps. Evaluate joint0.0625/0.125/0.25 and feed-only0.125 as full20 '
        'paths. Continue from a joint candidate only if actual primary constraints hold, '
        'absolute objective decreases and aggregate reverse AND exit do not worsen. '
        'Stop this method after no usable update or insufficient budget for another '
        'complete collection; no automatic unchanged training extension. All candidates '
        'remain research-only unless original per-corner/full64/fair/generalization '
        'gates pass. This is physical sampling and deterministic actor parameter '
        'optimization, not PPO, action imitation, a deployed template or speed rule.')
    started = time.perf_counter(); iterations=[]; pool=None; selected=None
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next','在当前成功策略状态重新采集完整后缀信用，比较联合与仅进给的真实完整收益。'],
        check=True,stdout=subprocess.DEVNULL)
    try:
        for iteration in range(2):
            if iteration and time.perf_counter()-started > 475: break
            folder=stage/f'iteration{iteration}';folder.mkdir()
            plan=query_plan(baseline,source,20260919+iteration)
            write_json(folder/'query_plan.json',dict(queries=plan,parent_sha256=sha(parent_path),
                baseline=baseline,source=str(source),amplitude=AMPLITUDE.tolist()))
            pool=mp.get_context('spawn').Pool(4,initialize,(folder,parent,baseline,retained,source,started+885))
            collected=[]
            for q in pool.imap_unordered(suffix.collect_query,plan):
                collected.append(q);write_json(folder/'query_results.json',dict(queries=collected))
                if len(collected)%12==0:
                    print('CURRENT_CREDIT',iteration,len(collected),'of',len(plan),'complete',
                        sum(r['success'] for q in collected for r in q['records']),flush=True)
            collected.sort(key=lambda q:q['query'])
            jobs=[];training={}
            for arm in ('joint','feed_only'):
                md,audit=fit_actor(parent,baseline,source,collected,AMPLITUDE,arm)
                training[arm]=audit
                for scale in ((.0625,.125,.25) if arm=='joint' else (.125,)):
                    candidate=copy.deepcopy(md);candidate['step_scale']=scale;tag=f'{arm}_{scale}'
                    torch.save(candidate,folder/(tag+'.pt'));jobs.append((tag,candidate))
            write_json(folder/'actor_optimization.json',training)
            outcomes=[]
            for result in pool.imap_unordered(suffix.evaluate_network,jobs):
                outcomes.append(result);write_json(folder/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('CURRENT_JOINT_ACTUAL',iteration,result['tag'],result['absolute_objective'],result['summary'],result['promotion'],flush=True)
            pool.close();pool.join();pool=None
            old_cost=absolute_actual(baseline,retained)
            good=[c for c in outcomes if c['tag'].startswith('joint') and c['primary_preserved']
                and c['absolute_objective']<old_cost-1e-5
                and c['summary']['reverse_sum_deg']<=baseline['summary']['reverse_sum_deg']+1e-6
                and c['summary']['exit_sum_mm']<=baseline['summary']['exit_sum_mm']+1e-7]
            best=min(good,key=lambda c:c['absolute_objective']) if good else None
            iterations.append(dict(iteration=iteration,queries=len(collected),candidates=len(outcomes),
                previous_objective=old_cost,selected=None if best is None else best['tag'],training=training))
            write_json(stage/'iterations.json',dict(iterations=iterations,adopted=False,goal_complete=False))
            if best is None: break
            selected=copy.deepcopy(best)
            for r in selected['records']:r['trace']=f'iteration{iteration}/'+r['trace']
            selected['tag']=f'iteration{iteration}/'+best['tag']
            chosen=next(md for tag,md in jobs if tag==best['tag'])
            torch.save(chosen,stage/'research_candidate.pt')
            write_json(stage/'results.json',dict(candidates=[selected],best=selected['tag'],iterations=iterations,
                adopted=False,goal_complete=False))
            baseline=best;source=folder;parent=chosen;parent_path=folder/(best['tag']+'.pt')
        receipt(stage,dict(status='completed_current_protected_joint_credit',iterations=iterations,
            wall_s=time.perf_counter()-started,best=None if selected is None else selected['tag'],
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_current_protected_joint_credit',error=str(exc),iterations=iterations,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next','读取当前策略信用重采样与联合/仅进给完整对照，继续修正原目标差距。'],
            check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
