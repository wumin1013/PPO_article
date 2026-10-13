"""Joint parameter-distribution learning from actual complete-policy outcomes."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import time
import numpy as np
import torch
from learn_full_pool_joint_credit import initialize,evaluate,candidate_metadata
from learn_full_pool_compensation import progress
from learn_pathwise_full_credit import actual_selection
from learn_retained_joint_growth import growth_deficits
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def measured_score(candidate,baseline,retained):
    """Actual returns only. Incomplete prefixes have no efficiency fitness."""
    if candidate['summary']['successes']!=64:
        return 1e6+1e4*(64-candidate['summary']['successes'])
    if not candidate['primary_preserved']:
        return 1e5+candidate['absolute_objective']
    v=growth_deficits(candidate,retained);base=growth_deficits(baseline,retained)
    guards=np.array([9*p+k for p in range(16) for k in range(1,8)])
    regret=np.maximum(v[guards]-np.maximum(base[guards],0.),0.)
    a,b=candidate['summary'],baseline['summary']
    aggregate=np.maximum(np.array([
        (a['reverse_sum_deg']-b['reverse_sum_deg'])/.5,
        (a['exit_sum_mm']-b['exit_sum_mm'])/.005,
        (a['energy_sum']-b['energy_sum'])/.02,
        (a['complete_time_s']-b['complete_time_s']-.02)/.02,
        (b['curved_cruise99_s']-.05-a['curved_cruise99_s'])/.05]),0.)
    return float(candidate['absolute_objective']+20*(regret@regret+aggregate@aggregate))


def bounded_sample(x,trust):
    # Offline parameter trust, not runtime action clipping.
    x=np.clip(x,-2.,2.)
    size=np.max(abs(trust@x))
    return x*min(1.,1./max(size,1e-12))


def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(202609193)
    source=BASE/'canonical_gate_band_current_bend_capacity_learning_20260919'
    reused_stages=[source,
        BASE/'canonical_gate_band_fine_full_policy_credit_learning_20260919',
        BASE/'canonical_gate_band_joint_interaction_subspace_learning_20260919']
    data=read(source/'baseline.json');baseline,retained=data['baseline'],data['retained']
    metadata=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:directions,trust=z['directions'].copy(),z['trust'].copy()
    n=len(directions)
    assert n==18 and len(metadata['centers'])==120
    cached=[]
    for folder in reused_stages:
        for c in read(folder/'results.json')['candidates']:
            c=copy.deepcopy(c)
            if len(c['parameters'])!=n:continue
            for r in c['records']:r['trace']=str(folder/r['trace'])
            c['source_stage']=folder.name
            c['measured_score']=measured_score(c,baseline,retained)
            cached.append(c)
    base=copy.deepcopy(baseline)
    base.update(tag='unchanged_parent',parameters=[0.]*n,primary_preserved=True,source_stage=source.name)
    base['measured_score']=measured_score(base,baseline,retained);cached.append(base)
    stage=register('canonical_gate_band_measured_joint_distribution_learning_20260919',
        [__file__,BASE/'learn_full_pool_joint_credit.py',BASE/'learn_pathwise_full_credit.py',
         BASE/'learn_retained_joint_growth.py',source/'zero_extension.pt',source/'parameter_space.npz',
         source/'baseline.json']+[f/name for f in reused_stages for name in ('results.json','receipt.json')],
        dict(wall_s=1500,workers=4,maximum_generations=6,population=12,elites=4,
             maximum_new_full64_candidates=72,seed=202609193,initial_parameter_sd=.10,
             maximum_normalized_function_change=1.,stop_after_no_score_gain_generations=3,
             training_shape_penalty_weight=20.,physical_ticks_upper=12000000),
        'Current deterministic response models repeatedly miss small actual per-corner effects '
        'even after validation and smaller amplitude; no model has been adopted. Compare a '
        'model-free COMPLETE-POLICY parameter-learning mechanism on the SAME120-center actor '
        'and18 joint coefficients. Reuse all cached actual outcomes to initialize the search '
        'distribution. Each generation measures12 antithetic Gaussian coefficient candidates '
        'on ALL64 training routes. Rank only actual complete trajectory objectives and shape '
        'regret; failures receive fixed completion penalties, never short-prefix time benefit. '
        'Update mean/covariance from four measured elites, including incumbent actual candidates. '
        'The search mean is exploratory and may trade shape metrics; it is NOT an accepted model. '
        'A research checkpoint can be selected only by the unchanged exact primary,per-corner, '
        'aggregate,time and cruise checks used previously. Formal86cd and best5ab checkpoints '
        'remain immutable. All parameter trust bounds are offline; deployment evaluates learned '
        'continuous state functions with the existing guide,projection and braking. No new '
        'response estimator, action labels, temporal/angle rules, PPO claim, sealed tests or '
        'fair/generalization claim. Save raw routes, distribution state and candidate hashes. '
        'Stop this method after6 generations,25min,or3 generations without measured score gain; '
        'these are experiment limits, not completion of the original research objective.')
    started=time.perf_counter();pool=None;outcomes=[];generations=[];selected=None
    progress(stage,True,'直接依据完整64路径真实回报学习联合参数分布，最多六批，原研究目标与正式保留模型不变。')
    try:
        write_json(stage/'baseline.json',data)
        ordered=sorted([c for c in cached if c['primary_preserved']],key=lambda c:c['measured_score'])
        elite=np.array([c['parameters'] for c in ordered[:4]])
        mean=elite.mean(0)
        centered=elite-mean
        covariance=.04*(centered.T@centered)/len(elite)+.10**2*np.eye(n)
        incumbent=ordered[0];old_score=incumbent['measured_score'];no_gain=0
        refs,curved={},{}
        for r in baseline['records']:
            with np.load(r['trace']) as z:refs[r['path']]=z['reference'].copy()
            curved[r['path']]=r['curved_path']
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,metadata,directions,refs,curved,retained,started+1490))
        write_json(stage/'initial_distribution.json',dict(mean=mean.tolist(),covariance=covariance.tolist(),
            cached_candidates=len(cached),cached_best=incumbent['tag'],cached_score=old_score,
            original_score=base['measured_score'],exploratory_mean_is_not_accepted=True))
        print('MEASURED_DISTRIBUTION_START',len(cached),'cached',incumbent['tag'],old_score,flush=True)
        reason='maximum_generations'
        for generation in range(6):
            if time.perf_counter()>started+1250:
                reason='wall_budget_reserve';break
            eigen,vectors=np.linalg.eigh(covariance)
            factor=vectors@np.diag(np.sqrt(np.maximum(eigen,1e-6)))
            noise=rng.normal(size=(6,n))
            proposals=[bounded_sample(mean+factor@z,trust) for z in np.vstack([noise,-noise])]
            jobs=[(f'g{generation}_candidate{i}',x) for i,x in enumerate(proposals)]
            batch=[]
            for c in pool.imap_unordered(evaluate,jobs):
                c['measured_score']=measured_score(c,baseline,retained)
                c['source_stage']=stage.name;batch.append(c);outcomes.append(c)
                write_json(stage/'results.json',dict(candidates=outcomes,generations=generations,adopted=False,goal_complete=False))
                print('MEASURED_CANDIDATE',generation,c['tag'],c['summary']['successes'],c['measured_score'],flush=True)
            feasible=[c for c in batch if c['primary_preserved']]
            leaders=sorted(ordered[:4]+feasible,key=lambda c:c['measured_score'])[:4]
            ordered=leaders
            incumbent=leaders[0]
            improved=incumbent['measured_score']<old_score-.001
            no_gain=0 if improved else no_gain+1
            elite=np.array([c['parameters'] for c in leaders]);target=elite.mean(0)
            shift=target-mean;centered=elite-target
            covariance=.70*covariance+.20*(centered.T@centered)/len(elite)+.10*np.outer(shift,shift)
            covariance*=1.15 if improved else .75
            mean=.35*mean+.65*target
            old_score=min(old_score,incumbent['measured_score'])
            best=actual_selection(batch,baseline,retained)
            if best is not None and (selected is None or best['absolute_objective']<selected['absolute_objective']):
                selected=best
                md=candidate_metadata(metadata,directions,best['parameters'])
                torch.save(md,stage/'research_candidate.pt')
                torch.save(md,stage/(best['tag']+'.pt'))
            # Keep the exploratory leader for reproducibility, explicitly unadopted.
            torch.save(candidate_metadata(metadata,directions,incumbent['parameters']),stage/f'exploratory_leader_g{generation}.pt')
            row=dict(generation=generation,full64_candidates=len(batch),primary_preserved=len(feasible),
                leader=incumbent['tag'],leader_source=incumbent['source_stage'],measured_score=incumbent['measured_score'],
                mean=mean.tolist(),covariance=covariance.tolist(),no_gain=no_gain,
                accepted_research_candidate=selected['tag'] if selected else None,
                wall_s=time.perf_counter()-started)
            generations.append(row)
            value=dict(candidates=outcomes,generations=generations,adopted=False,goal_complete=False)
            if selected:value['best']=selected['tag']
            write_json(stage/'results.json',value)
            np.savez_compressed(stage/'distribution_state.npz',mean=mean,covariance=covariance)
            print('MEASURED_GENERATION',generation,incumbent['tag'],incumbent['measured_score'],
                  'primary',len(feasible),'research_best',selected['tag'] if selected else None,flush=True)
            if no_gain>=3:
                reason='three_generations_without_measured_score_gain';break
        pool.close();pool.join();pool=None
        result=dict(candidates=outcomes,generations=generations,adopted=False,goal_complete=False)
        if selected:result['best']=selected['tag']
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_measured_joint_distribution_learning',full64_candidates=len(outcomes),
            best=selected['tag'] if selected else None,stop_reason=reason,wall_s=time.perf_counter()-started,
            research_model_sha256=sha(stage/'research_candidate.pt') if selected else None,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_measured_joint_distribution_learning',error=str(exc),full64_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据直接完整回报学习对照继续推进形状与巡航目标；未冻结公平比较和泛化方案。')


if __name__=='__main__':mp.freeze_support();main()
