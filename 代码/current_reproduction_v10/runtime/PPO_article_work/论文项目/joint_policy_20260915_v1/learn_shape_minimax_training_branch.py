"""Worst-corner joint training objective; final adoption guards remain separate."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from head_counterfactual_rollouts import cost
from duration_joint_metrics import guards
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from audit_early_geometry_outcomes import constraint_names
from learn_full_pool_joint_credit import aggregate
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


COMPONENTS=(('out_edge_outward_mm','exit',.005),
    ('full_physical_reverse_deg','reverse',.5),('reverse_lateral_total_mm','lateral',.002))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smooth-minimax',action='store_true')
    args=parser.parse_args()
    components=COMPONENTS+((('full_normalized_energy','energy',.02),
        ('full_spatial_peak_curvature_mm_inv','curve',.2)) if args.smooth_minimax else ())
    torch.set_num_threads(1)
    started=time.perf_counter();deadline=started+160.
    source=BASE/'canonical_gate_band_deterministic_local_credit_20260920'
    prior=BASE/('canonical_gate_band_shape_minimax_training_branch_20260920' if args.smooth_minimax
                else 'canonical_gate_band_joint_secant_credit_20260920')
    parent_file=BASE/'canonical_gate_band_reference_supported_joint_learning_20260920/research_candidate.pt'
    assert sha(parent_file)=='ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229'
    assert not read(prior/'receipt.json')['research_acceptable']
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    names=constraint_names(paths)
    retained={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    scales={key:max(r['shape'][key] for r in (selected if key in ('full_normalized_energy',
        'full_spatial_peak_curvature_mm_inv') else baseline) if 'shape' in r) for key,_,_ in components}
    def severity(records):
        return {key:max(r['shape'][key] for r in records)/scales[key] for key,_,_ in components}
    def total_shape(records):
        return {key:sum(r['shape'][key] for r in records) for key,_,_ in components}
    replay_file=BASE/'canonical_gate_band_shared_reference_transfer_joint_learning_20260920/affected_replay.npz'
    with np.load(replay_file) as z:replay={k:z[k].copy() for k in z.files}
    stages=[BASE/n for n in ('canonical_gate_band_deterministic_complete_credit_20260920',
        'canonical_gate_band_deterministic_local_credit_20260920','canonical_gate_band_minimum_norm_complete_credit_20260920',
        'canonical_gate_band_joint_secant_credit_20260920')]
    if args.smooth_minimax:stages.append(prior)
    known=[c for s in stages for c in read(s/'state_joint/results.json')['candidates'] if c['tag']!='reused_parent']
    stage=register('canonical_gate_band_smooth_minimax_training_branch_20260920' if args.smooth_minimax
                   else 'canonical_gate_band_shape_minimax_training_branch_20260920',
        [__file__,parent_file,source/'baseline.json',source/'receipt.json',prior/'receipt.json',replay_file]+[
         s/'state_joint/results.json' for s in stages]+[source/(m+s) for m in ('geometry','dynamic')
         for s in ('_initial_policy.pt','_complete_gradient.npz')]+[BASE/n for n in (
         'observable_memory_joint_policy.py','head_counterfactual_rollouts.py','saved_memory_action_identity.py',
         'learn_early_memory_joint_policy.py','shape_event_credit.py','duration_joint_metrics.py','learn_intrinsic_joint_recovery.py')],
        dict(wall_s=180,workers=4,new_suffixes=0,new_stochastic_episodes=0,ppo_updates=0,
             maximum_new_policies=4,trust_bounds=[.15,.5],coefficient_bound=.01),
        ('Add peak curvature and normalized curve energy to the worst-defect training '
         'objective, scaled by the affected-route baseline maxima. The three-defect '
         'objective reduced its maxima but worsened smoothness and lost3-4ms cruise. '
         'Also preserve EACH route cruise-event margin separately: summing signed '
         'order-statistic event margins does not protect the aggregate event duration. '
         'Reuse the four new joint responses for model calibration. Intermediate '
         'shadow selection now requires all FIVE category maxima and sums nonincreasing '
         'along with physical/time/cruise checks. Final all203-guard adoption is unchanged. '
         'This changes objective plus event-constraint handling; do not attribute it '
         'to a single factor. ' if args.smooth_minimax else '')+
        'Revise the TRAINING objective after calibrated and regularized local updates '
        'improved aggregate shape but failed per-route nonregression. Jointly minimize '
        'the worst normalized exit excursion, reversal and negative lateral distance '
        'over the affected routes, with category scales fixed from the current full88 '
        'training baseline. Keep predicted aggregate improvements in all three shape '
        'categories, energy/curvature, total time and cruise; keep entry,corridor and '
        'first-turn protections. Individual auxiliary shape nonregression is NOT imposed '
        'on intermediate TRAINING proposals, but every original guard is still measured '
        'and REQUIRED for final candidate adoption. Only complete physical-safe policies '
        'with all three worst defects nonincreasing, aggregate defects/energy/curvature '
        'nonincreasing, no total time/cruise loss, and a strict worst-defect gain may '
        'be saved as a SHADOW TRAINING state. Formal86cd and protected ded2a93f remain '
        'unchanged. This is a combined objective/optimization change, not a one-factor '
        'PPO superiority claim. Reuse all measured complete joint responses, same '
        'two162-parameter actors; no new collection, angle/time schedules or weakened '
        'V/A/J/corridor/completion/rest rules.180s then decide from actual trajectories.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'selection_hold.json',dict(reason='Intermediate training branch; no research promotion or formal adoption.',
        final_original_guards_unchanged=True,category_scales=scales))
    history,blocks,models=[],[],{}
    pool=None
    try:
        for mode in ('geometry','dynamic'):
            md=torch.load(source/(mode+'_initial_policy.pt'),map_location='cpu',weights_only=False)
            models[mode]=md;torch.save(md,stage/(mode+'_initial_policy.pt'))
            _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
            def entry(records,u,tag):
                first=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
                extra=intrinsic_guards(records,selected)
                return dict(tag=tag,context_arm=mode,coefficients=np.asarray(u).tolist(),records=records,
                    score=sum(cost(r,b,retained[r['path']]) for r,b in zip(records,selected)),
                    measured_inequalities=first.tolist(),augmented_measured_inequalities=np.r_[first,extra].tolist(),
                    augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,first),extra].tolist(),
                    severity=severity(records) if all(r['success'] for r in records) else {},
                    actual_rerun_paths=[],exact_zero_support_paths=paths)
            if not history:
                zero=entry(copy.deepcopy(selected),np.zeros(md['coefficients'].size),'reused_parent');history.append(zero)
                torch.save(md,stage/'initial_policy.pt')
            with np.load(source/(mode+'_complete_gradient.npz')) as z:
                original=np.vstack([z['loss_gradient'],z['constraint_jacobian']])
                margins=z['baseline_margins'].copy();parent=z['parent_matrix'].copy()
            identity=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
            pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
            for trust in (.15,.5):
                if time.perf_counter()>deadline:raise TimeoutError('Shape minimax training budget')
                records=[c for c in known+history[1:] if c['context_arm']==mode]
                inputs=np.array([c['coefficients'] for c in records])
                outputs=np.array([[c['score'],*c['augmented_learning_inequalities']] for c in records])
                base=np.r_[zero['score'],margins]
                correction,_,rank,_=np.linalg.lstsq(inputs,outputs-base-inputs@original.T,rcond=1e-6)
                model=original+correction.T;jac=model[1:]
                levels,derivatives=[],[]
                for key,name,unit in components:
                    for r in selected:
                        levels.append(r['shape'][key]/scales[key])
                        derivatives.append(-jac[names.index(f"{r['path']}.{name}")]*unit/scales[key])
                levels=np.array(levels);derivatives=np.array(derivatives);n=md['coefficients'].size
                matrices=[np.c_[derivatives,-np.ones(len(levels))],np.c_[parent,np.zeros(len(parent))],
                    np.c_[-parent,np.zeros(len(parent))]]
                bounds=[-levels,np.full(len(parent),trust),np.full(len(parent),trust)]
                per_route=('entry','first','corridor')+(('motion.cruise_ge99_s','motion.longest_cruise_ge99_s') if args.smooth_minimax else ())
                for name in per_route:
                    ids=[names.index(f'{p}.{name}') for p in paths]
                    matrices.append(np.c_[-jac[ids],np.zeros(len(ids))]);bounds.append(margins[ids])
                for name in ('exit','reverse','lateral','energy','curve','complete_time_s',
                             'motion.cruise_ge99_s','motion.longest_cruise_ge99_s'):
                    ids=[names.index(f'{p}.{name}') for p in paths]
                    matrices.append(np.r_[-jac[ids].sum(0),0.][None,:]);bounds.append([margins[ids].sum()])
                objective=np.r_[.02*derivatives.mean(0),1.]
                solved=linprog(objective,A_ub=np.vstack(matrices),b_ub=np.concatenate(bounds),
                    bounds=[(-.01,.01)]*n+[(0.,None)],method='highs',
                    options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
                block=dict(context_arm=mode,trust=trust,secant_rank=int(rank),solver_success=bool(solved.success))
                if solved.success:
                    u=solved.x[:-1];block.update(predicted_worst_defect=float(solved.x[-1]),coefficient_norm=float(np.linalg.norm(u)))
                    if np.linalg.norm(u)>1e-10:
                        tag=f'{mode}_trust{trust}';actor=copy.deepcopy(md);actor['coefficients']=u.reshape(-1,3)
                        torch.save(actor,stage/(tag+'.pt'))
                        unchanged=[p for p in paths if identity.unchanged(p,u)];actual=[p for p in paths if p not in unchanged]
                        measured=pool.map(worker_evaluate,[(p,u,tag) for p in actual]);by={r['path']:r for r in measured}
                        c=entry([by.get(r['path'],copy.deepcopy(r)) for r in selected],u,tag)
                        c.update(actual_rerun_paths=actual,exact_zero_support_paths=unchanged)
                        physical=all(r['success'] and r['audit']['kinematic_pass'] and r['audit']['rest_pass']
                            and r['corridor']['margin_lower_mm']>=0. and r['target_motion']['first_turn_inside'] for r in c['records'])
                        progress=False
                        if physical:
                            before,after=aggregate(selected),aggregate(c['records'])
                            bt,at=total_shape(selected),total_shape(c['records'])
                            progress=(max(c['severity'].values())<=1.+1e-10 and sum(c['severity'].values())<len(components)-1e-8
                                and all(at[k]<=bt[k]+1e-10 for k in bt)
                                and after['energy_sum']<=before['energy_sum']+1e-10
                                and sum(r['shape']['full_spatial_peak_curvature_mm_inv'] for r in c['records'])<=sum(r['shape']['full_spatial_peak_curvature_mm_inv'] for r in selected)+1e-10
                                and after['complete_time_s']<=before['complete_time_s']+1e-10
                                and after['curved_cruise99_s']>=before['curved_cruise99_s']-1e-10)
                        c.update(physical_pass=physical,shadow_training_progress=bool(progress),
                            original_all_guard_pass=min(c['augmented_measured_inequalities'])>=-1e-8)
                        history.append(c)
                        block.update(actual_severity=c['severity'],physical_pass=physical,shadow_training_progress=bool(progress),
                            original_guard_min=min(c['augmented_measured_inequalities']))
                        # The original final-adoption guard is NEVER replaced by this training objective.
                        best=min((r for r in history if min(r['augmented_measured_inequalities'])>=-1e-8),key=lambda r:r['score'])
                        write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
                blocks.append(block);write_json(stage/'minimax_updates.json',blocks)
                print('SHAPE_MINIMAX_TRAINING',block,flush=True)
            pool.close();pool.join();pool=None
        shadows=[c for c in history[1:] if c.get('shadow_training_progress')]
        if shadows:
            chosen=min(shadows,key=lambda c:(max(c['severity'].values()),sum(c['severity'].values())))
            md=copy.deepcopy(models[chosen['context_arm']]);md['coefficients']=np.array(chosen['coefficients']).reshape(-1,3)
            md.update(accepted=False,learning_method='worst_corner_complete_response_parameter_learning',training_branch_only=True)
            torch.save(md,stage/'shadow_training_policy.pt')
            by={r['path']:r for r in chosen['records']}
            full=[copy.deepcopy(by.get(r['path'],r)) for r in baseline]
            write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
                severity=chosen['severity'],original_all_guard_pass=chosen['original_all_guard_pass'],
                adopted=False,not_protected_research_candidate=True,goal_complete=False))
        receipt(stage,dict(status='completed_shape_minimax_training_branch',new_policies=len(history)-1,
            shadow_training_candidates=len(shadows),shadow_selected=chosen['tag'] if shadows else None,
            shadow_sha256=sha(stage/'shadow_training_policy.pt') if shadows else None,
            research_acceptable=False,adopted=False,goal_complete=False,new_suffixes=0,new_stochastic_episodes=0,
            ppo_updates=0,source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_shape_minimax_training_branch',error=str(exc),
            new_policies=max(0,len(history)-1),adopted=False,goal_complete=False,wall_s=time.perf_counter()-started))
        raise


if __name__=='__main__':mp.freeze_support();main()
