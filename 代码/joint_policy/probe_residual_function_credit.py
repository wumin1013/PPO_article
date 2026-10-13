"""Actual closed-loop credit for the expanded, skill-preserving learned function."""
import argparse
import time
from datetime import datetime,timezone
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from learn_single_decision_credit import SCREEN,review,adoption_checks
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recenter',action='store_true')
    parser.add_argument('--parent');parser.add_argument('--stage');parser.add_argument('--include-long',action='store_true')
    parser.add_argument('--headwise',action='store_true');parser.add_argument('--reuse-source');args=parser.parse_args()
    if args.headwise:assert args.parent and args.reuse_source, 'Headwise control requires explicit matched parent and existing joint probes'
    if args.parent:
        assert args.stage, 'A new parent requires a distinct registered stage'
        args.recenter=True
    screen=SCREEN+([38,39,42,43,44,45,46,47] if args.include_long else [])
    call_budget,tick_budget,wall_budget=(32000,800000,600) if args.include_long else (18000,420000,420)
    if args.headwise:call_budget,tick_budget,wall_budget=60000,1400000,900
    torch.set_num_threads(1)
    source=BASE/'protected_residual_capacity_initialization_20260917'
    assert read(source/'receipt.json')['initial_mean_error']==0.
    parent=BASE/('protected_residual_complete_function_credit_learning_20260917' if args.recenter else 'protected_complete_function_credit_learning_20260917')
    if args.parent:parent=BASE/args.parent
    selected=read(parent/'training_selection.json')
    initial=parent/selected['model'] if args.recenter else source/'initial.pt'
    stage=register(args.stage or ('recentered_residual_function_credit_probe_20260917' if args.recenter else 'protected_residual_function_credit_probe_20260917'),
        [__file__,source/'receipt.json',source/'results.json',source/'initial.pt',source/'basis.npz',
         parent/'training_selection.json',initial,BASE/'kernel_local_feedback_policy.py',
         BASE/'learn_single_decision_credit.py',BASE/'joint_runtime_v5.py']+
         ([BASE/args.reuse_source/'results.json',BASE/args.reuse_source/'receipt.json'] if args.reuse_source else []),
        dict(parameter_candidates=26 if args.headwise else 10,coefficient_amplitude=.5,evaluation_actor_calls=call_budget,
             evaluation_physics_ticks=tick_budget,wall_s=wall_budget,protected_paths=[16,31],screen_paths=screen),
        ('Matched proposal-space expansion: split the four coupled shape directions into their three nonzero output-head rows, retaining the average-feed direction. SAME parent, original coefficients, amplitudes, support,29 training paths and executor as the existing five-direction control. Thirteen directions expose whether fixed head ratios obstruct simultaneous entry/exit/reverse and curvature-distribution improvement. Reuse the two identical average-feed probes by checkpoint parent identity;24 new complete-function probes. No new hidden features, fitting labels or runtime rule. If smoothness-preserving improvement is still infeasible, stop this basis and revisit representation/credit support instead of extending it unchanged. ' if args.headwise else '')+
        ('Include eight already available long TRAINING routes omitted from previous function-credit measurements. Audit requested/executed speed before attributing deficits to execution. Pool extension and credit recentering both change here; do not claim single-factor superiority. Screen-only results can be extracted from the same traces without repeated simulation. ' if args.include_long and not args.headwise else '')+
        ('Recenter the five fixed proposal directions at the newly selected64/64,12/12 residual candidate after a substantial actual reduction in both edge excursions. Recompute full-feedback consequences because old-parent sensitivities no longer represent this policy. This is one bounded new-state credit round, not repeating completed branches. The same basis and frozen metric are retained; raw positive/negative curvature of the full-policy response will be audited before another update. ' if args.recenter else '')+
        'Measure complete learned-policy consequences of five expanded-feedback '
        'proposal directions at+/-0.5 coefficient. Same21 training routes, unchanged '
        'physics,full stop and objective. Base neural actor remains frozen; the768 '
        'reflection-equivariant state-feedback coefficients change all three outputs. '
        'Proposal construction preserves137 stored action states of paths16+31 in '
        'real arithmetic. Measure actual float32 action and physical XY preservation '
        'separately. Initialization reproduces current actor exactly without distillation. '
        'No off-policy local derivative is treated as a new-model gradient until these '
        'complete-function results exist. No development,sealed data or repeated '
        'single-action probes. Stop if useful safe directions do not survive actual '
        'feedback; no MSE improvement claim or task success from added capacity.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    p=KernelLocalFeedbackPolicy(centers=256);p.load_state_dict(saved['policy'])
    with np.load(source/'basis.npz') as z:prepared={k:z[k].copy() for k in z.files}
    basis=prepared['basis'];names=prepared['names'].tolist();cx=prepared['protected_observations']
    mapping=[dict(original_basis=i,head=None) for i in range(5)]
    reused=None
    if args.headwise:
        expanded=[];expanded_names=[];mapping=[]
        for i,name in enumerate(names):
            for head in range(3):
                if not np.any(basis[i,head]):continue
                direction=np.zeros_like(basis[i]);direction[head]=basis[i,head]
                expanded.append(direction);expanded_names.append(f'{name}:head{head}')
                mapping.append(dict(original_basis=i,head=head))
        basis=np.asarray(expanded);names=expanded_names;assert len(basis)==13
        reused=read(BASE/args.reuse_source/'results.json')
        prepared.update(basis=basis,names=np.asarray(names))
        np.savez_compressed(stage/'basis.npz',**prepared)
        write_json(stage/'basis_mapping.json',dict(mapping=mapping,normalization_changed=False,parent_sha256=sha(initial)))
    with torch.no_grad():initial_mean=p.distribution(torch.tensor(cx)).mean.numpy()
    base_rs=[selected['results'][i] for i in screen];base_traces=[BASE/selected['traces'][i] for i in screen]
    baseline=review(base_rs,base_traces);protected_rows={}
    for i in (16,31):
        with np.load(BASE/selected['traces'][i]) as z:protected_rows[i]=z['rows'].copy()
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];candidates=[]
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='protected_residual_function_credit_running',training_running=False,
        physical_diagnostic_running=True,diagnostic_stage=stage.name)
    status['running']=stage.name;status['active_experiment']=stage.name;status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)
    for i,name in enumerate(names):
        for sign in (-1,1):
            if args.headwise and mapping[i]['original_basis']==4:
                original=next(r for r in reused['candidates'] if r['basis']==4 and r['sign']==sign)
                old_checkpoint=BASE/args.reuse_source/f"{original['tag']}.pt"
                assert torch.load(old_checkpoint,map_location='cpu',weights_only=False)['parent_sha256']==sha(initial)
                assert [r['original_path'] for r in original['results']]==screen
                r=dict(original,basis=i,name=name,reused_from=args.reuse_source,source_model=str(old_checkpoint.relative_to(BASE)))
                candidates.append(r)
                write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,costs=costs,goal_complete=False))
                print('REUSED_IDENTICAL_FEED',i,sign,flush=True)
                continue
            p.load_state_dict(saved['policy'])
            with torch.no_grad():
                p.residual_coefficients.add_(torch.tensor(sign*.5*basis[i],dtype=torch.float32))
                error=float(abs(p.distribution(torch.tensor(cx)).mean.numpy()-initial_mean).max())
            tag=f'b{i}_{sign:+d}'
            save_torch(stage/f'{tag}.pt',dict(policy=p.state_dict(),architecture='kernel_local_feedback',centers=256,
                parent_sha256=sha(initial),basis=i,coefficient=sign*.5,learning_method='protected_residual_function_probe'))
            rs=evaluate(p,[np.asarray(paths[k]) for k in screen],stage,tag,None,costs,started+wall_budget,environment_factory)
            assert costs['evaluation_actor_calls']<=call_budget and costs['evaluation_physics_ticks']<=tick_budget
            rs=[dict(r,original_path=k) for r,k in zip(rs,screen)];traces=[stage/tag/f'{j}.npz' for j in range(len(screen))]
            checks=[]
            for path in (16,31):
                with np.load(traces[screen.index(path)]) as z:rows=z['rows']
                original=protected_rows[path]
                checks.append(dict(path=path,success=rs[screen.index(path)]['success'],
                    xy_max_error_mm=float(abs(rows[:,:2]-original[:,:2]).max()) if rows.shape==original.shape else None))
            r=dict(review(rs,traces),basis=i,name=name,sign=sign,coefficient=sign*.5,tag=tag,results=rs,
                traces=[str(t.relative_to(BASE)) for t in traces],protected_float32_latent_error=error,protected_paths=checks)
            r['checks']=adoption_checks(r,baseline);r['eligible']=all(r['checks'].values());candidates.append(r)
            write_json(stage/'results.json',dict(baseline=baseline,candidates=candidates,costs=costs,goal_complete=False))
            print('RESIDUAL_FUNCTION',name,sign,{k:r[k] for k in ('successes','full_time_s','entry_sum_mm',
                'exit_sum_mm','full_reverse_sum_deg','eligible')},'protected',checks,flush=True)
    receipt(stage,dict(status='completed_protected_residual_function_credit_probe',costs=costs,
        candidates=len(candidates),screen_eligible=sum(r['eligible'] for r in candidates),wall_s=time.perf_counter()-started))
    status=read(status_path);status['current'].update(state='protected_residual_function_credit_complete',physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['running']=None;status['active_experiment']=None;status['updated_utc']=datetime.now(timezone.utc).isoformat()
    status['next']='Analyze complete residual-function consequences and coordinate only useful feasible directions; preserve actual learned shape and cruise requirements.'
    write_json(status_path,status)


if __name__=='__main__':main()
