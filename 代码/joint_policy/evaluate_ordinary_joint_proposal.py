"""Full feedback validation of a learned ordinary-motion proposal, all outcomes kept."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from train_pilot import evaluate, save_torch
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from audit_current_command_coordinates import command_signals
from search_complete_return import read
from study_io import BASE, ROOT, register, receipt, write_json, sha


def verified_terminal_reuse(source, proposal, probe, metadata):
    """Permit diagnostic feedback of a support-limited model; never relabel it feasible."""
    saved = read(probe/'results.json')
    registered = read(probe/'registration.json')
    completed = read(probe/'receipt.json')
    assert completed['complete_success']
    assert completed['registration_sha256'] == sha(probe/'registration.json')
    assert all(sha(ROOT/p) == digest for p, digest in registered['source_sha256'].items())
    assert saved['source_direction_sha256'] == sha(source/'direction.npz')
    assert saved['parent_sha256'] == proposal['parent_sha256'] and saved['scale'] == 1.
    last = proposal['history'][-1]
    assert last['conditional_model_feasible']
    rows = last['exact_branch_checks']['records']
    missing = [r['path'] for r in rows if not r['supported']]
    assert 1 <= len(missing) <= 2
    assert all(not r.get('shape_violations') for r in rows)
    assert all(r['prefix_inside'] for r in rows if r['supported'])
    checkpoint = torch.load(probe/'diagnostic_candidate.pt',map_location='cpu',weights_only=False)
    assert checkpoint['diagnostic_only']
    assert checkpoint['source_direction_sha256'] == saved['source_direction_sha256']
    expected = CanonicalGatePolicy(256); expected.load_state_dict(metadata['policy'])
    with np.load(source/'direction.npz') as z:
        with torch.no_grad(): expected.residual_coefficients.add_(torch.tensor(z['direction'],dtype=torch.float32))
    assert all(torch.equal(value,checkpoint['policy'][key]) for key,value in expected.state_dict().items())
    records = {r['path']:r for r in saved['records']}
    assert sorted(records) == sorted(missing)
    for r in records.values():
        assert r['result']['success'] and r['strict_vector_limits'] and r['result']['audit']['rest_pass']
    return records


def checks(candidate, baseline, names):
    summary = candidate['summary']; old = baseline['summary']
    complete = summary['comparable_complete_pool']
    out = dict(all_complete=complete)
    if not complete:
        out['eligible']=False
        return out
    bounds = {r['path']:np.array(r['values']) for r in baseline['metrics']}
    regressions=[]
    for r in candidate['metrics']:
        if r['path'] >= 16:continue
        delta=np.asarray(r['values'])[:9]-bounds[r['path']][:9]
        for k in np.flatnonzero(delta > 1e-6):
            regressions.append(dict(path=r['path'],metric=names[k],increase=float(delta[k])))
    totals = ('entry_sum_mm','exit_sum_mm','full_reverse_sum_deg')
    ratios={key:summary[key]/old[key] for key in totals if old[key]>1e-8}
    out.update(shape_regressions=regressions,each_shape_nonworse=not regressions,
        positive_shape_totals_reduced_one_percent=all(v<=.99 for v in ratios.values()),
        remaining_shape_fractions=ratios,
        complete_time_nonworse=summary['full_time_s']<=old['full_time_s']+1e-9,
        cruise99_nonworse=summary['complete_cruise_s']['99']>=old['complete_cruise_s']['99']-1e-9,
        strict_vector_limits=all(np.all(np.array(r['audit']['peaks'])<=np.array([100.,2000.,62500.]))
            for r in candidate['results']),
        full_stop=all(r['audit']['rest_pass'] for r in candidate['results']))
    out['eligible']=all(out[k] for k in ('each_shape_nonworse','positive_shape_totals_reduced_one_percent',
        'complete_time_nonworse','cruise99_nonworse','strict_vector_limits','full_stop'))
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='canonical_gate_band_ordinary_joint_proposal_20260918')
    parser.add_argument('--stage',default='canonical_gate_band_ordinary_joint_feedback_20260918')
    parser.add_argument('--scales',type=float,nargs='+',default=[.1,.25,.5,1.])
    parser.add_argument('--reuse-terminal-probe')
    args=parser.parse_args()
    assert len(args.scales)<=4 and all(0<s<=1 for s in args.scales)
    torch.set_num_threads(1)
    source=BASE/args.source
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    proposal,data=read(source/'results.json'),read(parent/'results.json')
    model=BASE/proposal['parent_model']
    assert sha(model)==proposal['parent_sha256']==sha(parent/data['chosen']['model'])
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    reused={};reuse_sources=[]
    if args.reuse_terminal_probe:
        assert args.scales == [1.] and not proposal['eligible_for_actual_probe']
        probe=BASE/args.reuse_terminal_probe
        reused=verified_terminal_reuse(source,proposal,probe,metadata)
        reuse_sources=[probe/'results.json',probe/'registration.json',probe/'receipt.json',
                       probe/'diagnostic_candidate.pt']+[BASE/r['trace'] for r in reused.values()]
    else:
        assert proposal['eligible_for_actual_probe']
    scales=tuple(args.scales)
    stage=register(args.stage,
        [__file__,model,parent/'results.json',parent/'receipt.json',source/'direction.npz',
         source/'results.json',source/'receipt.json',BASE/'train_pilot.py',
         BASE/'canonical_gate_state.py',BASE/'joint_runtime.py',BASE/'validate_actions.py']+reuse_sources,
        dict(scales=list(scales),screen_paths=[r['original_path'] for r in data['chosen']['results']],
             evaluation_actor_calls=18000,evaluation_physics_ticks=450000,wall_s=600,
             maximum_additional_full_paths=35,automatic_adoption=False,
             reused_paths=sorted(reused),predictor_eligible=proposal['eligible_for_actual_probe']),
        ('The complete-branch predictor remains ineligible because the explicitly registered paths lack later '
         'ordinary support. Their already completed actual full rollouts succeeded; '
         'verify exact parameter equality, direction and registered runtime hashes, '
         'reuse those traces and evaluate only the remaining screen paths. This is '
         'diagnostic validation of the changed complete-tail proposal, not a claim '
         'that the predictor passed or its error is bounded. ' if reused else '')+
        'Execute the explicitly registered fixed scales of the jointly learned current399a '
        'proposal using unchanged three-action full feedback, guide and frozen '
        'executor. Original terminal planner chooses the actual complete brake '
        'and hold. Failed prefixes have no comparable time or shape summary. '
        'Keep all original1ms XY and actions, full per-corner shape/curvature, '
        'speed, cruise, complete time and independent exact-limit difference '
        'checks. Intermediate promotion requires all29 complete, EVERY existing '
        'corner shape and curvature nonworse within1e-6 numerical slack, positive '
        'entry/exit/reverse totals at least1percent lower, nonincreasing time '
        'and nondecreasing actual99mm/s cruise. Select by worst shape fraction '
        'then time, and check remaining35 training paths with the identical '
        'criteria versus saved current399a full64. Do not adopt before XY review. '
        'No sealed or unseen tuning access; no final success claim.')
    start=time.perf_counter();deadline=start+595
    costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    factory=checkpoint_environment_factory(metadata['architecture'])
    policy=CanonicalGatePolicy(256)
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    ids=[r['original_path'] for r in data['chosen']['results']]
    names=data['metric_names']
    with np.load(source/'direction.npz') as z:direction=z['direction'].copy()
    candidates=[];chosen=None;full_candidate=None
    def audit(item_ids,tag):
        fresh=[i for i in item_ids if i not in reused]
        results=evaluate(policy,[np.asarray(paths[i]) for i in fresh],stage,tag,None,costs,deadline,factory)
        combined={i:(dict(r,original_path=i),stage/tag/f'{j}.npz')
                  for j,(r,i) in enumerate(zip(results,fresh))}
        combined.update({i:(r['result'],BASE/r['trace']) for i,r in reused.items() if i in item_ids})
        return [combined[i][0] for i in item_ids],[combined[i][1] for i in item_ids]
    def pack(results,traces):
        summary=review(results,traces)
        return dict(results=results,traces=[str(t.relative_to(BASE)) for t in traces],summary=summary,
                    metrics=metric_rows(results,traces,summary,names))
    def persist():
        write_json(stage/'results.json',dict(parent_sha256=sha(model),parent_model=proposal['parent_model'],
            baseline=data['chosen'],candidates=candidates,chosen=chosen,full_candidate=full_candidate,
            reused_paths=sorted(reused),predictor_eligible=proposal['eligible_for_actual_probe'],
            metric_names=names,costs=costs,learned_candidate_adopted=False,goal_complete=False))
    try:
        for scale in scales:
            policy.load_state_dict(metadata['policy'])
            with torch.no_grad():policy.residual_coefficients.add_(torch.tensor(scale*direction,dtype=torch.float32))
            name=f'candidate_{scale:g}.pt'
            save_torch(stage/name,dict(metadata,policy=policy.state_dict(),parent_sha256=sha(model),
                learning_method=('complete_branch_constrained_feedback_proposal' if 'history' in proposal else
                                 'ordinary_joint_constrained_complete_feedback_proposal'),
                requires_physical_validation=True,source_stage=source.name,
                source_direction_sha256=sha(source/'direction.npz')))
            results,traces=audit(ids,f'screen_{scale:g}')
            item=dict(step=scale,model=name,**pack(results,traces))
            item['checks']=checks(item,data['chosen'],names)
            commands=[]
            for r,trace in zip(results,traces):
                if r['original_path']>=16 or not r['success']:continue
                with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
                signal=command_signals(ref,rows,actions)
                apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
                use=(np.arange(len(rows))<apex)&(rows[:,12]==0)&(np.linalg.norm(rows[:,2:4],axis=1)>5.)
                commands.append(dict(path=r['original_path'],
                    physical_negative_heading_ms=int((use&(signal['physical_heading_deg']<-.05)).sum()),
                    combined_negative_target_ms=int((use&(signal['combined_deg']<-.05)).sum())))
            item['commands']=commands
            candidates.append(item);persist()
            s=item['summary']
            print('ORDINARY_FEEDBACK',scale,{k:s[k] for k in ('successes','exit_sum_mm',
                'full_reverse_sum_deg','full_time_s','complete_cruise_s')},
                'eligible',item['checks']['eligible'],'regressions',item['checks'].get('shape_regressions',[])[:5],flush=True)
            assert costs['evaluation_actor_calls']<=18000 and costs['evaluation_physics_ticks']<=450000
        eligible=[c for c in candidates if c['checks']['eligible']]
        if eligible:
            chosen=min(eligible,key=lambda c:(max(c['checks']['remaining_shape_fractions'].values()),c['summary']['full_time_s']))
            missing=[i for i in range(64) if i not in ids]
            policy.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
            extra,traces=audit(missing,'remaining_training')
            combined={r['original_path']:(r,BASE/t) for r,t in zip(chosen['results'],chosen['traces'])}
            combined.update({r['original_path']:(r,t) for r,t in zip(extra,traces)})
            results,traces=zip(*(combined[i] for i in range(64)))
            full_candidate=dict(step=chosen['step'],model=chosen['model'],**pack(list(results),list(traces)))
            full_candidate['checks']=checks(full_candidate,data['full_candidate'],names)
            persist()
        receipt(stage,dict(status='completed_ordinary_joint_feedback',costs=costs,
            wall_s=time.perf_counter()-start,eligible_screen_scales=[c['step'] for c in eligible],
            full_candidate_eligible=bool(full_candidate and full_candidate['checks']['eligible']),
            learned_candidate_adopted=False,goal_complete=False))
    except Exception as error:
        persist()
        receipt(stage,dict(status='failed_ordinary_joint_feedback',error=str(error),costs=costs,
            wall_s=time.perf_counter()-start,learned_candidate_adopted=False,goal_complete=False))
        raise


if __name__=='__main__':main()
