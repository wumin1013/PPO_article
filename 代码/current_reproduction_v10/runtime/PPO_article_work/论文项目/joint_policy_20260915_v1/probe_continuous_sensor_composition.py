"""Whole-policy validation of a new continuous-sensor credit direction."""
import argparse
import time
import numpy as np
import torch
from continuous_sensor_runtime import environment_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='continuous_sensor_current_parameter_solve_20260917')
    parser.add_argument('--parent',default='continuous_sensor_matched_function_transfer_20260917')
    parser.add_argument('--arm',default='updated')
    parser.add_argument('--stage',default='continuous_sensor_complete_composition_20260917')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/args.source
    optimization=read(source/'optimizer.json')
    assert optimization['eligible_for_small_probe']
    parent=BASE/args.parent
    selected=next(r for r in read(parent/'results.json')['candidates']
        if r.get('arm',r.get('tag'))==args.arm or r.get('model')==args.arm)
    model=parent/selected['model'];ids=[r['original_path'] for r in selected['results']]
    assert optimization['parent_sha256']==sha(model),'Credit must match the evaluated parent'
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    policy_class=KernelLocalFeedbackPolicy;current_factory=environment_factory
    if metadata['architecture'] in ('canonical_gate_kernel','gate_band_canonical_kernel'):
        from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
        policy_class=CanonicalGatePolicy;current_factory=checkpoint_environment_factory(metadata['architecture'])
    stage=register(args.stage,
        [__file__,source/'direction.npz',source/'optimizer.json',source/'receipt.json',
         parent/'results.json',model,BASE/'continuous_sensor_runtime.py',
         BASE/'probe_direct_constraint_composition.py',BASE/'learn_single_decision_credit.py',
         BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py',BASE/'gate_band_state.py'],
        dict(steps=[-.125,.125],screen_paths=ids,evaluation_actor_calls=6000,
             evaluation_physics_ticks=140000,wall_s=120,parameter_directions=1,
             current_credit_paths=optimization['path_ids'],checkpoint_architecture=metadata['architecture']),
        'The explicit current credit-path list and checkpoint architecture define '
        'the paired experiment. Canonical-gate models '
        'must use both their canonical continuous state and continuous guide; never '
        'load them in the old reference-frame environment. '+
        'Test one learned current-credit parameter direction at equal small signs '
        'using the continuous sensor in every policy call and unchanged physical '
        'execution. Current credit covers the registered routes; all29 saved training '
        'screen routes are evaluated to expose transfer interference. Reuse nominal '
        'trajectories. Measure full XY shape and raw/physical curvature, corridor, '
        'time and cruise. Report signed complete-policy response versus prediction '
        'on credit routes; no old gradients or borrowed uncertainty. Require positive '
        'aggregate entry/exit/reverse and directed-heading violation improvement '
        'when the parent violation is nonzero. Require every corner shape metric '
        'nonworse, including both directed headings and all four curvatures, '
        'full29/29 and total time<=102% '
        'parent to proceed. This is a composition diagnostic, not final shape or '
        'cruise acceptance. No larger step, ridge/width search, development or '
        'sealed evaluation is authorized by this script.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    with np.load(source/'direction.npz') as z:
        direction=z['direction'].copy();names=z['metric_names'].tolist()
        predicted=dict(zip(z['path_id'].tolist(),z['predicted_changes'].tolist()))
        scaled_predictions=({float(step):dict(zip(z['path_id'].tolist(),values.tolist()))
            for step,values in zip(z['prediction_scales'],z['predictions'])}
            if 'prediction_scales' in z else None)
    p=policy_class(256);saved=metadata['policy']
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    baseline=selected.get('summary',selected).copy();bm=metric_rows(selected['results'],[BASE/t for t in selected['traces']],baseline,names)
    candidates=[]
    for step in (-.125,.125):
        p.load_state_dict(saved)
        with torch.no_grad():p.residual_coefficients.add_(torch.tensor(step*direction,dtype=torch.float32))
        tag=f'step{step:+g}'
        save_torch(stage/f'{tag}.pt',dict(policy=p.state_dict(),architecture=metadata['architecture'],centers=256,
            observation_encoding=metadata['observation_encoding'],guidance_encoding=metadata.get('guidance_encoding'),parent_sha256=sha(model),
            state_support_encoding=metadata.get('state_support_encoding'),
            learning_method=optimization.get('learning_method','current_continuous_sensor_constrained_credit'),coefficient=step,diagnostic_only=True))
        rs=evaluate(p,[np.asarray(paths[i]) for i in ids],stage,tag,None,costs,start+120,current_factory)
        rs=[dict(r,original_path=i) for r,i in zip(rs,ids)];ts=[stage/tag/f'{i}.npz' for i in range(len(ids))]
        value=review(rs,ts);metrics=metric_rows(rs,ts,value,names)
        r=dict(step=step,summary=value,metrics=metrics,results=rs,traces=[str(t.relative_to(BASE)) for t in ts])
        candidates.append(r)
        write_json(stage/'results.json',dict(baseline=baseline,baseline_metrics=bm,candidates=candidates,
            metric_names=names,predicted=predicted,scaled_predictions=scaled_predictions,
            prediction_kind=optimization.get('prediction_kind','linear_current_scalar_credit'),
            costs=costs,goal_complete=False))
        print('CONTINUOUS_COMPOSITION',step,{k:value[k] for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s')},flush=True)
    comparisons=[];regressions=[]
    for j,path in enumerate(ids):
        lo,hi=[c['metrics'][j] for c in candidates]
        if not(lo['success'] and hi['success']):comparisons.append(dict(path=path,complete_pair=False));continue
        old=np.asarray(bm[j]['values']);positive=np.asarray(hi['values'])-old
        comparisons.append(dict(path=path,complete_pair=True,positive_change=positive.tolist(),
            central_per_unit=((np.asarray(hi['values'])-np.asarray(lo['values']))/.25).tolist(),
            predicted_per_unit=predicted.get(path) if scaled_predictions is None else None,
            predicted_positive_change=(scaled_predictions[.125].get(path) if scaled_predictions is not None
                else (np.asarray(predicted[path])*.125).tolist() if path in predicted else None),
            predicted_central_per_unit=((np.asarray(scaled_predictions[.125][path])-
                np.asarray(scaled_predictions[-.125][path]))/.25).tolist()
                if scaled_predictions is not None and path in predicted else predicted.get(path)))
        for m,name in enumerate(names[:names.index('segment_max_distance_mm')]):
            if positive[m]>1e-6:regressions.append(dict(path=path,metric=name,change=float(positive[m]),
                supplied_credit=path in predicted))
    plus=candidates[1]['summary']
    improvement_names=[name for name in ('in_edge_outward_mm','out_edge_outward_mm',
        'full_physical_reverse_deg','negative_signed_heading_deg','heading_overshoot_deg') if name in names]
    aggregate_shape=[]
    for name in improvement_names:
        index=names.index(name)
        old=sum(row['values'][index] for row in bm if row['success'])
        new=sum(row['values'][index] for row in candidates[1]['metrics'] if row['success'])
        aggregate_shape.append(dict(metric=name,baseline=old,positive=new,
            passed=bool(new<old if old>1e-8 else new<=1e-8)))
    good=bool(plus['successes']==len(ids) and candidates[0]['summary']['successes']==len(ids) and
        not regressions and all(row['passed'] for row in aggregate_shape) and
        plus['full_time_s']<=1.02*baseline['full_time_s'])
    write_json(stage/'comparison.json',dict(comparisons=comparisons,regressions=regressions,
        metric_names=names,aggregate_shape=aggregate_shape,passed_composition=good,goal_complete=False))
    assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=140000
    receipt(stage,dict(status='completed_continuous_sensor_composition_check',costs=costs,
        passed_composition=good,regressions=len(regressions),wall_s=time.perf_counter()-start,goal_complete=False))
    print('CONTINUOUS_COMPOSITION_CHECK',good,regressions,flush=True)


if __name__=='__main__':main()
