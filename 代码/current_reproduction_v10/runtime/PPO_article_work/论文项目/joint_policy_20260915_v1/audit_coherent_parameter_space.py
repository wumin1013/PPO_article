"""Bounded, saved-data feasibility audit of whole-policy parameter responses."""
import argparse
import time
import numpy as np
from scipy.optimize import linprog
from learn_canonical_credit_step import compare
from solve_continuous_sensor_credit import IMPROVEMENT_NAMES
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--guard-ablation',action='store_true')
    args=parser.parse_args()
    source = BASE/'canonical_gate_band_coherent_parameter_credit_20260917'
    data = read(source/'results.json')
    assert read(source/'receipt.json')['complete_paths'] == 348
    model = BASE/data['parent_model']
    assert sha(model) == data['parent_sha256']
    stage_name=('canonical_gate_band_coherent_guard_ablation_20260917' if args.guard_ablation
                else 'canonical_gate_band_coherent_subspace_audit_20260917')
    stage = register(stage_name,
        [__file__, source/'results.json', source/'receipt.json', source/'basis.npz',
         source/'basis.json', model, BASE/'learn_canonical_credit_step.py'],
        dict(new_actor_decisions=0, new_physics_ticks=0, wall_s=60,
             maximum_sum_absolute_probe_units=1., maximum_saved_action_change=.125,
             maximum_parameter_metric_norm=.025, maximum_common_reduction=.2,
             guard_ablation=args.guard_ablation),
        'Use only completed current-center whole-policy responses. Compare central '
        'scalar secants with a piecewise one-sided response model using the same '
        'six measured directions and trust domain. Enforce per-corner nonincrease '
        'in all existing shape/curvature metrics and nonnegative predictions, '
        'nonincreasing predicted total time and half corridor margin. Maximize '
        'common reduction in the five existing shape totals. L1 probe units<=1 '
        'keeps each coordinate inside measured amplitude and guarantees the '
        'registered support/norm limits. Neither scalar model is a physical '
        'guarantee. Record actual per-direction regressions and cruise. Do not '
        'promote a diagnostic checkpoint or add directions after an unfavorable solve. '
        'When guard_ablation is true, omit ONLY the optimizer nonincrease guards '
        'for curvature peaks, curvature energies, or both in separate matched '
        'saved-data controls. Keep their nonnegativity plus all directed shape, '
        'corridor and time conditions. These guards are learning heuristics, not '
        'the physical V/A/J limits. This ablation authorizes no physical rollout '
        'or changed acceptance rule; it localizes the zero-progress constraint.')
    start = time.perf_counter()
    names = data['metric_names']; baseline = data['baseline']
    ids = [r['path'] for r in baseline['metrics']]
    base = np.array([r['values'] for r in baseline['metrics']])
    variants = {(r['direction'], r['sign']): r for r in data['candidates']}
    change = {}
    audit = []
    for key, r in variants.items():
        assert [m['path'] for m in r['metrics']] == ids
        change[key] = np.array([m['values'] for m in r['metrics']])-base
        checks = compare(r, baseline, names, required_reduction=0.)
        audit.append(dict(direction=key[0], sign=key[1], checks=checks,
            time_s=r['summary']['full_time_s'], cruise_s=r['summary']['complete_cruise_s']))
    central = np.stack([(change[j,1]-change[j,-1])*.5 for j in range(6)], axis=-1)
    # Positive/negative variables are nonnegative probe units, not new parameters.
    models = dict(central=np.concatenate([central, -central], axis=-1),
        one_sided=np.stack([change[j,s] for s in (1,-1) for j in range(6)], axis=-1))
    if args.guard_ablation:
        models={f'{kind}_omit_{omit}':g for kind,g in models.items()
                for omit in ('peak','energy','curvature')}
    outputs = {}
    for kind, g in models.items():
        a=[]; b=[]; labels=[]
        def add(row, bound, label, eta=0.):
            a.append(np.r_[row,eta]); b.append(float(bound)); labels.append(label)
        for i,p in enumerate(ids):
            for m in range(names.index('segment_max_distance_mm')):
                omitted=(args.guard_ablation and (
                    (kind.endswith('_peak') and 'peak_curvature' in names[m]) or
                    (kind.endswith('_energy') and 'normalized_energy' in names[m]) or
                    (kind.endswith('_curvature') and ('peak_curvature' in names[m] or 'normalized_energy' in names[m]))))
                if not omitted:add(g[i,m], 0., f'{p}_{names[m]}')
                add(-g[i,m], base[i,m], f'{p}_{names[m]}_nonnegative')
            m=names.index('segment_max_distance_mm')
            add(g[i,m], .5*(1.25-base[i,m]), f'{p}_corridor')
        for name in IMPROVEMENT_NAMES:
            m=names.index(name)
            add(g[:,m].sum(0), 0., 'total_'+name, base[:,m].sum())
        add(g[:,-1].sum(0),0.,'total_time')
        add(np.ones(12),1.,'probe_L1')
        a=np.asarray(a); b=np.asarray(b)
        scale=np.maximum(np.linalg.norm(a,axis=1),1e-10)
        fit=linprog(np.r_[np.zeros(12),-1.], A_ub=a/scale[:,None], b_ub=b/scale,
            bounds=[(0.,1.)]*12+[(0.,.2)], method='highs',
            options={'time_limit':20., 'dual_feasibility_tolerance':1e-9,
                     'primal_feasibility_tolerance':1e-9})
        result=dict(success=bool(fit.success),message=str(fit.message))
        if fit.success:
            x=fit.x[:6]-fit.x[6:12]
            result.update(common_fraction=float(fit.x[-1]),probe_units=x.tolist(),
                split_units=fit.x[:12].tolist(),raw_max_violation=float(max(0.,(a@fit.x-b).max())),
                active_constraints=[dict(name=n,dual=float(d)) for n,d in
                    zip(labels,fit.ineqlin.marginals/scale) if abs(d)>1e-5],
                predicted_change=np.einsum('pmj,j->pm',g,fit.x[:12]).tolist())
            with np.load(source/'basis.npz') as z:
                direction=(x*z['amplitudes'])@z['basis']
                result['metric_norm']=float(np.sqrt(direction@z['metric']@direction))
                result['maximum_saved_action_change']=float(abs(z['support']@direction).max())
            np.savez_compressed(stage/f'{kind}_direction.npz',direction=direction.reshape(3,256),
                probe_units=x,path_id=ids,metric_names=np.array(names))
        outputs[kind]=result
        print('COHERENT_SPACE',kind,{k:v for k,v in result.items()
            if k not in ('active_constraints','predicted_change','split_units')},flush=True)
    write_json(stage/'results.json',dict(parent_sha256=sha(model),actual_variants=audit,
        models=outputs,baseline_shape_totals={n:float(base[:,names.index(n)].sum())
        for n in IMPROVEMENT_NAMES},goal_complete=False))
    receipt(stage,dict(status='completed_coherent_subspace_saved_audit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start,
        goal_complete=False))
    if args.guard_ablation:return
    status_file=BASE/'当前执行状态.json'; status=read(status_file)
    status.update(running=None,active_experiment=None,observed_research_processes=[],
        next_action='Inspect measured coherent subspace feasibility and current real shape; no automatic expansion or continuation.')
    status['latest_execution_findings']['current_research_model']=dict(
        model=data['parent_model'],sha256=sha(model),training_successes=64,
        final_shape_accepted=False,working_model_pointer_changed=False)
    status['latest_execution_findings']['coherent_parameter_credit']=dict(
        receipt=str((source/'receipt.json').relative_to(BASE)),
        complete_paths=348,wall_s=read(source/'receipt.json')['wall_s'],
        audit_receipt=str((stage/'receipt.json').relative_to(BASE)))
    write_json(status_file,status)


if __name__=='__main__':main()
