"""Replay the saved response fit and quantify local information at accepted moves."""
import numpy as np
from learn_verified_trust_joint import positive_sum
from duration_joint_metrics import guard_names
from study_io import BASE,write_json,sha
from search_complete_return import read


STAGE=BASE/'canonical_gate_band_coupled_early_late_joint_completion_20260920'


def labels(records):
    corners=[r for r in records if 'shape' in r];multi=[r for r in records if 'shape' not in r]
    first=['entry_outward','exit_outward','reverse_deg','energy','spatial_curvature','apex_speed',
           'time_with_20ms_margin','cruise_with_20ms_margin','first_turn_inside','corridor','negative_lateral']
    names=[f"path{r['path']}.{key}" for r in corners for key in first]
    names += [f"path{r['path']}.{key}" for r in corners for key in guard_names()]
    names += [f"path{r['path']}.intrinsic_{key}" for r in corners for key in ('peak','energy')]
    for r in multi:
        names += [f"path{r['path']}.multi_{key}" for key in ('time','cruise','corridor','success')]
        for w in r['multi_corner_windows']:
            names += [f"path{r['path']}.corner{w['vertex']}.{key}" for key in (
                'entry_outward','exit_outward','reverse_deg','energy','spatial_curvature','negative_lateral',
                'apex_speed','first_turn_inside','coarse_curvature','coarse_energy','negative_lateral_duration')]
        names += [f"path{r['path']}.{key}" for key in ('complete_time','cruise','longest_cruise')]
    return names


def main():
    history=read(STAGE/'state_joint/results.json')['candidates'];by={c['tag']:c for c in history}
    updates=read(STAGE/'trust_updates.json');credit=read(STAGE/'measured_coupled_credit.json')
    steps=np.array(credit['steps']);prior=np.column_stack([np.array(credit['jacobian']).T,credit['loss_gradient']])
    normalizer=by['protected_reference']['training_score'];rows=[]
    for block in updates:
        if block['arm']!='joint':continue
        anchor=by[block['anchor']];tested=by[block['measured']]
        available=history[:next(i for i,c in enumerate(history) if c['tag']==tested['tag'])]
        if block['iteration']==0:available=[c for c in available if c['tag']!='early_only_control']
        u=np.array(anchor['coefficients']);values=np.array(anchor['augmented_learning_inequalities'])
        near=[c for c in available if c['physical_pass'] and c['tag']!=anchor['tag'] and
              np.max(abs((np.array(c['coefficients'])-u)/steps))<=4.]
        x=np.array([(np.array(c['coefficients'])-u)/steps for c in near])
        y=np.array([np.r_[np.array(c['augmented_learning_inequalities'])-values,
                                c['training_score']-anchor['training_score']] for c in near])
        weight=1./np.maximum(np.linalg.norm(x,axis=1),.25);gram=(x*weight[:,None]).T@x
        model=np.linalg.solve(gram+.05*np.eye(54),(x*weight[:,None]).T@y+.05*prior)
        jac,gradient=model[:,:-1].T,model[:,-1]
        delta=(np.array(tested['coefficients'])-u)/steps
        predicted=.1*(positive_sum(values)-positive_sum(values+jac@delta))-gradient@delta/normalizer
        assert abs(predicted-block['predicted_drop'])<1e-8,(block['iteration'],predicted,block['predicted_drop'])
        eigen=np.maximum(np.linalg.eigvalsh(gram),0.)
        row=dict(iteration=block['iteration'],anchor=anchor['tag'],samples=len(near),
            design_rank=int(np.linalg.matrix_rank(x,tol=1e-10)),
            prior_effective_dimensions=float(np.sum(.05/(eigen+.05))),
            max_shift_from_credit_anchor=float(np.max(abs((u-np.array(by[credit['anchor']]['coefficients']))/steps))),
            predicted_drop=float(predicted),actual_drop=block['actual_drop'],ratio=block['ratio'],accepted=block['accepted'])
        rows.append(row);print('COUPLED_MODEL_REPLAY',row,flush=True)
    chosen=by[read(STAGE/'receipt.json')['training_best']]
    names=labels(chosen['records']);values=np.array(chosen['augmented_measured_inequalities'])
    assert len(names)==len(values)
    worst=[dict(channel=names[i],margin=float(values[i])) for i in np.argsort(values)[:15]]
    write_json(STAGE/'local_response_information_audit.json',dict(records=rows,worst_final_guards=worst,
        final_failed_guard_count=int(np.count_nonzero(values < -1e-8)),
        results_sha256=sha(STAGE/'state_joint/results.json'),new_physical_runs=0,
        scope='Exact reconstruction of this deterministic response-model fit. Rank and stale-prior dependence '
        'are local information diagnostics, not proof of an irreducible policy-class limit or PPO estimator bias.'))
    print('COUPLED_FINAL_GUARDS',worst,flush=True)


if __name__=='__main__':main()
