"""Audit saved early/late complete-loop credit; no physical reruns."""
import numpy as np
from duration_joint_metrics import guard_names
from study_io import BASE,write_json,sha
from search_complete_return import read


def constraint_names(paths):
    original=['incoming','outgoing','reverse_deg','energy_01','peak_01','apex_speed',
              'loose_time','loose_cruise','first_inside','corridor','all_lateral']
    return ([f'{p}:{k}' for p in paths for k in original]+
            [f'{p}:{k}' for p in paths for k in guard_names()]+
            [f'{p}:{k}' for p in paths for k in ('intrinsic_peak','intrinsic_energy')])


def main():
    stage=BASE/'canonical_gate_band_preemptive_joint_credit_learning_20260920'
    result=read(stage/'state_joint/results.json');history=result['candidates']
    zero=history[0];by_tag={r['tag']:r for r in history}
    paths=[r['path'] for r in zero['records']];names=constraint_names(paths)
    model=read(stage/'joint_credit_models.json')
    jac=np.asarray(model['base_constraints']).reshape(len(names),-1)
    lat=np.asarray(model['base_lateral']).reshape(len(paths),-1)
    steps=np.asarray(model['steps']);report=[]
    for c in history[-4:]:
        u=np.asarray(c['coefficients']).reshape(3,7,7,3)[:,0,[0,1,5,6]].reshape(-1)/steps.reshape(-1)
        predicted=np.asarray(zero['augmented_learning_inequalities'])+jac@u
        actual=np.asarray(c['augmented_learning_inequalities'])
        values=np.asarray(c['augmented_measured_inequalities'])
        violations=[dict(guard=names[i],value=float(values[i]),learning_actual=float(actual[i]),
                         predicted=float(predicted[i])) for i in np.flatnonzero(values < -1e-8)]
        records=[]
        for r,b in zip(c['records'],zero['records']):
            s,t=r['shape'],b['shape']
            records.append(dict(path=r['path'],complete=r['success'],
                lateral_change_mm=s['reverse_lateral_significant_mm']-t['reverse_lateral_significant_mm'],
                lateral_duration_change_ms=1000*(s['reverse_lateral_significant_duration_s']-t['reverse_lateral_significant_duration_s']),
                exit_change_mm=s['out_edge_outward_mm']-t['out_edge_outward_mm'],
                reverse_change_deg=s['full_physical_reverse_deg']-t['full_physical_reverse_deg'],
                time_change_s=r['complete_time_s']-b['complete_time_s'],
                cruise_change_s=r['motion']['cruise_ge99_s']-b['motion']['cruise_ge99_s']))
        row=dict(tag=c['tag'],kind=c['kind'],violations=violations,records=records,
                 predicted_lateral_change=float((lat@u).sum()),loss_change=c['score']-zero['score'])
        report.append(row)
        print('CREDIT_AUDIT',c['kind'],len(violations),violations,flush=True)
    joint,_,early,late=history[-4:]
    interaction=np.asarray(joint['augmented_learning_inequalities'])-early['augmented_learning_inequalities']-np.asarray(late['augmented_learning_inequalities'])+zero['augmented_learning_inequalities']
    ids=np.argsort(abs(interaction))[-10:][::-1]
    write_json(stage/'matched_credit_audit.json',dict(receipt_sha256=sha(stage/'receipt.json'),arms=report,
        interaction=[dict(guard=names[i],value=float(interaction[i])) for i in ids],new_physical_runs=0,
        new_parameter_vectors=len(history)-1,selected=result['best'],
        interpretation='Measured earlier joint actions reduce incoming wrong-way movement, but the '
        'minimum-context linear model does not preserve every complete-route guard. Early/late '
        'controls identify effects only for these coefficients and seven training routes. '
        'This is neither PPO estimator bias evidence nor final geometric acceptance.'))


if __name__=='__main__':main()
