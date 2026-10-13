"""Audit saved local constraint prediction errors without physical reruns."""
import numpy as np
from geometric_turn_metrics import spatial_turn_profile
from learn_signed_slack_credit import signed_guards
from study_io import BASE, write_json
from search_complete_return import read


NAMES=('entry','exit','reverse','energy','curvature','apex_speed','time',
       'cruise','first_turn','corridor','lateral')


def peak(record):
    with np.load(record['trace']) as z:
        profile=spatial_turn_profile(np.vstack([z['reference'][0],z['rows'][:,:2]]))
    values=np.abs(profile['turn'])/.1
    indices=np.argsort(values)[-3:][::-1]
    return dict(indices=indices.tolist(),arc_mm=profile['grid'][1:-1][indices].tolist(),
                curvature=values[indices].tolist())


def main():
    stage=BASE/'canonical_gate_band_signed_slack_credit_learning_20260920'
    source=BASE/'canonical_gate_band_scaled_command_credit_learning_20260920'
    audit=read(stage/'signed_slack_audit.json')
    history=read(source/'innovation_coordinates/results.json')['candidates']
    current=read(stage/'innovation_coordinates/results.json')['candidates']
    zero=current[0];cache={};g0=signed_guards(zero,cache)
    jac=np.asarray(audit['jacobian'])[:,1:];steps=np.asarray(read(source/'local_response_audit.json')['steps'])
    even=np.array([.5*(signed_guards(next(c for c in history if c['tag']==f'axis_{i}_+1'),cache)+
                      signed_guards(next(c for c in history if c['tag']==f'axis_{i}_-1'),cache))-g0
                   for i in range(36)])
    rows=[]
    for c in current[1:]:
        u=np.array(c['coefficients']);g=signed_guards(c,cache);linear=g0+u@jac
        quadratic=linear+(u/steps)**2@even
        failures=[]
        for j in np.flatnonzero(np.array(c['measured_inequalities']) < -1e-8):
            item=dict(path=c['records'][j//11]['path'],metric=NAMES[j%11],
                      actual=float(g[j]),linear=float(linear[j]),diagonal_quadratic=float(quadratic[j]),
                      absolute_linear_contribution=float(np.abs(u*jac[:,j]).sum()),
                      absolute_even_contribution=float(np.abs((u/steps)**2*even[:,j]).sum()))
            if j%11==4:
                item.update(baseline_peak=peak(zero['records'][j//11]),candidate_peak=peak(c['records'][j//11]))
            failures.append(item)
        rows.append(dict(tag=c['tag'],score=c['score'],failures=failures))
    result=dict(source='Saved 72 paired full-trajectory outcomes and three joint outcomes only',
                physical_reruns=0,validations=rows)
    write_json(stage/'constraint_prediction_audit.json',result)
    for r in rows:print(r,flush=True)


if __name__=='__main__':main()
