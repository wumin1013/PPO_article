"""Rebuild observed local credit using exact cruise-event slack, without reruns."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from cruise_duration_credit import event_learning_guards
from audit_context_restoration_scaling import solve_correction
from audit_remaining_joint_credit import names_for
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_bend_metric_joint_learning_20260920'
    history=read(stage/'state_joint/results.json')['candidates'];index={c['tag']:c for c in history}
    baseline=history[0]['records'];checked=0
    for c in history:
        c['event_inequalities']=event_learning_guards(c['records'],baseline,c['measured_inequalities']).tolist()
        checked+=len(c['records'])
    measured=read(stage/'measured_local_credit.json');anchor=index['local_joint_0']
    values=np.asarray(anchor['event_inequalities']);jac=[];even=[]
    for pair in measured['pairs']:
        p,m=[np.asarray(index[pair[k]]['event_inequalities']) for k in ('plus','minus')]
        jac.append(p-m);even.append(p+m-2*values)
    jac=np.asarray(jac).T;even=np.asarray(even).T
    gradient=np.asarray(measured['gradient']);models=[]
    for radius in (.25,1.,4.,16.):
        r=solve_correction(jac,gradient,values,history[1]['score']-anchor['score'],radius=radius)
        models.append(dict(radius=radius,**r))
        print('CRUISE_EVENT_MODEL',radius,r['success'],r.get('predicted_loss_change'),flush=True)
    names=names_for(baseline);row=names.index('14:motion.cruise_ge99_s')
    old_jac=np.asarray(measured['jacobian'])
    output=dict(anchor=anchor['tag'],jacobian=jac.tolist(),loss_gradient=gradient.tolist(),
        values=values.tolist(),models=models,even_response=even.tolist(),
        verified_record_comparisons=checked,actual_acceptance_unchanged=True,
        path14_count_gradient_norm=float(np.linalg.norm(old_jac[row])),
        path14_speed_margin_gradient_norm=float(np.linalg.norm(jac[row])),
        path14_margin_mm_s=float(values[row]),new_physical_runs=0,
        source_sha256=sha(stage/'state_joint/results.json'),
        equivalences=['At least N samples at speed>=99 iff the Nth largest speed>=99.',
                      'At least N consecutive samples at speed>=99 iff max over N-sample windows of min speed>=99.'])
    write_json(stage/'cruise_event_credit_audit.json',output)
    print('CRUISE_EVENT_EQUIVALENCE',checked,'COUNT_GRADIENT',output['path14_count_gradient_norm'],
          'MARGIN_GRADIENT',output['path14_speed_margin_gradient_norm'],'MARGIN',values[row],flush=True)


if __name__=='__main__':main()
