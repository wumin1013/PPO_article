"""Finite reward/termination checks on four newly specified synthetic paths."""
import math
import time
import numpy as np
from study_io import BASE,register,receipt,write_json
from joint_runtime_v5 import environment_factory
from validate_actions import difference_audit


if __name__=='__main__':
    cases=[(63.,23.,29.,35.,3.,True),(147.,37.,26.,30.,4.6,True),
        (93.,23.,27.,50.,12.,False),(147.,37.,26.,85.,3.,False)]
    sources=[__file__,BASE/'joint_runtime_v5.py',BASE/'terminal_candidate_runtime_v1.py',
        BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',
        BASE/'geometry_spec.py',BASE/'validate_actions.py',
        BASE/'terminal_candidate_diagnosis_v1/receipt.json',BASE/'failure_credit_analysis_v1/receipt.json']
    stage=register('reward_runtime_validation_v5',sources,
        dict(cases=cases,runs=4,max_ticks_per_run=6000,models=0,training_updates=0,wall_s=180),
        'Validate implemented remaining-progress failure credit with unchanged successful-time return and common full-command audit; four new synthetic diagnostic paths only.')
    start=time.perf_counter()
    reports=[]
    for index,(angle,first,last,feed,distance,expected_success) in enumerate(cases):
        theta=math.radians(angle)
        ref=np.array([[0.,0.],[first,0.],[first+last*math.cos(theta),last*math.sin(theta)]])
        env=environment_factory(ref,fixed_d=distance)
        action=np.array([0.,np.arctanh(feed/50.-1.),0.])
        chunks=[]
        total=0.
        penalty=0.
        while not env.done:
            if time.perf_counter()-start>180 or env.ticks>6000:
                raise TimeoutError('Synthetic validation budget exceeded')
            _,reward,_,info=env.step(action)
            total+=reward
            penalty+=info['failure_remaining_penalty_s']
            chunks.append(info['rows'])
        rows=np.vstack(chunks)
        audit=difference_audit(ref[0],rows)
        expected=-env.ticks*.001+env.corridor.arcs[-1]/100.
        if not env.success:
            expected-=env.max_ticks*.001+1.+max(0.,env.corridor.arcs[-1]-env.progress()[0])/20.
        assert abs(total-expected)<1e-8
        assert env.success==expected_success
        assert audit['kinematic_pass']
        if env.success:
            assert audit['rest_pass'] and penalty==0.
        else:
            assert penalty>0.
        np.savez_compressed(stage/f'{index}.npz',reference=ref,rows=rows)
        reports.append(dict(case=index,success=env.success,reason=env.reason,physics_ticks=env.ticks,
            complete_time_s=env.ticks*.001 if env.success else None,return_sum=total,
            analytic_return=expected,remaining_penalty=penalty,audit=audit))
    write_json(stage/'results.json',dict(cases=reports,models=0,training_updates=0,
        no_inference_about_learning_improvement=True,active_v4_unchanged=True))
    receipt(stage,dict(status='four_new_synthetic_reward_and_completion_checks_passed',
        runs=4,success_cases=2,failure_cases=2,physics_ticks=sum(r['physics_ticks'] for r in reports),
        models=0,training_updates=0,wall_s=time.perf_counter()-start))
    print(reports)
