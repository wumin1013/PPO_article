"""Attribute a saved local model's restriction; never changes real acceptance."""
import numpy as np
from scipy.optimize import linprog
from audit_preemptive_joint_credit import constraint_names
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_midturn_supported_joint_learning_20260920'
    model=read(stage/'local_response.json');jac=np.asarray(model['jacobian']);g=np.asarray(model['outgoing_gradient'])
    zero=read(stage/'state_joint/results.json')['candidates'][0];values=np.asarray(zero['augmented_learning_inequalities'])
    names=constraint_names([104]);groups={'all':[], 'without_apex_sample':['104:apex_speed'],
        'without_three_speed_samples':['104:apex_speed','104:motion.before_apex_5mm_speed_mm_s','104:motion.after_apex_5mm_speed_mm_s'],
        'without_intrinsic_curve':['104:intrinsic_peak','104:intrinsic_energy']}
    records=[]
    for label,removed in groups.items():
        keep=np.array([name not in removed for name in names]);a=-jac[keep];b=values[keep]
        scales=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
        solved=linprog(g,A_ub=a/scales[:,None],b_ub=b/scales,bounds=[(-16.,16.)]*len(g),method='highs',
            options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
        row=dict(arm=label,success=bool(solved.success),removed_for_offline_attribution_only=removed)
        if solved.success:
            predicted=values+jac@solved.x
            row.update(predicted_exit_reduction_mm=float(-solved.fun),scaled_parameter_step=solved.x.tolist(),
                active_original_guards=[name for name,dual in zip(np.array(names)[keep],solved.ineqlin.marginals) if abs(dual)>1e-10],
                predicted_violations_of_original=[dict(guard=names[i],value=float(predicted[i])) for i in np.flatnonzero(predicted < -1e-8)])
        records.append(row)
        print('CAPACITY_ATTRIBUTION',{k:v for k,v in row.items() if k!='scaled_parameter_step'},flush=True)
    write_json(stage/'constraint_capacity_attribution.json',dict(records=records,
        source_response_sha256=sha(stage/'local_response.json'),new_physical_runs=0,
        original_acceptance_unchanged=True,interpretation='These are local surrogate counterfactuals. '
        'They distinguish constraints that restrict the measured direction space. '
        'No guard is removed from physical rollout validation or model selection.'))


if __name__=='__main__':main()
