"""Zero-physics metric identity, bounded-return and conditional-density checks."""
import numpy as np
import torch
from bounded_shape_return import bounded_shape_cost
from conditional_feedback_policy import ConditionalFeedbackPolicy
from geometric_turn_metrics import geometric_turn_metrics
from dynamic_residual_policy import PreviewResidualPolicy
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    sources=[BASE/f for f in ('bounded_shape_return.py','conditional_feedback_policy.py','geometric_turn_metrics.py')]
    model_path=BASE/'interior_teacher_feedback_learning_20260916/epoch_50.pt'
    stage=register('bounded_shape_return_audit_20260916',[__file__,model_path]+sources,
        dict(physics_ticks=0,weight_updates=0,model_queries=32,wall_s=120),
        'Verify corrected spatial metric identity, bounded dense-credit sums and exact AR conditional log probabilities before full-feedback PPO. Algebraically preserve success-over-failure ordering using the existing terminal penalty.')
    old=read(BASE/'spatial_metric_resolution_audit_20260916/results.json')['saved_geometry'];records=[]
    folders={'PPO_source':'ar_conditional_joint_20260916/final_validation','correlated_g4':'correlated_parameter_search_20260916/validation_4','balanced_g5':'balanced_shape_time_learning_20260916/validation_5'}
    for r in old:
        with np.load(BASE/folders[r['method']]/f"{r['path']}.npz") as z:
            measured=geometric_turn_metrics(z['reference'],z['rows']);cost,details=bounded_shape_cost(z['reference'],z['rows'],z['actions'])
        assert abs(measured['bending_energy_mm_inv']-r['grid_01']['bending_energy_mm_inv'])<1e-12
        assert abs(details['normalized_energy']-1.25*measured['normalized_bending_mm_inv'])<1e-12
        assert 0<=cost.sum()<=1
        records.append(dict(method=r['method'],path=r['path'],cost=details))
    torch.set_num_threads(1);torch.manual_seed(916701)
    saved=torch.load(model_path,map_location='cpu',weights_only=False)
    policy=ConditionalFeedbackPolicy(72);policy.load_state_dict(saved['policy'])
    plain=PreviewResidualPolicy(72);plain.load_state_dict(saved['policy'])
    with np.load(BASE/'interior_teacher_feedback_learning_20260916/training_data.npz') as z:obs=z['observation'][:32]
    with torch.no_grad():assert torch.equal(policy.distribution(torch.tensor(obs)).mean,plain.distribution(torch.tensor(obs)).mean)
    maximum=0.
    for o in obs:
        latent,old_log,_=policy.act(o);c=policy.last_context
        with torch.no_grad():new_log,_,_=policy.evaluate_conditioned(torch.tensor(o),torch.tensor(latent),torch.tensor(c['previous_observation']),torch.tensor(c['previous_latent']),torch.tensor(c['context_live']))
        maximum=max(maximum,abs(old_log-float(new_log)))
    assert maximum<1e-6
    proof='For identical path lengthL, Rs=-Ts+L/100-Cs and Rf=-Tf-Tmax-1+L/100-remaining/20-Cf. With Ts<=Tmax,0<=Cs<=1,Cf>=0,Tf>0: Rs-Rf=Tmax-Ts+Tf+1-Cs+remaining/20+Cf>0. No early-failure reward advantage is introduced.'
    write_json(stage/'results.json',dict(records=records,conditional_log_probability_max_error=maximum,deterministic_actor_identity=True,
        success_failure_ordering_proof=proof,physics_ticks=0,training_updates=0,goal_complete=False))
    receipt(stage,dict(status='passed',physics_ticks=0,weight_updates=0,conditional_max_error=maximum,results_sha256=sha(stage/'results.json')))
    print('metric identity9/9; bounded costs; conditional log probability error',maximum,flush=True)


if __name__=='__main__':main()
