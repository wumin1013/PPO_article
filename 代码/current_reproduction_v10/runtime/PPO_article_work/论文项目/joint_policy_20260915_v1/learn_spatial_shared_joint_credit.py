"""Direct full-feedback learning of shared geometry/motion-conditioned expert weights."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import numpy as np
import torch
from joint_anticipation_features import anticipation_features
from learn_shared_temporal_joint_policy import initialize,optimize
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_lateral_aware_expert_weight_learning_20260920'
    neighborhood=BASE/'canonical_gate_band_joint_state_representation_neighborhood_20260920'
    parent_file=source/'width_1.2/candidate.pt'
    expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
    assert sha(parent_file)=='59e1bf592d2f2a5f519a46825f0d673a3bdd6978b492962f8289de00757c5771'
    assert sha(expert_file)=='86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
    result=read(source/'results.json')
    records=copy.deepcopy(next(c for c in result['candidates'] if c['tag']=='width_1.2')['records'])
    for r in records:r['trace']=str(source/r['trace'])
    retained=copy.deepcopy(read(source/'baseline.json')['retained'])
    added=read(neighborhood/'results.json')['records']
    for r in sorted((r for r in added if r['arm']=='expert_weights_width_1.2'),key=lambda r:r['path']):
        r=copy.deepcopy(r);r['trace']=str(neighborhood/r['trace']);records.append(r)
        old=next(o for o in added if o['path']==r['path'] and o['arm']=='retained')
        retained.append(dict(path=r['path'],baseline_shape=old['shape'],complete_time_s=old['complete_time_s']))
    baseline=dict(records=records,summary=aggregate(records))
    phases=[]
    for r in records:
        with np.load(r['trace']) as z:obs=z['observations']
        for x in obs:
            bend=anticipation_features(x,np.zeros(3))[:2].mean()
            speed=100.*np.linalg.norm(x[2:4])
            if bend>1e-3 and speed>5.:
                phases.append(30.*np.linalg.norm(x[11:13])/(speed+5.))
    centers=np.quantile(phases,[.15,.5,.85])
    feature_spec=dict(centers=centers.tolist(),width=float(max(np.median(np.diff(centers)),.02)),
        definition='Three data-quantile radial features of local observed gate-range/(speed+5), '
        'plus continuous future-bend strength and squared target-gap/motion-lag features. '
        'All six are reflection-even; no path identity,absolute time,angle table or teacher schedule.')
    paths=[14,104,204,206,304]
    stage=register('canonical_gate_band_spatial_shared_joint_credit_learning_20260920',
        [__file__,BASE/'learn_shared_temporal_joint_policy.py',BASE/'learned_action_pulse_policy.py',
         BASE/'joint_anticipation_features.py',BASE/'complete_corner_metrics.py',parent_file,expert_file,
         source/'receipt.json',source/'results.json',source/'raw_result_audit_width_1.2.json',
         neighborhood/'receipt.json',neighborhood/'results.json'],
        dict(wall_s=600,training_pool_paths=88,optimization_paths=paths,shared_parameters=18,
             maximum_new_complete_candidate_vectors=80,feature_spec=feature_spec,
             source_zero_rollouts_repeated=0,decision_ms=20,execution_ms=1),
        'Four precisely fitted state functions pass80 training paths but still leave '
        'a0.216mm outgoing excursion on the new304 training geometry. Broader feedback '
        'expert weights transfer partially, without eliminating the shape gap. Move '
        'from teacher-label interpolation to DIRECT actual-complete-return optimization '
        'of one shared18-parameter state function. Continuous local gate range relative '
        'to measured speed separates spatial approach phases, while target-gap/motion-lag '
        'features supply exit feedback. Centers are fixed data quantiles of ALL88 saved '
        'training states; coefficients for steering,feed,preview are learned jointly. '
        'Unlike prior reactive/anticipatory18-parameter trials, the new features explicitly '
        'resolve local geometry relative to motion and the parent is the audited59e model. '
        'Optimize original14 together with104,204,206,304; keep the original14 per-metric '
        'bounds, current lateral displacement/energy/exit on all cases, declared '
        'current/retained curvature and apex-speed envelopes on the four unresolved cases, '
        'and all physical/completion/time/cruise guards. No shape envelope is deployed. '
        'Reuse complete zero baselines and verify only startup20ms identity. If a feasible '
        'shared update exists, require the entire88 training pool before saving a research '
        'checkpoint. Reject if shared credit cannot protect strong paths; do not repeat '
        'this unchanged optimization or claim PPO/generalization/final acceptance.')
    write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
    write_json(stage/'feature_audit.json',dict(eligible_saved_states=len(phases),feature_spec=feature_spec,
        no_labels_used_to_choose_centers=True,no_new_physics_for_features=True))
    started=time.perf_counter()
    try:
        parent=torch.load(parent_file,map_location='cpu',weights_only=False)
        expert=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        training=[r for r in records if r['path'] in paths]
        old=[next(o['baseline_shape'] for o in retained if o['path']==r['path']) for r in training]
        initialize(stage,parent,training,old,started+585,True,None,expert,True,feature_spec,[104,204,206,304])
        outcome=optimize('spatial_expert')
        write_json(stage/'results.json',dict(arms=[outcome],training_only=True,adopted=False))
        receipt(stage,dict(status='completed_spatial_shared_joint_credit_learning',
            new_candidates=outcome['new_candidates'],improved_arms=['spatial_expert'] if outcome['improved'] else [],
            source_model_sha256=sha(parent_file),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_spatial_shared_joint_credit_learning',error=str(exc),
            wall_s=time.perf_counter()-started,goal_complete=False))
        raise


if __name__=='__main__':main()
