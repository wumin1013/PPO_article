"""One shared-policy restoration from measured endpoint constraints, then full rollouts."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from cruise_update_geometry import CruiseUpdateGeometry
from learn_canonical_credit_step import compare
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from solve_continuous_sensor_credit import IMPROVEMENT_NAMES
from train_pilot import evaluate, save_torch
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def feasible_interval(left, right, baseline, names):
    """Intersect scalar secant constraints; this is a proposal, not a certificate."""
    matrices=[np.array([r['values'] for r in c['metrics']]) for c in (left,right,baseline)]
    for c in (left,right):
        assert [r['path'] for r in c['metrics']]==[r['path'] for r in baseline['metrics']]
    a,b,original=matrices
    bounds=[]
    for i,row in enumerate(baseline['metrics']):
        for j in range(names.index('segment_max_distance_mm')):
            bounds.append((a[i,j]-original[i,j],b[i,j]-a[i,j],f"path{row['path']}_{names[j]}"))
    for name in IMPROVEMENT_NAMES+('full_xy_normalized_energy','full_physical_normalized_energy'):
        j=names.index(name);fraction=.99 if 'energy' in name else .9
        bounds.append((a[:,j].sum()-fraction*original[:,j].sum(),
                       (b[:,j]-a[:,j]).sum(),'aggregate_'+name))
    lo,hi=0.,1.;active=[]
    for offset,slope,label in bounds:
        if abs(slope)<1e-14:
            assert offset<=1e-12,('No secant-feasible interval',label,offset)
            continue
        boundary=-offset/slope
        if slope>0 and boundary<hi:hi=boundary;active.append(dict(label=label,kind='upper',value=boundary))
        if slope<0 and boundary>lo:lo=boundary;active.append(dict(label=label,kind='lower',value=boundary))
    assert 0<=lo<hi<=1.,(lo,hi)
    return dict(lower=lo,upper=hi,selected=.5*(lo+hi),active=active)


def main():
    torch.set_num_threads(1)
    anchor=BASE/'canonical_gate_band_correct_physical_cruise_full_learning_20260917'
    joint=BASE/'canonical_gate_band_joint_regret_learning_20260917'
    shape=BASE/'canonical_gate_band_smooth_shape_learning_20260917'
    retained=BASE/'canonical_gate_band_shape_cruise_statewise_half_full_20260917'
    sources=[__file__,BASE/'learn_canonical_credit_step.py',BASE/'train_pilot.py',
             BASE/'canonical_gate_band_joint_regret_round_20260917/exit_response_error_layers.json',
             BASE/'learning_preparation_v4/paths.json',
             BASE/'canonical_gate_band_shape_cruise_current_geometries_20260917/geometry.npz']
    for folder in (anchor,joint,shape,retained):sources += [folder/'results.json',folder/'receipt.json']
    sources += [anchor/'candidate_1.pt',joint/'candidate_1.pt',shape/'candidate_1.pt',retained/'candidate_0.5.pt']
    stage=register('canonical_gate_band_measured_segment_restoration_20260917',sources,
        dict(new_parameter_candidates=1,new_training_paths=64,new_local_models=0,
             evaluation_actor_calls=6000,evaluation_physics_ticks=160000,wall_s=240),
        'The matched joint-regret direction improved shape and energy but failed path2/6 exit nonregression; '
        'saved waveform layers attribute the missed response to finite closed-loop state error, not omission of moving extrema. '
        'Do not enlarge empirical error coefficients. Test measured endpoint secants as a single bounded nonlinear restoration '
        'of the shared network parameters, using joint and shape full-step endpoints at the SAME730d parent. '
        'Keep the joint objective study rejected. Intersect measured per-corner nonincrease, all positive shape totals down10%, '
        'and both energy totals down1% on the parameter segment. Choose its midpoint to leave coefficient room from either '
        'predicted feasibility boundary; endpoints and this scalar rule are fixed before new evaluation. '
        'The secant is only a proposal. Evaluate all29 screen routes with the unchanged exact deployed actor and physical runtime. '
        'Stop on any original shape/curvature regression above1e-6, completion failure or missed aggregate reduction. '
        'If eligible, evaluate the remaining35 once and require64/64, unchanged physical constraints, full time and99mm/s '
        'cruise at least as good as retained4067. No additional scalar search, new credit or error inflation in this experiment. '
        'Parameters, not online actions or trajectories, are interpolated; runtime contains one shared learned policy. '
        'This is a training-set restoration experiment, not independent generalization, a causal proof of historical PPO, '
        'or final target-shape success. Preserve every stronger model, failure and S2; no sealed access.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    try:
        jd=read(joint/'results.json');sd=read(shape/'results.json');ad=read(anchor/'results.json')
        left=next(r for r in jd['candidates'] if r['step']==1.)
        right=next(r for r in sd['candidates'] if r['step']==1.)
        baseline=jd['baseline'];names=jd['metric_names'];assert names==sd['metric_names']
        interval=feasible_interval(left,right,baseline,names)
        write_json(stage/'coefficient_selection.json',dict(**interval,selection_before_new_rollouts=True))
        md=[torch.load(f/'candidate_1.pt',map_location='cpu',weights_only=False) for f in (anchor,joint,shape)]
        assert sha(anchor/'candidate_1.pt')=='730d835277a7cdcae7941e5ab799b2e664f728cb3ff5b98bc078873f75921f30'
        assert all(m['parent_sha256']==sha(anchor/'candidate_1.pt') for m in md[1:])
        for key in md[0]['policy']:
            if key!='residual_coefficients':
                assert all(torch.equal(m['policy'][key],md[0]['policy'][key]) for m in md[1:]),key
        alpha=interval['selected'];policy=CanonicalGatePolicy(256);policy.load_state_dict(md[0]['policy'])
        with torch.no_grad():
            policy.residual_coefficients.copy_(((1-alpha)*md[1]['policy']['residual_coefficients'].double()+
                                              alpha*md[2]['policy']['residual_coefficients'].double()).float())
        delta=(policy.residual_coefficients.detach().double()-md[0]['policy']['residual_coefficients'].double()).numpy().ravel()
        geometry=CruiseUpdateGeometry(BASE/'canonical_gate_band_shape_cruise_current_geometries_20260917','physical',sha(anchor/'candidate_1.pt'))
        norm=float(np.sqrt(delta@geometry.metric@delta));assert norm<=.1
        assert geometry.support_constraint(delta)[0]<=1e-7
        assert (geometry.features@delta[256:512]).min()>=-1e-7
        model='candidate.pt'
        save_torch(stage/model,dict(md[0],policy=policy.state_dict(),parent_sha256=sha(anchor/'candidate_1.pt'),
            learning_method='measured_shared_parameter_secant_restoration',coefficient=alpha,
            endpoint_model_sha256=[sha(joint/'candidate_1.pt'),sha(shape/'candidate_1.pt')]))
        paths=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['training']]
        def run(ids,tag):
            rs=evaluate(policy,[paths[i] for i in ids],stage,tag,None,costs,start+240,
                        checkpoint_environment_factory(md[0]['architecture']))
            rs=[dict(r,original_path=i) for r,i in zip(rs,ids)];traces=[stage/tag/f'{j}.npz' for j in range(len(ids))]
            summary=review(rs,traces)
            assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=160000
            return dict(summary=summary,metrics=metric_rows(rs,traces,summary,names),results=rs,
                        traces=[str(t.relative_to(BASE)) for t in traces])
        ids=[r['original_path'] for r in left['results']];candidate=run(ids,'screen')
        candidate.update(model=model,step=1.,coefficient=alpha)
        candidate['checks']=compare(candidate,baseline,names,required_energy_reduction=.01,energy_scope='aggregate',check_screen_time=False)
        print('MEASURED_SEGMENT_SCREEN',alpha,candidate['checks'],flush=True)
        out=dict(baseline=baseline,candidates=[candidate],chosen=None,full_candidate=None,metric_names=names,
                 coefficient_selection=interval,parameter_norm=norm,costs=costs,goal_complete=False)
        write_json(stage/'results.json',out)
        if candidate['checks']['eligible']:
            missing=[i for i in range(64) if i not in ids];remainder=run(missing,'remaining')
            pairs=sorted(zip(candidate['results']+remainder['results'],candidate['traces']+remainder['traces']),key=lambda r:r[0]['original_path'])
            results,traces=map(list,zip(*pairs));summary=review(results,[BASE/t for t in traces])
            full=dict(summary=summary,metrics=sorted(candidate['metrics']+remainder['metrics'],key=lambda r:r['path']),results=results,traces=traces)
            full['checks']=compare(full,ad['full_candidate'],names,required_energy_reduction=.01,energy_scope='aggregate',preserve_actual_cruise=True)
            rd=read(retained/'results.json')['full_candidate']['summary']
            assert sha(retained/'candidate_0.5.pt')=='406729c098e21566ded12bad87100ade1226e5874e4647a8a54ba0287a19f856'
            efficient=bool(summary['successes']==64 and summary['full_time_s']<=rd['full_time_s']+1e-12 and summary['complete_cruise_s']['99']>=rd['complete_cruise_s']['99']-1e-12)
            full['checks']['retained_efficiency']=dict(passes=efficient,reference_time_s=rd['full_time_s'],candidate_time_s=summary['full_time_s'],reference_cruise99_s=rd['complete_cruise_s']['99'],candidate_cruise99_s=summary['complete_cruise_s']['99'])
            full['checks']['eligible']=bool(full['checks']['eligible'] and efficient)
            out.update(chosen=candidate,full_candidate=full,full_parent=ad['full_candidate'])
            write_json(stage/'results.json',out)
            print('MEASURED_SEGMENT_FULL',full['checks']['eligible'],full['checks']['retained_efficiency'],flush=True)
        eligible=bool(out['full_candidate'] and out['full_candidate']['checks']['eligible'])
        receipt(stage,dict(status='completed_measured_segment_restoration',full_training_eligible=eligible,
            costs=costs,wall_s=time.perf_counter()-start,working_model_pointer_changed=False,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_measured_segment_restoration',error=str(exc),costs=costs,
                          wall_s=time.perf_counter()-start,goal_complete=False))
        raise


if __name__=='__main__':main()
