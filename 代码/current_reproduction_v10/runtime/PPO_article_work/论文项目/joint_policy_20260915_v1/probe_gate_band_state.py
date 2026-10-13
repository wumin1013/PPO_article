"""Matched generic state-support repair, reusing both old-runtime baselines."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,canonical_observation,environment_factory as old_factory
from gate_band_state import environment_factory
from probe_coordinated_suffix import reconstruct
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from learn_canonical_credit_step import compare
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_waveform_learned_update_20260917'
    source=BASE/'canonical_recentered_waveform_corrected_composition_20260917'
    selected=read(parent/'training_selection.json');old=read(source/'results.json')
    parent_screen=next(r for r in read(parent/'results.json')['candidates'] if r['model']==selected['model'])
    positive=old['candidates'][1];names=old['metric_names']
    ids=[r['original_path'] for r in positive['results']]
    models=[parent/selected['model'],source/'step+0.125.pt']
    stage=register('canonical_gate_band_matched_probe_20260917',
        [__file__,*models,parent/'results.json',source/'results.json',source/'receipt.json',
         BASE/'canonical_gate_lateral_support_causal_probe_20260917/results.json',
         BASE/'canonical_gate_lateral_support_causal_probe_20260917/receipt.json',
         BASE/'gate_band_state.py',BASE/'canonical_gate_state.py',
         BASE/'continuous_gate_guidance.py',BASE/'learn_canonical_credit_step.py',BASE/'train_pilot.py'],
        dict(parameter_training_updates=0,paths=ids,new_complete_paths=58,
             evaluation_actor_calls=6000,evaluation_physics_ticks=140000,
             saved_prefix_geometry_ticks=500,wall_s=120),
        'The bidirectional one-decision experiment confirms lateral support '
        'membership creates a harmful observation discontinuity. Remove the '
        'lateral gate_contains Boolean ONLY from state-transition eligibility. '
        'Use the same generic2mm longitudinal band and ordered adjacent gates. '
        'The guide retains its original0.1mm support, all physical execution '
        'and coverage checks remain unchanged, and overlapping bands still '
        'raise an explicit error. Check spatial continuity around the observed '
        'lateral edge at three perturbation sizes, then evaluate the parent '
        'weights and the already learned positive small weights on29 complete '
        'training routes each. The old-runtime pair is reused: this is a2x2 '
        'representation-by-parameter diagnostic, not fresh learning. Report '
        'full constraints, shape, time and cruise. Do not promote by aggregate '
        'success alone or reuse old-runtime derivatives as current credit. '
        'No development/sealed access; no ideal-trajectory template.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0,saved_prefix_geometry_ticks=0)
    p=CanonicalGatePolicy(256);metadata=torch.load(models[0],map_location='cpu',weights_only=False)
    p.load_state_dict(metadata['policy'])
    j=next(i for i,r in enumerate(parent_screen['results']) if r['original_path']==11)
    with np.load(BASE/parent_screen['traces'][j]) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
    env=reconstruct(ref,rows,actions,500,costs,env_factory=old_factory)
    gate=env.corridor.gates[0];h=float((env.position-gate.point)@gate.normal)
    q=env.position-h*gate.normal;ray=q-gate.point
    # At this incoming-interior gate ray the segment distance scales linearly.
    radial_distance=abs(ray[1]);assert ref[0,1]==ref[1,1]==0.
    boundary=gate.point+ray*(1.25/radial_distance)+h*gate.normal
    radial=ray/np.linalg.norm(ray);checks=[]
    for delta in (1e-3,1e-5,1e-7):
        values={}
        for repair in (False,True):
            obs=[];held=[]
            for sign in (-1,1):
                env.position=boundary+sign*delta*radial
                x=canonical_observation(env,gate_channel_support=not repair)
                obs.append(x);held.append(env.decode_action(p.act(x,True)[0]))
            values[str(repair)]=dict(observation_max_jump=float(abs(obs[1]-obs[0]).max()),
                action_jump=(held[1]-held[0]).tolist())
        checks.append(dict(position_separation_mm=2*delta,values=values))
    assert checks[-1]['values']['True']['observation_max_jump']<1e-5
    assert checks[-1]['values']['False']['observation_max_jump']>.1
    write_json(stage/'continuity_check.json',dict(checks=checks,coverage_changed=False,guide_changed=False))
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];candidates=[]
    for index,(arm,model) in enumerate(zip(('parent','positive'),models)):
        meta=torch.load(model,map_location='cpu',weights_only=False);p.load_state_dict(meta['policy'])
        saved=dict(meta,architecture='gate_band_canonical_kernel',
            state_support_encoding='ordered_adjacent_longitudinal_band_2mm',
            converted_from_sha256=sha(model),learning_method='untrained_state_support_repair')
        save_torch(stage/f'{arm}.pt',saved)
        rs=evaluate(p,[np.asarray(paths[i]) for i in ids],stage,arm,None,costs,start+120,environment_factory)
        rs=[dict(r,original_path=i) for r,i in zip(rs,ids)];ts=[stage/arm/f'{j}.npz' for j in range(len(ids))]
        summary=review(rs,ts);metrics=metric_rows(rs,ts,summary,names)
        candidate=dict(arm=arm,model=f'{arm}.pt',summary=summary,metrics=metrics,
            results=rs,traces=[str(t.relative_to(BASE)) for t in ts]);candidates.append(candidate)
        reference=dict(summary=old['baseline'],metrics=old['baseline_metrics']) if index==0 else positive
        candidate['same_weights_old_runtime_comparison']=compare(candidate,reference,names,required_reduction=0.)
        write_json(stage/'results.json',dict(candidates=candidates,metric_names=names,costs=costs,goal_complete=False))
        print('GATE_BAND',arm,{k:summary[k] for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s')},flush=True)
    parameter_compare=compare(candidates[1],candidates[0],names,required_reduction=0.)
    write_json(stage/'comparison.json',dict(parameter_comparison_in_repaired_runtime=parameter_compare,
        representation_comparisons=[c['same_weights_old_runtime_comparison'] for c in candidates],
        new_training=False,goal_complete=False))
    assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=140000
    receipt(stage,dict(status='completed_matched_gate_band_state_probe',costs=costs,
        complete_paths=sum(c['summary']['successes'] for c in candidates),
        wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
