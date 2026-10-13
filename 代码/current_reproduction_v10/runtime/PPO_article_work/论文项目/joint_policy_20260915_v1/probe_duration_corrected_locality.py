"""Complete the preselected signed small-step check, reusing existing paths."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    reference=BASE/'canonical_gate_band_retained_center_reference_20260918'
    locality=BASE/'canonical_gate_band_retained_center_locality_20260918'
    data=read(reference/'results.json');model=BASE/data['parent_model']
    stage=register('canonical_gate_band_duration_small_reference_20260918',
        [__file__,model,reference/'results.json',reference/'basis.npz',reference/'receipt.json',
         locality/'results.json',locality/'receipt.json',
         BASE/'canonical_gate_band_duration_corrected_credit_20260918/receipt.json',
         BASE/'canonical_gate_band_duration_corrected_events_20260918/receipt.json',
         BASE/'train_pilot.py',BASE/'learn_single_decision_credit.py'],
        dict(new_full_paths=54,reused_full_paths=4,actor_decisions=6500,physics_ticks=150000,wall_s=120,
             source_direction=1,scales=[-.125,.125],training_updates=0),
        'The current4067 ordinary response has finite nonlinear error, while terminal '
        'duration contamination is now repaired. Complete the SAME preselected '
        'shared-parameter direction and signed0.125 amplitudes across all29 screening '
        'paths, reusing the four completed path31/41 traces and exact diagnostic models. '
        'Only amplitude differs from the original full direction, all state/actor/physical '
        'rules remain fixed. This evaluates a smaller local response domain, not new '
        'learning or a guarantee over a parameter ball. Require complete paths, the '
        'unchanged half-margin signed-capsule prediction reserve, and explicit initiation '
        'and duration event checks before any local learning proposal. Preserve broad '
        'coverage failures. Any new learned direction must separately pass both-sign '
        'checks and full64 shape/cruise/time acceptance; no automatic step adoption.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    md=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);factory=checkpoint_environment_factory(md['architecture'])
    ids=[r['original_path'] for r in data['baseline']['results']]
    missing=[p for p in ids if p not in (31,41)]
    all_paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    with np.load(reference/'basis.npz') as z:arrays={k:z[k].copy() for k in z.files}
    direction=arrays['basis'][1]*arrays['amplitudes'][1]
    arrays['basis']=arrays['basis'][1:2];arrays['amplitudes']=arrays['amplitudes'][1:2]*.125
    np.savez_compressed(stage/'basis.npz',**arrays)
    candidates=[];old_records=read(locality/'results.json')['records']
    for sign in (-1,1):
        scale=sign*.125;diagnostic_model=locality/f'scale{scale:g}.pt'
        saved=torch.load(diagnostic_model,map_location='cpu',weights_only=False)
        expected=md['policy']['residual_coefficients']+torch.tensor(scale*direction.reshape(3,256),dtype=torch.float32)
        assert torch.equal(saved['policy']['residual_coefficients'],expected)
        policy.load_state_dict(saved['policy']);tag=f'scale{scale:g}'
        results=evaluate(policy,[np.asarray(all_paths[p]) for p in missing],stage,tag,None,costs,start+120,factory)
        assert costs['evaluation_actor_calls']<=6500 and costs['evaluation_physics_ticks']<=150000
        mapped={p:(dict(r,original_path=p),stage/tag/f'{j}.npz') for j,(p,r) in enumerate(zip(missing,results))}
        for path in (31,41):
            old=next(r for r in old_records if r['path']==path and r['scale']==scale)
            mapped[path]=(dict(old['result'],original_path=path),locality/old['trace'])
        ordered=[mapped[p][0] for p in ids];traces=[mapped[p][1] for p in ids]
        summary=review(ordered,traces)
        item=dict(direction=0,sign=sign,amplitude=float(arrays['amplitudes'][0]),
            model=str(diagnostic_model.relative_to(BASE)),model_sha256=sha(diagnostic_model),
            summary=summary,results=ordered,traces=[str(p.relative_to(BASE)) for p in traces],
            metrics=metric_rows(ordered,traces,summary,data['metric_names']))
        candidates.append(item)
        write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],parent_model=data['parent_model'],
            baseline=data['baseline'],candidates=candidates,metric_names=data['metric_names'],costs=costs,
            original_full_reference=reference.name,source_direction=1,source_scale=.125,
            diagnostic_only=True,goal_complete=False))
        print('SMALL_COMPLETE_REFERENCE',sign,'success',summary['successes'],flush=True)
    receipt(stage,dict(status='completed_small_complete_response_reference',costs=costs,reused_paths=4,
        new_paths=54,wall_s=time.perf_counter()-start,diagnostic_only=True,goal_complete=False))


if __name__=='__main__':main()
