"""Current three-head parameter factorial, with measured full variants reused."""
import itertools
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from prepare_current_waveform_credit import wave_signals
from complete_corner_metrics import complete_corner_metrics
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_quadratic_error_learned_update_20260917'
    model=parent/'candidate_0.5.pt';selection=read(parent/'training_selection.json')
    small=BASE/'canonical_gate_band_half_step_locality_probe_20260917'
    known=read(small/'results.json')
    direction_source=BASE/'canonical_gate_band_half_step_locality_direction_20260917'
    wave=BASE/'canonical_gate_band_half_step_waveform_credit_20260917'
    with np.load(direction_source/'direction.npz') as z:direction=z['direction'].copy()
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    factory=checkpoint_environment_factory(metadata['architecture'])
    stage=register('canonical_gate_band_half_step_head_interaction_probe_20260917',
        [__file__,model,parent/'training_selection.json',parent/'receipt.json',
         small/'results.json',small/'receipt.json',direction_source/'direction.npz',
         wave/'results.json',wave/'receipt.json',BASE/'canonical_gate_state.py',
         BASE/'gate_band_state.py',BASE/'continuous_gate_guidance.py',BASE/'train_pilot.py'],
        dict(paths=[14,15],steps=[-.125,.125],new_complete_paths=24,reused_full_variants=4,
             reused_parent_paths=2,evaluation_actor_calls=3000,evaluation_physics_ticks=80000,wall_s=120),
        'Single-action amplitude reduction failed to improve path14 composition '
        'prediction. Isolate nonlinear interaction among the three shared learned '
        'parameter heads: all six proper nonempty subsets of steering/feed/preview '
        'at each measured sign, on paths14 and15. Reuse the unchanged parent and '
        'already measured all-three variants after exact checkpoint verification. '
        'Every active head receives the same known parameter displacement; other '
        'heads keep parent parameters but respond to the real state. The factorial '
        'decomposes single-head prediction errors, pair interactions and triple '
        'interaction at the waveform level, before max or reversal metrics. '
        'These are diagnostic parameter interventions, not selected learned models. '
        'All physical constraints, coverage and stopping remain intact. No labels, '
        'trajectory rules, development or sealed access; no training until analysis.')
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    policy=CanonicalGatePolicy(256);records=[];decomposition=[]
    for path in (14,15):
        j=next(j for j,r in enumerate(selection['results']) if r['original_path']==path)
        parent_trace=BASE/selection['traces'][j]
        with np.load(parent_trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        nominal,valid=wave_signals(ref,rows)
        with np.load(wave/f'path{path}_waveform.npz') as z:
            jac=z['gradient'].copy();jac[~z['common_valid']]=0.
        for step in (-.125,.125):
            full=next(r for r in known['candidates'] if r['step']==step)
            full_model=torch.load(small/f'step{step:+g}.pt',map_location='cpu',weights_only=False)
            policy.load_state_dict(metadata['policy'])
            with torch.no_grad():policy.residual_coefficients.add_(torch.tensor(step*direction,dtype=torch.float32))
            assert all(torch.equal(v,full_model['policy'][k]) for k,v in policy.state_dict().items())
            j=next(j for j,r in enumerate(full['results']) if r['original_path']==path)
            assert full['results'][j]['success']
            with np.load(BASE/full['traces'][j]) as z:full_wave,full_valid=wave_signals(ref,z['rows'],len(rows))
            signals={(0,0,0):nominal,(1,1,1):full_wave};common=valid&full_valid
            changes={}
            for bits in itertools.product((0,1),repeat=3):
                if sum(bits) in (0,3):continue
                policy.load_state_dict(metadata['policy'])
                with torch.no_grad():policy.residual_coefficients.add_(torch.tensor(step*direction*np.asarray(bits)[:,None],dtype=torch.float32))
                tag=f'p{path}_{step:+g}_'+''.join(map(str,bits))
                rs=evaluate(policy,[ref],stage,tag,None,costs,start+120,factory)
                assert costs['evaluation_actor_calls']<=3000 and costs['evaluation_physics_ticks']<=80000
                file=stage/tag/'0.npz'
                with np.load(file) as z:
                    value,mask=wave_signals(ref,z['rows'],len(rows))
                    shape=complete_corner_metrics(ref,z['rows'])
                record=dict(path=path,step=step,bits=list(bits),result=rs[0],shape=shape,
                    trace=str(file.relative_to(BASE)))
                records.append(record)
                write_json(stage/'results.json',dict(records=records,costs=costs,diagnostic_only=True,goal_complete=False))
                assert rs[0]['success'],'Incomplete paths cannot supply this matched waveform decomposition'
                signals[bits]=value;common&=mask
                changes[bits]=(policy.residual_coefficients.detach().double()-metadata['policy']['residual_coefficients'].double()).numpy().ravel()
            singles=sum((signals[b]-nominal for b in signals if sum(b)==1),np.zeros_like(nominal))
            pair_terms={}
            for bits in signals:
                if sum(bits)!=2:continue
                active=[tuple(int(j==i) for j in range(3)) for i in range(3) if bits[i]]
                pair_terms[bits]=signals[bits]-signals[active[0]]-signals[active[1]]+nominal
            pairs=sum(pair_terms.values(),np.zeros_like(nominal))
            triple=full_wave-nominal-singles-pairs
            full_delta=(full_model['policy']['residual_coefficients'].double()-metadata['policy']['residual_coefficients'].double()).numpy().ravel()
            prediction=np.einsum('tcd,d->tc',jac,full_delta)
            single_error=singles-prediction;total_error=full_wave-nominal-prediction
            assert np.max(abs(total_error-single_error-pairs-triple))<1e-10
            moving=common[:,2]
            def describe(value):
                ids=np.flatnonzero(moving);tick=int(ids[np.argmax(abs(value[ids,2]))])
                return dict(heading_max_abs_deg=float(abs(value[tick,2])),heading_peak_tick=tick,
                    heading_rms_deg=float(np.sqrt(np.mean(value[moving,2]**2))))
            row=dict(path=path,step=step,single_head_error=describe(single_error),
                pair_interaction=describe(pairs),triple_interaction=describe(triple),
                total_error=describe(total_error),pair_terms={''.join(map(str,k)):describe(v) for k,v in pair_terms.items()})
            decomposition.append(row)
            np.savez_compressed(stage/f'p{path}_{step:+g}_decomposition.npz',nominal=nominal,
                single_error=single_error,pairs=pairs,triple=triple,total_error=total_error,valid=common)
            print('HEAD_INTERACTION',row,flush=True)
            write_json(stage/'results.json',dict(records=records,decomposition=decomposition,costs=costs,
                parent_sha256=sha(model),diagnostic_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_current_head_interaction_factorial',complete_paths=len(records),
        costs=costs,wall_s=time.perf_counter()-start,diagnostic_only=True,goal_complete=False))


if __name__=='__main__':main()
