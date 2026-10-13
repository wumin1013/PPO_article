"""Current local-model error using an already measured new-runtime weight pair."""
import argparse
import time
import numpy as np
import torch
from waveform_constraint_model import waveform_facets
from solve_continuous_sensor_credit import vector
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','waveforms','stage'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();source=BASE/args.source;wave=BASE/args.waveforms
    measured=BASE/'canonical_gate_band_matched_probe_20260917'
    direction_source=BASE/'canonical_recentered_waveform_feasibility_recovery_20260917'
    data=read(measured/'results.json');parent,positive=data['candidates'];step=.125
    model=measured/parent['model'];metadata=torch.load(model,map_location='cpu',weights_only=False)
    target=torch.load(measured/positive['model'],map_location='cpu',weights_only=False)['policy']
    with np.load(direction_source/'direction.npz') as z:direction=z['direction'].ravel().copy()
    expected={k:v.clone() for k,v in metadata['policy'].items()}
    expected['residual_coefficients']+=torch.tensor(step*direction.reshape(3,256),dtype=torch.float32)
    assert expected.keys()==target.keys() and all(torch.equal(expected[k],target[k]) for k in expected)
    assert read(source/'optimizer.json')['parent_sha256']==sha(model)==read(wave/'results.json')['parent_sha256']
    stage=register(args.stage,[__file__,source/'gradients.npz',source/'support.npz',source/'baselines.json',
        source/'optimizer.json',source/'receipt.json',wave/'results.json',wave/'receipt.json',
        measured/'results.json',measured/'receipt.json',model,measured/positive['model'],
        direction_source/'direction.npz',BASE/'waveform_constraint_model.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,measured_signs=[1],step=step,wall_s=45),
        'Reuse the completed gate-band parent/positive weight contrast. Verify '
        'the positive checkpoint equals the parent plus the saved parameter '
        'direction exactly in float32. Evaluate that direction using FRESH '
        'gate-band waveform/scalar credit and its current support metric. Old '
        'direction coordinates identify the measured experiment only; old '
        'derivatives, predictions and margins are not reused. Estimate positive '
        'model underprediction per current norm for curvature/energy constraints. '
        'Only one sign has been measured in this runtime; disclose this limitation '
        'and require new BOTH-sign checks of any new proposal. No new physics, '
        'development or sealed access, no safety-guarantee claim.')
    start=time.perf_counter()
    with np.load(source/'gradients.npz') as z:gradient=z['gradient'].copy();ids=z['path_id'].tolist();names=z['metric_names'].tolist()
    with np.load(source/'support.npz') as z:metric=z['metric'].copy()
    norm=float(np.sqrt(direction@metric@direction))
    baselines=read(source/'baselines.json')['baselines'];base={b['path']:vector(b,b['curvature']) for b in baselines}
    actual={r['path']:np.asarray(r['values']) for r in positive['metrics']}
    errors=np.zeros((len(ids),len(names)));records=[]
    for i,b in enumerate(baselines):
        path=b['path'];prediction=step*gradient[i]@direction
        if b['shape'] is not None:
            with np.load(wave/f'path{path}_waveform.npz') as z:
                nominal=z['nominal'].copy();g=z['gradient'].copy();g[~z['common_valid']]=0.;valid=z['nominal_valid'].copy()
            values=waveform_facets(nominal,g,valid,b['shape']['turn_deg'],step*direction)
            for name,(value,_,_) in values.items():prediction[names.index(name)]=value-base[path][names.index(name)]
        change=actual[path]-base[path];error=change-prediction
        errors[i]=np.maximum(error,0.)/(step*norm)
        records.append(dict(path=path,step=step,actual_change=change.tolist(),predicted_change=prediction.tolist(),error=error.tolist()))
    write_json(stage/'results.json',dict(parent_sha256=sha(model),metric_names=names,path_ids=ids,
        source=args.source,probe=measured.name,measured_signs=[1],measured_direction_norm=norm,
        one_sided_error_per_norm=errors.tolist(),records=records,
        proposed_margin_metrics=[n for n in names if 'curvature' in n or 'energy' in n],
        guaranteed_bound=False,goal_complete=False))
    receipt(stage,dict(status='completed_current_gate_band_known_direction_error_audit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('GATE_BAND_CURRENT_ERROR',dict(norm=norm,paths=len(ids),measured_signs=[1]),flush=True)


if __name__=='__main__':main()
