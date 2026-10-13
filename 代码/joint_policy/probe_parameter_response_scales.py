"""Bounded single-path signed scale checks of a learned parameter direction."""
import argparse
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha
from waveform_constraint_model import waveform_facets


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','parent','arm','waveforms','stage'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--path',type=int,required=True)
    parser.add_argument('--scales',type=float,nargs='+',default=[-.0625,.0625,-.03125,.03125,-.015625,.015625])
    args=parser.parse_args();torch.set_num_threads(1)
    source=BASE/args.source;parent=BASE/args.parent;waves=BASE/args.waveforms
    selected=next(r for r in read(parent/'results.json')['candidates'] if r['model']==args.arm)
    model=parent/selected['model'];opt=read(source/'optimizer.json')
    assert opt['parent_sha256']==sha(model)
    assert len(args.scales)<=8 and all(0.<abs(v)<=.125 for v in args.scales)
    wd=read(waves/'results.json');assert wd['parent_sha256']==sha(model)
    record=next(r for r in wd['records'] if r['path']==args.path)
    with np.load(waves/record['waveform_file']) as z:w={k:z[k].copy() for k in z.files}
    with np.load(source/'direction.npz') as z:direction=z['direction'].copy();names=z['metric_names'].tolist()
    stage=register(args.stage,[__file__,model,source/'optimizer.json',source/'direction.npz',
        source/'receipt.json',parent/'results.json',waves/'results.json',waves/record['waveform_file'],
        BASE/'train_pilot.py',BASE/'canonical_gate_state.py'],
        dict(paths=[args.path],scales=args.scales,actor_calls=1500,physics_ticks=30000,wall_s=90),
        'Measure only the already identified training-path response at smaller equal signs. '
        'Reuse nominal and0.125 results; no new whole-screen evaluation. Compare raw shape '
        'to its unchanged affine response, record odd/even errors in analysis, and preserve '
        'full completion. This diagnoses derivative accuracy versus finite-step nonlinear '
        'response; it does not authorize larger steps or relax failed acceptance.')
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    policy=CanonicalGatePolicy(256);factory=checkpoint_environment_factory(metadata['architecture'])
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    j=next(j for j,r in enumerate(selected['results']) if r['original_path']==args.path)
    rs=[selected['results'][j]];ts=[BASE/selected['traces'][j]]
    baseline=metric_rows(rs,ts,review(rs,ts),names)[0]
    start=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0);records=[]
    for scale in args.scales:
        policy.load_state_dict(metadata['policy'])
        with torch.no_grad():policy.residual_coefficients.add_(torch.tensor(scale*direction,dtype=torch.float32))
        tag=f'scale_{scale:+g}'
        result=evaluate(policy,[np.asarray(paths[args.path])],stage,tag,None,costs,start+90,factory)
        assert costs['evaluation_actor_calls']<=1500 and costs['evaluation_physics_ticks']<=30000
        result=[dict(r,original_path=args.path) for r in result];trace=stage/tag/'0.npz'
        metrics=metric_rows(result,[trace],review(result,[trace]),names)[0]
        predicted={n:v[0] for n,v in waveform_facets(w['nominal'],w['gradient'],w['valid'],float(w['turn_deg']),scale*direction.ravel()).items()}
        records.append(dict(scale=scale,result=result[0],metrics=metrics,predicted=predicted,trace=str(trace.relative_to(BASE))))
        write_json(stage/'results.json',dict(parent_sha256=sha(model),baseline=baseline,metric_names=names,records=records,costs=costs,goal_complete=False))
        print('PARAMETER_RESPONSE_SCALE',scale,'exit',metrics['values'][names.index('out_edge_outward_mm')] if metrics['success'] else None,flush=True)
    receipt(stage,dict(status='completed_bounded_signed_parameter_scale_probe',costs=costs,wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
