"""Zero-physics correction audit; preserve the earlier fine-grid results."""
import json
import shutil
import numpy as np
import torch
from geometric_turn_metrics import geometric_turn_metrics
from search_complete_return import actor_parameter_key
from study_io import BASE, register, receipt, sha, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arc_case(radius, source_spacing):
    theta=np.linspace(-np.pi/2,0,int(np.ceil(np.pi*radius/2/source_spacing))+1)
    curve=np.c_[radius*np.cos(theta),radius+radius*np.sin(theta)]
    before=np.c_[np.linspace(-10,0,int(np.ceil(10/source_spacing))+1),np.zeros(int(np.ceil(10/source_spacing))+1)]
    after=np.c_[np.full(int(np.ceil(10/source_spacing))+1,radius),np.linspace(radius,radius+10,int(np.ceil(10/source_spacing))+1)]
    xy=np.vstack([before[:-1],curve,after[1:]])
    return np.array([xy[0],[radius,0],xy[-1]]),xy[1:]


def main():
    left=BASE/'correlated_parameter_search_20260916'
    right=BASE/'balanced_shape_time_learning_20260916'
    stage=BASE/'spatial_metric_resolution_audit_20260916'
    if stage.exists():
        assert not (stage/'receipt.json').exists()
        shutil.copy2(__file__,stage/'source/corrected_verify.py')
        write_json(stage/'correction.json',dict(reason='Initial audit stopped on unrounded coefficient identity. Coefficients differ by <=1.4e-17; actual float32 network parameters must match exactly and all raw arrays are checked.',
            corrected_source_sha256=sha(__file__),physics_ticks=0))
    else: stage=register('spatial_metric_resolution_audit_20260916',
        [__file__,BASE/'geometric_turn_metrics.py',BASE/'search_complete_return.py',
         BASE/'turn_geometry_gate_audit_20260916/receipt.json',left/'receipt.json',right/'receipt.json'],
        dict(physics_ticks=0,training_updates=0,saved_trace_pairs=192,wall_s=120),
        'Correct undersampled curvature-energy interpretation using 0.1/0.2 mm grids, analytic circles and rigid/mirror/repeated-point invariance; compare existing serial/parallel physics bit-for-bit. No new rollout.')
    cases=[]
    for radius in (.2,1.,2.):
        for raw in (.1,.01,.001):
            ref,rows=arc_case(radius,raw)
            for grid in (.01,.1,.2):
                metric=geometric_turn_metrics(ref,rows,grid)
                analytic=np.pi/(2*radius)
                cases.append(dict(radius=radius,raw_spacing=raw,grid=grid,analytic=analytic,
                    measured=metric['bending_energy_mm_inv'],relative_error=metric['bending_energy_mm_inv']/analytic-1))
            original=geometric_turn_metrics(ref,rows)
            rotation=np.array([[np.cos(.71),-np.sin(.71)],[np.sin(.71),np.cos(.71)]])
            transformed=geometric_turn_metrics(ref@rotation+7,rows@rotation+7)
            mirrored=geometric_turn_metrics(ref*[1,-1],rows*[1,-1])
            repeated=geometric_turn_metrics(ref,np.repeat(rows,3,axis=0))
            for other in (transformed,mirrored,repeated):
                assert abs(original['bending_energy_mm_inv']-other['bending_energy_mm_inv'])<1e-7
    for radius in (1.,2.):
        energies=[r['measured'] for r in cases if r['radius']==radius and r['grid']==.1]
        assert max(energies)/min(energies)<1.03
        assert all(abs(r['relative_error'])<.05 for r in cases if r['radius']==radius and r['grid']==.1)
    records=[]
    for label,folder in [('PPO_source','ar_conditional_joint_20260916/final_validation'),
                         ('correlated_g4','correlated_parameter_search_20260916/validation_4'),
                         ('balanced_g5','balanced_shape_time_learning_20260916/validation_5')]:
        for i in (1,2,3):
            path=BASE/folder/f'{i}.npz'
            with np.load(path) as d:
                fine=geometric_turn_metrics(d['reference'],d['rows'],.1)
                coarse=geometric_turn_metrics(d['reference'],d['rows'],.2)
            assert fine['raw_resolution_supported']
            records.append(dict(method=label,path=i,trace_sha256=sha(path),grid_01=fine,grid_02=coarse))
            print(label,i,round(fine['bending_energy_mm_inv'],3),round(coarse['bending_energy_mm_inv'],3),flush=True)
    l={r['tag']:r for r in read(left/'training_results.json')['candidates']}
    r={r['tag']:r for r in read(right/'training_results.json')['candidates']}
    model=torch.load(BASE/'reflection_shape_search_20260916/best_generation_5.pt',map_location='cpu',weights_only=False)['policy']
    basis=np.load(left/'basis.npz')['basis']
    weight,bias=model['actor.4.weight'].numpy(),model['actor.4.bias'].numpy()
    pairs=[]
    for candidate in range(12):
        tag=f'g2_c{candidate}'
        assert actor_parameter_key(l[tag]['coefficients'],basis,weight,bias)==actor_parameter_key(r[tag]['coefficients'],basis,weight,bias)
        for index in range(16):
            a=left/'training_traces'/f'{tag}_{index}.npz'
            b=right/'training_traces'/f'{tag}_{index}.npz'
            with np.load(a) as da,np.load(b) as db:
                assert da.files==db.files
                assert all(np.array_equal(da[k],db[k]) for k in da.files)
            pairs.append(dict(trace=a.name,left_sha256=sha(a),right_sha256=sha(b),arrays_exact_equal=True))
    result=dict(analytic_cases=cases,saved_geometry=records,parallel_pairs=pairs,
        duplicate_training_decisions=sum(r[f'g2_c{i}']['actor_calls'] for i in range(12)),
        conclusion='Earlier 0.01/0.02 mm energies alias raw polygon vertices and are superseded for magnitude interpretation. Corrected 0.1/0.2 mm estimates still show severe acute-turn concentration. Approximation underestimates very tight circles; no continuous-curvature certificate or final shape acceptance.',
        physics_ticks=0,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed',physics_ticks=0,training_updates=0,exact_trace_pairs=len(pairs),
        analytic_cases=len(cases),results_sha256=sha(stage/'results.json')))
    current=read(BASE/'当前执行状态.json')
    for folder in (right,stage):
        if not any(x['stage']==folder.name for x in current['completed']):
            current['completed'].append(dict(stage=folder.name,receipt_sha256=sha(folder/'receipt.json')))
    model=right/'best_generation_5.pt'
    assert sha(model)==next(g['model_sha256'] for g in read(right/'generations.json')['generations'] if g['generation']==5)
    current['preserved_candidates'].append(dict(model=model.relative_to(BASE).as_posix(),sha256=sha(model),
        successes='12/12 development',time_s=20.503,limitation='No outward excursion but 150-degree turning concentrated in 0.8 mm, low apex speed, no cruise. Not accepted.'))
    current['current']=dict(stage=stage.name,state='completed_zero_physics_diagnostic',training_running=False)
    current['next']='Register a bounded single-objective replacement contrast only after resolving spatial metric sampling. Original research remains active.'
    current['updated_utc']=read(stage/'receipt.json')['completed_utc']
    write_json(BASE/'当前执行状态.json',current)
    print('exact_pairs',len(pairs),'duplicate_cost',result['duplicate_training_decisions'],flush=True)


if __name__=='__main__':main()
