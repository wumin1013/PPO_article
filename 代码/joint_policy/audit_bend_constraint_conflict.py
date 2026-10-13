"""Checkable lower-radius certificates from subsets of the local constraints."""
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from waveform_constraint_model import waveform_facets
from solve_continuous_sensor_credit import IMPROVEMENT_NAMES
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    wave=BASE/'canonical_gate_band_bend_current_waveform_model_20260917'
    ref=BASE/'canonical_gate_band_bend_learning_reference_20260917'
    events=BASE/'canonical_gate_band_bend_current_trigger_boundaries_20260917'
    proposals=[BASE/n for n in ('canonical_gate_band_bend_control_restore_20260917',
        'canonical_gate_band_bend_energy_solve_20260917','canonical_gate_band_bend_aggregate_solve_20260917')]
    stage=register('canonical_gate_band_bend_constraint_conflict_20260917',
        [__file__,wave/'results.json',wave/'receipt.json',ref/'basis.npz',ref/'results.json',
         events/'constraints.npz',events/'receipt.json',BASE/'waveform_constraint_model.py']+
        [f/n for f in proposals for n in ('last_iterate.npz','optimizer.json','receipt.json')],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=150,dual_iterations_per_arm=2000),
        'Audit the failed aggregate5% local energy target without new physical '
        'experiments or expanded training. Convex waveform quantities give '
        'globally supporting linear inequalities at zero and each saved solver '
        'iterate. Keep labeled groups for energy target, existing per-corner '
        'shape/energy/curvature limits, common15% shape gain, original brake '
        'branch, half-corridor local displacement and held-action support. '
        'Nonnegative dual combinations provide necessary metric-radius lower '
        'bounds regardless of optimizer convergence. Compare cumulative '
        'groups and bounded leave-one-group-out audits; a bound>0.1 certifies '
        'incompatibility only inside this affine waveform learning proposal. '
        'A smaller bound does NOT prove feasibility. Save the exact matrix and '
        'weights for independent checking. No runtime or physical-limit change.')
    start=time.perf_counter();data=read(wave/'results.json');physical=read(ref/'results.json')
    names=physical['metric_names'];base={r['path']:dict(zip(names,r['values'])) for r in physical['baseline']['metrics']}
    with np.load(ref/'basis.npz') as z:
        chol=np.linalg.cholesky(z['metric']);support=z['support'].copy()
    points=[np.zeros(768)]
    for folder in proposals:
        with np.load(folder/'last_iterate.npz') as z:points.append(z['direction'].ravel())
    rows=[];bounds=[];labels=[];groups=[]
    def add(row,bound,label,group):
        rows.append(row);bounds.append(float(bound));labels.append(label);groups.append(group)
    all_values=[{} for _ in points]
    for r in data['records']:
        p=r['path']
        with np.load(wave/r['xy_file']) as z:g=z['gradient'].copy()
        margin=.5*(1.25-r['segment_max_distance_mm'])
        for k,x in enumerate(points[1:],1):
            displacement=np.einsum('tcd,d->tc',g,x);norm=np.linalg.norm(displacement,axis=1)
            t=int(norm.argmax());row=displacement[t]@g[t]/max(norm[t],1e-30)
            add(row,margin,f'{p}_corridor_at{k}','corridor_reserve')
        if 'waveform_file' not in r:continue
        with np.load(wave/r['waveform_file']) as z:w={n:z[n].copy() for n in z.files}
        for k,x in enumerate(points):
            d=waveform_facets(w['nominal'],w['gradient'],w['valid'],float(w['turn_deg']),x)
            q=w['energy_quadratic']@x
            e=w['energy_constant']+w['energy_linear']@x+q@x
            for j,n in enumerate(('full_xy_normalized_energy','full_physical_normalized_energy')):
                d[n]=(float(e[j]),w['energy_linear'][j]+2*q[j],0.)
            all_values[k][p]=d
            for n,(v,row,_) in d.items():
                group='energy_nonincrease' if 'energy' in n else 'curvature_peak' if 'curvature' in n else 'shape_nonincrease'
                b=base[p][n]*(1.-1e-6 if group in ('energy_nonincrease','curvature_peak') else 1.)
                add(row,b-v+row@x,f'{p}_{n}_at{k}',group)
    for k,(x,current) in enumerate(zip(points,all_values)):
        for name in (*IMPROVEMENT_NAMES,'full_xy_normalized_energy','full_physical_normalized_energy'):
            fraction=.05 if 'energy' in name else .15
            value=sum(v[name][0] for v in current.values())
            row=sum((v[name][1] for v in current.values()),np.zeros(768))
            b=(1.-fraction)*sum(base[p][name] for p in current)
            add(row,b-value+row@x,f'total_{name}_at{k}',
                'energy_target' if 'energy' in name else 'shape_reduction')
    with np.load(events/'constraints.npz') as z:
        for a,b,label in zip(z['matrix'],z['bound'],z['labels']):add(a,b,str(label),'brake_branch')
    selected=set()
    for x in points[1:]:
        action=support@x
        selected.update(np.argsort(abs(action))[-50:].tolist())
    for j in sorted(selected):
        for sign in (-1,1):add(sign*support[j],.5,f'action{j}_{sign}','action_support')
    matrix=solve_triangular(chol,np.asarray(rows).T,lower=True).T
    bound=np.asarray(bounds);groups=np.array(groups);norms=np.linalg.norm(matrix,axis=1)
    use=norms>1e-12
    assert np.all(bound[~use]>=-1e-8)
    matrix=matrix[use]/norms[use,None];bound=bound[use]/norms[use]
    groups=groups[use];labels=np.array(labels)[use]
    categories=['energy_target','energy_nonincrease','curvature_peak','shape_nonincrease',
                'shape_reduction','brake_branch','corridor_reserve','action_support']
    cases=[('energy_only',['energy_target']),('energy_and_corner_shape',categories[:4]),
           ('plus_shape_reduction',categories[:5]),('all',categories)]
    cases.extend((f'omit_{group}',[g for g in categories if g!=group]) for group in categories[1:])
    results=[];weights=[]
    for name,included in cases:
        assert time.perf_counter()-start<145
        mask=np.isin(groups,included);a=matrix[mask];b=bound[mask];gram=a@a.T
        fit=minimize(lambda v:.5*v@gram@v+b@v,np.zeros(len(b)),jac=lambda v:gram@v+b,
            method='L-BFGS-B',bounds=[(0.,None)]*len(b),
            options=dict(maxiter=2000,ftol=1e-14,gtol=1e-10))
        dual=np.maximum(fit.x,0.);normal=dual@a;normal_norm=np.linalg.norm(normal)
        lower=max(0.,float(-dual@b/max(normal_norm,1e-30)))
        full=np.zeros(len(bound));full[mask]=dual;weights.append(full)
        totals={g:float(sum(full[groups==g])) for g in categories}
        row=dict(arm=name,groups=included,necessary_radius_lower_bound=lower,
            local_incompatibility_certified=lower>.1+1e-8,dual_weight_by_group=totals,
            message=str(fit.message),iterations=int(fit.nit),constraints=int(mask.sum()))
        results.append(row);write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],
            records=results,smaller_bound_does_not_prove_feasibility=True,physical_impossibility_claim=False))
        print('BEND_CONSTRAINT_BOUND',name,lower,flush=True)
    np.savez_compressed(stage/'certificates.npz',matrix=matrix,bound=bound,groups=groups,
        labels=labels,nonnegative_weights=np.array(weights),arms=np.array([r['arm'] for r in results]))
    receipt(stage,dict(status='completed_local_bending_constraint_audit',arms=len(results),
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
