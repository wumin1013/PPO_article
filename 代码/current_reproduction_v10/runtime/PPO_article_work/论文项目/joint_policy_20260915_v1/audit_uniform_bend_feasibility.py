"""A necessary local radius bound, independent of nonlinear solver success."""
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_band_bend_current_waveform_model_20260917'
    reference=BASE/'canonical_gate_band_bend_learning_reference_20260917'
    solve=BASE/'canonical_gate_band_bend_energy_solve_20260917'
    stage=register('canonical_gate_band_uniform_bend_feasibility_20260917',
        [__file__,source/'results.json',source/'receipt.json',reference/'basis.npz',
         reference/'receipt.json',solve/'optimizer.json',solve/'receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,dual_iterations=1000),
        'Determine whether the uniform5% energy target is already impossible '
        'inside the registered metric radius for the LOCAL convex waveform '
        'model. Each convex squared-density energy is lower bounded by its '
        'tangent at zero. A nonnegative combination of these necessary linear '
        'inequalities yields radius >= -lambda.b / ||lambda.A|| after metric '
        'whitening, by Cauchy-Schwarz. This inequality holds for ANY nonnegative '
        'lambda, regardless of convergence of the dual search. All omitted '
        'shape, corridor, action and brake constraints could only tighten it. '
        'It is not a physical impossibility result or permission to loosen '
        'constraints. Save the complete checkable certificate and each '
        'individual necessary radius to distinguish joint demand from one '
        'unresponsive corner. No new training or rollout.')
    start=time.perf_counter();data=read(source/'results.json');rows=[];labels=[]
    with np.load(reference/'basis.npz') as z:chol=np.linalg.cholesky(z['metric'])
    for record in data['records']:
        if 'waveform_file' not in record:continue
        with np.load(source/record['waveform_file']) as z:
            rows.extend(solve_triangular(chol,z['energy_linear'].T,lower=True).T/z['energy_constant'][:,None])
        labels.extend([f"path{record['path']}_{name}" for name in ('xy_energy','physical_energy')])
    matrix=np.asarray(rows);target=.05;bound=np.full(len(rows),-target)
    gram=matrix@matrix.T
    result=minimize(lambda v:.5*v@gram@v+bound@v,np.zeros(len(rows)),
        jac=lambda v:gram@v+bound,method='L-BFGS-B',bounds=[(0.,None)]*len(rows),
        options=dict(maxiter=1000,ftol=1e-15,gtol=1e-12))
    weights=np.maximum(result.x,0.);normal=weights@matrix
    lower=float(-(weights@bound)/np.linalg.norm(normal));radius=.1
    individual=target/np.linalg.norm(matrix,axis=1)
    # Independently reconstruct the combination in the original parameter metric.
    primal_normal=chol@normal
    dual_norm=np.linalg.norm(solve_triangular(chol,primal_normal,lower=True))
    assert abs(dual_norm-np.linalg.norm(normal))<1e-12
    np.savez_compressed(stage/'certificate.npz',matrix=matrix,bound=bound,
        nonnegative_weights=weights,labels=np.array(labels),metric_cholesky=chol)
    write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],
        energy_target=target,registered_radius=radius,necessary_radius_lower_bound=lower,
        uniform_target_locally_infeasible=lower>radius+1e-8,
        individual_necessary_radii=dict(zip(labels,individual.tolist())),
        certificate_support=[dict(label=n,weight=float(v)) for n,v in zip(labels,weights) if v>1e-8],
        dual_search_message=str(result.message),certificate_requires_search_convergence=False,
        physical_impossibility_claim=False,goal_complete=False))
    receipt(stage,dict(status='completed_uniform_energy_necessary_radius_audit',
        uniform_target_locally_infeasible=lower>radius+1e-8,necessary_radius_lower_bound=lower,
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('UNIFORM_ENERGY_RADIUS_BOUND',lower,'registered',radius,
          'max_individual',float(individual.max()),flush=True)


if __name__=='__main__':main()
