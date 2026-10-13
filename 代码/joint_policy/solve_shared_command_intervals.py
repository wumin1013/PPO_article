"""Diagnostic shared-basis capacity for saved20ms command-heading intervals."""
import sys
import argparse
import time
import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extra-support')
    parser.add_argument('--stage',default='canonical_gate_band_shared_command_interval_capacity_20260918')
    args=parser.parse_args()
    sys.path.insert(0,str(BASE/'.runtime_dependencies'))
    import clarabel
    geometry=BASE/'canonical_gate_band_current_hold_geometry_20260918'
    intervals=BASE/'canonical_gate_band_current_hold_feasibility_20260918'
    source=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    data=read(source/'results.json');model=source/data['chosen']['model']
    nominal=read(intervals/'results.json')
    assert nominal['parent_sha256']==sha(model)==read(geometry/'results.json')['parent_sha256']
    extra_sources=[]
    extra_metadata={}
    if args.extra_support:
        extra_dir=BASE/args.extra_support
        extra_metadata=read(extra_dir/'results.json')
        assert extra_metadata['parent_sha256']==sha(model)
        extra_sources=[extra_dir/'support.npz',extra_dir/'results.json',extra_dir/'receipt.json']
    stage=register(args.stage,
        [__file__,model,source/'results.json',source/'receipt.json',geometry/'geometry.npz',
         geometry/'results.json',geometry/'receipt.json',intervals/'results.json',intervals/'receipt.json']+extra_sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,optimizer_wall_s_per_arm=60,
             arms=['unrestricted_update','half_degree_support'],free_steering_coefficients=256,
             extra_affine_parameters=extra_metadata.get('extra_parameters',0),
             extra_feature_kind=extra_metadata.get('extra_feature_kind','actor_output_weights') if args.extra_support else None,
             feed_preview_coefficients_fixed=True,maximum_physical_probe_paths_per_arm=58,
             later_physical_probe_scales=[.25,1.],later_decisions_per_arm=6000,later_wall_s_per_arm=120),
        'Distinguish shared-policy representability from an overly small update restriction. '
        'Independent saved20ms steering intervals are all nonempty. Fit the CURRENT256 '
        'steering residual coefficients jointly across every ordinary isolated-corner '
        'decision, using exact inverse-tanh interval inequalities and the unchanged '
        'path-balanced physical metric. Feed and preview coefficients stay fixed as '
        'a single-factor diagnostic. Compare unrestricted update with the SAME problem '
        'plus existing half-degree saved-action-change bounds over all64 routes. '
        'No parameter-radius cap is imposed: report the minimum required norm rather '
        'than silently enlarging a learning trust region. Verify every inequality '
        'and the exact decoded saved-state heading interval. The saved guide/XY remains '
        'frozen only in this algebraic diagnostic, so feasibility does not predict '
        'new closed-loop shape. A feasible bounded arm may receive at most two '
        'complete29-route probes at0.25/1; keep failures and compare actual reversal, '
        'exit, curvature, time and cruise. No adoption, deployed interval projection, '
        'new physical limits, curve templates or final-goal claim.')
    start=time.perf_counter()
    with np.load(geometry/'geometry.npz') as z:
        metric=z['metric_physical'][:256,:256].copy()
        support=z['support'].reshape(-1,3,768)[:,0,:256].copy()
        latent=z['latent'][:,0].copy();paths=z['state_path'].copy()
        lower=z['lower_physical'].reshape(-1,3)[:,0].copy()
        upper=z['upper_physical'].reshape(-1,3)[:,0].copy()
        weights=z['state_weight'].copy()
    if args.extra_support:
        with np.load(extra_dir/'support.npz') as z:
            assert np.array_equal(paths,z['state_path'])
            support=np.column_stack([support,z['extra_steering_features']])
            assert np.max(abs(metric-z['extended_metric'][:256,:256]))<1e-10
            metric=z['extended_metric'].copy()
    dimension=support.shape[1]
    chol=np.linalg.cholesky(metric)
    transform=solve_triangular(chol.T,np.eye(dimension),lower=False,check_finite=False)
    trace_by_path={r['original_path']:t for r,t in zip(data['full_candidate']['results'],data['full_candidate']['traces'])}
    signs={}
    for p in range(16):
        with np.load(BASE/trace_by_path[p]) as z:u,v=np.diff(z['reference'],axis=0)
        signs[p]=float(np.sign(u[0]*v[1]-u[1]*v[0]))
    rows=[];bounds=[];labels=[];indices=[];signed_features=[];lo_deg=[];hi_deg=[];sign_array=[]
    for item in nominal['records']:
        p=item['path'];k=item['decision'];index=int(np.flatnonzero(paths==p)[k]);sign=signs[p]
        feature=sign*support[index];z0=sign*latent[index]
        assert abs(90*np.tanh(z0)-item['nominal_residual_deg'])<1e-10
        lo,hi=item['lower_residual_deg'],item['upper_residual_deg']
        if lo>-90:
            rows.append(-feature);bounds.append(z0-np.arctanh(lo/90.));labels.append(f'{p}_{k}_lower')
        if hi<90:
            rows.append(feature);bounds.append(np.arctanh(hi/90.)-z0);labels.append(f'{p}_{k}_upper')
        indices.append(index);signed_features.append(feature);lo_deg.append(lo);hi_deg.append(hi);sign_array.append(sign)
    command_a=np.asarray(rows);command_b=np.asarray(bounds)
    indices=np.asarray(indices);sign_array=np.asarray(sign_array)
    lo_deg,hi_deg=np.asarray(lo_deg),np.asarray(hi_deg)
    output=[]
    for arm in ('unrestricted_update','half_degree_support'):
        a,b=command_a.copy(),command_b.copy()
        if arm=='half_degree_support':
            a=np.vstack([a,support,-support]);b=np.r_[b,upper,-lower]
        finite=np.isfinite(b);a,b=a[finite],b[finite]
        aw=a@transform;norm=np.linalg.norm(aw,axis=1);nonzero=norm>1e-14
        assert np.all(b[~nonzero]>=-1e-10)
        aw=aw[nonzero]/norm[nonzero,None];bw=b[nonzero]/norm[nonzero]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1
        settings.time_limit=60.;settings.max_iter=120
        settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
        fit=clarabel.DefaultSolver(sparse.eye(dimension,format='csc'),np.zeros(dimension),
            sparse.csc_matrix(aw),bw,[clarabel.NonnegativeConeT(len(bw))],settings).solve()
        x=transform@np.asarray(fit.x)
        decoded=90*np.tanh(latent+support@x)
        commands=sign_array*decoded[indices]
        violation=np.maximum(lo_deg-commands,commands-hi_deg)
        change=decoded-90*np.tanh(latent)
        passed=bool(str(fit.status) in ('Solved','AlmostSolved') and np.max(a@x-b)<=1e-8
                    and np.max(violation)<=1e-6 and
                    (arm=='unrestricted_update' or abs(change).max()<=.5+1e-6))
        record=dict(arm=arm,status=str(fit.status),verified_feasible=passed,
            free_parameters=dimension,extra_feature_kind=extra_metadata.get('extra_feature_kind','actor_output_weights') if args.extra_support else None,
            maximum_latent_violation=float(max(0.,np.max(a@x-b))),
            maximum_decoded_interval_violation_deg=float(max(0.,violation.max())),
            metric_norm=float(np.sqrt(x@metric@x)),maximum_steering_change_deg=float(abs(change).max()),
            path_balanced_rms_steering_change_deg=float(np.sqrt(weights@change**2)),
            conic_iterations=int(fit.iterations),conic_solve_s=float(fit.solve_time),
            radius_used_by_previous_learning=.0069,diagnostic_only=True)
        if str(fit.status)=='PrimalInfeasible':
            dual=np.asarray(fit.z)
            record['infeasibility_certificate']=dict(min_multiplier=float(dual.min()),
                stationarity_residual=float(abs(aw.T@dual).max()),bound_pairing=float(bw@dual))
        if passed:
            direction=np.zeros((3,256));direction[0]=x[:256]
            np.savez_compressed(stage/f'{arm}.npz',direction=direction,
                actor_output_delta=x[256:],
                extra_coefficients=x[256:],
                frozen_steering_change_deg=change,interval_violation_deg=violation)
        output.append(record)
        write_json(stage/'results.json',dict(parent_sha256=sha(model),parent_model=str(model.relative_to(BASE)),
            arms=output,interval_constraints=len(command_b),physical_trials_started=False,
            diagnostic_only=True,goal_complete=False))
        print('SHARED_COMMAND_CAPACITY',record,flush=True)
    receipt(stage,dict(status='completed_shared_command_interval_capacity',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start,
        feasible_arms=sum(r['verified_feasible'] for r in output),diagnostic_only=True,goal_complete=False))


if __name__=='__main__':main()
