"""Use unfiltered-XY/dynamics curvature credit and measured function secants."""
import argparse
import time
import numpy as np
import torch
from audit_xy_curvature_resolution import curvature_metrics, CURVATURE_NAMES
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fine-remainder')
    parser.add_argument('--stage',default='raw_xy_physical_constraint_credit_20260917')
    parser.add_argument('--full-space-extra-credit')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.full_space_extra_credit:
        assert args.fine_remainder and (BASE/args.full_space_extra_credit/'receipt.json').exists()
    source=BASE/'current_dense_constraint_credit_collection_20260917'
    small=BASE/'small_amplitude_constraint_credit_20260917'
    prepared=BASE/'direct_constraint_parameter_preparation_20260917'
    local=BASE/'amplitude_secant_constraint_credit_20260917'
    checks=['direct_constraint_composition_probe_20260917','amplitude_secant_composition_probe_20260917']
    proposals=['direct_full_subspace_constraint_solve_20260917','amplitude_secant_constraint_solve_20260917']
    if args.fine_remainder:
        assert (BASE/args.fine_remainder/'receipt.json').exists()
        checks+=['raw_xy_physical_composition_probe_20260917','empirical_curvature_margin_composition_probe_20260917']
        proposals+=['raw_xy_physical_constraint_solve_20260917','empirical_curvature_margin_feasible_proposal_20260917']
    stage=register(args.stage,
        [__file__,BASE/'audit_xy_curvature_resolution.py',
         BASE/'actual_xy_curvature_resolution_audit_20260917/results.json',
         source/'results.json',source/'receipt.json',small/'results.json',small/'receipt.json',
         local/'gradients.npz',prepared/'support.npz']+
         ([BASE/args.fine_remainder/'results.json',BASE/args.fine_remainder/'receipt.json',
           BASE/'raw_xy_physical_constraint_credit_20260917/gradients.npz'] if args.fine_remainder else [])+
         ([BASE/args.full_space_extra_credit/'results.json',BASE/args.full_space_extra_credit/'queries.npz',
           BASE/args.full_space_extra_credit/'receipt.json',
           BASE/'full_feedback_parameter_space_preparation_20260917/support.npz',
           BASE/'geometry_homotopy_exact_preparation_20260917/jacobians.npz',
           BASE/'prepare_geometry_homotopy.py',BASE/'diagnose_local_basis_fit.py',
           BASE/'recentered_reverse_shape_credit_learning_20260917/candidate_1.5.pt'] if args.full_space_extra_credit else [])+
         [BASE/p/f for p in checks for f in ('results.json','receipt.json')]+
         [BASE/p/'direction.npz' for p in proposals],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=180),
        ('Extend to19 current training routes and all768 kernel coefficients plus an optional geometry-blend Jacobian. The added16/31 routes replace blind exact-action freezing with measured failure exclusions and continuous-margin credit. Preserve every old measured direction via the exact coefficient embedding; verify the old366-dimensional metric-gradient projection. No full-space proposal may ignore the11 newly measured failures. Geometry is an additional optional arm, not a combined causal attribution. ' if args.full_space_extra_credit else '')+
        ('Matched all-route amplitude completion: combine the already measured three small-amplitude paths with the disjoint14-route remainder. Recompute ALL metrics from this one uniform amplitude, at the identical parent and832 states. Compare uncalibrated gradients against saved whole-policy directions before treating the smaller amplitude as improved credit. No repeated three-route physical work. The later secant calibration now uses four explicitly listed measured directions; this is not a single-factor learning result. ' if args.fine_remainder else '')+
        'Saved XY confirms fixed-arc interpolation peak aliasing: paths5/7 old spatial '
        'peak increases while unfiltered1ms XY circumcircle AND independent v/a peaks '
        'decrease; path9 truly worsens under both. Replace only the smoothness constraint '
        'estimators with unfiltered XY curvature energy/peak and require independent '
        'physical energy/peak nonincrease too. Retain/report original spatial metrics '
        'as diagnostics; do not change corridor or V/A/J checks, actor, reward or reference '
        'geometry. Recompute curvature credit from all already saved single-action '
        'suffixes, using matched smaller amplitudes for3/7/13. Other metric gradients '
        'are reused. Calibrate forward/backward gradients to BOTH measured whole-policy '
        'directions where available, using minimum dual-metric multisection correction. '
        'Require all three local models in the next proposal and actual full-policy '
        'validation; no safety certificate or adoption from estimator replacement.')
    started=time.perf_counter()
    with np.load(local/'gradients.npz') as z:
        ids=z['path_id'].tolist();old=z['gradient'].copy();jac=z['query_jacobian'].copy()
        exclusions=z['failed_axis_constraints'].copy()
    names=['in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
           CURVATURE_NAMES[0],CURVATURE_NAMES[1],'segment_max_distance_mm','complete_time_s',
           CURVATURE_NAMES[2],CURVATURE_NAMES[3]]
    data=read(source/'results.json');fine=read(small/'results.json')
    if args.fine_remainder:
        extra=read(BASE/args.fine_remainder/'results.json')
        assert not ({r['path'] for r in fine['baselines']}&{r['path'] for r in extra['baselines']})
        fine['baselines']+=extra['baselines'];fine['records']+=extra['records']
        assert {r['path'] for r in fine['baselines']}==set(ids) and len(fine['records'])==4992
    full_embedding=None
    if args.full_space_extra_credit:
        from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
        from diagnose_local_basis_fit import feature_batches
        from prepare_geometry_homotopy import mix_jacobian
        from probe_one_decision_credit import AMPLITUDE
        extra=read(BASE/args.full_space_extra_credit/'results.json')
        assert {r['path'] for r in extra['baselines']}=={16,31}
        original_query_count=len(data['queries'])
        data['queries'] += [dict(q,query=original_query_count+q['query']) for q in extra['queries']]
        data['baselines']+=extra['baselines'];data['records']+=extra['records']
        fine['baselines']+=extra['baselines'];fine['records']+=extra['records']
        ids += [r['path'] for r in extra['baselines']]
        with np.load(source/'queries.npz') as z:old_qx=z['observation'].copy()
        with np.load(BASE/args.full_space_extra_credit/'queries.npz') as z:new_qx=z['observation'].copy()
        qx=np.concatenate([old_qx,new_qx])
        p=KernelLocalFeedbackPolicy(256)
        p.load_state_dict(torch.load(BASE/'recentered_reverse_shape_credit_learning_20260917/candidate_1.5.pt',map_location='cpu',weights_only=False)['policy'])
        assert np.allclose(p.residual_scale.detach().numpy(),AMPLITUDE,rtol=0,atol=1e-8)
        f=feature_batches(p,qx);jac=np.zeros((len(qx),3,769))
        for h in range(3):jac[:,h,h*256:(h+1)*256]=f[:,h]*AMPLITUDE[h]
        with np.load(BASE/'geometry_homotopy_exact_preparation_20260917/jacobians.npz') as z:old_mix=z['query_mix_jacobian'].copy()
        jac[:,:,-1]=np.concatenate([old_mix,mix_jacobian(p,new_qx)])
        query_index={(q['path'],q['anchor']):q['query'] for q in data['queries']}
        bad=sorted({(r['path'],r['anchor'],r['axis'],r['direction']) for r in data['records']+fine['records'] if not r['success']})
        exclusions=np.asarray([sign*jac[query_index[path,anchor],head]/AMPLITUDE[head]
            for path,anchor,head,sign in bad])
        with np.load(BASE/'full_feedback_parameter_space_preparation_20260917/support.npz') as z:full_embedding=z['old_embedding'].copy()
    g=np.zeros((len(ids),len(names),jac.shape[-1]))
    replacement_paths={r['path'] for r in fine['baselines']}
    baseline={r['path']:r for r in data['baselines']}
    lookup={(r['path'],r['anchor'],r['axis'],r['direction']):r for r in data['records']}
    lookup.update({(r['path'],r['anchor'],r['axis'],r['direction']):r for r in fine['records']})
    nominal={};curvature=[];modes={}
    for path in ids:
        with np.load(BASE/baseline[path]['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        c=curvature_metrics(ref,rows) if len(ref)==3 else dict.fromkeys(CURVATURE_NAMES,0.)
        nominal[path]=(ref,rows,c)
        baseline[path]['full_time_s']=len(rows)*.001
    def metric_vector(record,curv):
        shape=record['shape'] or dict.fromkeys(names[:3],0.)
        return np.asarray([shape[k] for k in names[:3]]+
            [curv[k] for k in CURVATURE_NAMES[:2]]+
            [record['segment_distance']['max_distance_upper_mm'],record['full_time_s']]+
            [curv[k] for k in CURVATURE_NAMES[2:]])
    for q in data['queries']:
        path=q['path'];i=ids.index(path);ref,rows,b=nominal[path]
        base=metric_vector(baseline[path],b)
        for h in range(3):
            pair=[lookup[path,q['anchor'],h,s] for s in (-1,1)]
            vals=[]
            for r in pair:
                if not r['success']:vals.append(None);continue
                if len(ref)==3:
                    with np.load(BASE/r['trace']) as z:full=np.vstack([rows[:q['anchor']*20],z['suffix_rows']])
                    c=curvature_metrics(ref,full)
                else:c=dict.fromkeys(CURVATURE_NAMES,0.)
                vals.append(metric_vector(r,c))
                curvature.append(dict(path=path,anchor=q['anchor'],head=h,sign=r['direction'],
                    amplitude=r['latent_offset'],values=[c[k] for k in CURVATURE_NAMES]))
            if all(v is not None for v in vals):
                slope=(vals[1]-vals[0])/(pair[1]['latent_offset']-pair[0]['latent_offset']);mode='central'
            else:
                good=[j for j,v in enumerate(vals) if v is not None]
                slope=(vals[good[0]]-base)/pair[good[0]]['latent_offset'] if good else np.zeros(len(names))
                mode='one_sided' if good else 'neither_complete'
            g[i]+=slope[:,None]*jac[q['query'],h]
            modes[mode]=modes.get(mode,0)+1
    if args.full_space_extra_credit:
        with np.load(BASE/'full_feedback_parameter_space_preparation_20260917/support.npz') as z:metric=z['extended_metric'].copy()
    else:
        with np.load(prepared/'support.npz') as z:metric=z['metric'].copy()
    measured={path:[] for path in ids}
    for check,proposal in zip(checks,proposals):
        result=read(BASE/check/'results.json')
        with np.load(BASE/proposal/'direction.npz') as z:
            d=np.r_[z['direction'].ravel(),0.] if args.full_space_extra_credit else z['reduced_direction'].copy()
        baselines={r['path']:r['values'] for r in result['baseline_metrics']}
        lower,upper=result['candidates'];alpha=upper['coefficient']
        for j,(lo,hi) in enumerate(zip(lower['metrics'],upper['metrics'])):
            path=lo['path'];assert path==hi['path'] and lo['success'] and hi['success']
            ref,_,c=nominal[path]
            def extend(value,curv):
                return np.r_[value[:3],[curv[k] for k in CURVATURE_NAMES[:2]],value[5:7],
                    [curv[k] for k in CURVATURE_NAMES[2:]]]
            base=extend(baselines[path],c);pair=[]
            for r in (lower,upper):
                with np.load(BASE/r['traces'][j]) as z:rows=z['rows'].copy()
                cv=curvature_metrics(ref,rows) if len(ref)==3 else dict.fromkeys(CURVATURE_NAMES,0.)
                pair.append(extend(r['metrics'][j]['values'],cv))
            measured[path].append((d,(pair[1]-base)/alpha,(base-pair[0])/alpha))
    comparison=[]
    if args.fine_remainder:
        with np.load(BASE/'raw_xy_physical_constraint_credit_20260917/gradients.npz') as z:coarse=z['gradient'].copy()
        for m,name in enumerate(names):
            old_prediction=[];new_prediction=[];actual=[]
            for i,path in enumerate(ids):
                for d,up,down in measured[path]:
                    reduced=full_embedding.T@d[:768] if full_embedding is not None else d
                    old_prediction.append(float(coarse[i,m]@reduced));new_prediction.append(float(g[i,m]@d))
                    actual.append(float(.5*(up[m]+down[m])))
            a,b,c=map(np.asarray,(old_prediction,new_prediction,actual))
            signal=abs(c)>1e-6
            comparison.append(dict(metric=name,coarse_mean_absolute_error=float(abs(a-c).mean()),
                fine_mean_absolute_error=float(abs(b-c).mean()),mean_absolute_actual=float(abs(c).mean()),
                signal_pairs=int(signal.sum()),
                coarse_sign_agreement=float(np.mean(a[signal]*c[signal]>0)) if signal.any() else None,
                fine_sign_agreement=float(np.mean(b[signal]*c[signal]>0)) if signal.any() else None))
    forward=g.copy();backward=g.copy();errors=[]
    for i,path in enumerate(ids):
        items=measured[path]
        if not items:continue
        directions=np.column_stack([r[0] for r in items])
        correction=np.linalg.solve(directions.T@metric@directions,directions.T@metric)
        for variant,slot in ((forward,1),(backward,2)):
            slopes=np.column_stack([r[slot] for r in items])
            variant[i]+=(slopes-g[i]@directions)@correction
            errors.append(float(abs(variant[i]@directions-slopes).max()))
    assert max(errors)<1e-10 and time.perf_counter()-started<180
    embedding_error=None
    if full_embedding is not None:
        with np.load(BASE/'uniform_small_xy_physical_constraint_credit_20260917/gradients.npz') as z:previous=z['gradient'].copy()
        embedding_error=float(abs(g[:17,:,:768]@full_embedding-previous).max())
        assert embedding_error<1e-8
        write_json(stage/'baselines.json',dict(baselines=data['baselines'],queries=data['queries'],failed_axes=bad))
    np.savez_compressed(stage/'gradients.npz',gradient=g,gradient_variants=np.stack([g,forward,backward]),
        path_id=ids,metric_names=names,failed_axis_constraints=exclusions,query_jacobian=jac)
    write_json(stage/'results.json',dict(metric_names=names,smaller_amplitude_paths=sorted(replacement_paths),
        curvature_branch_metrics=curvature,nominal_curvature={p:c for p,(_,_,c) in nominal.items()},
        modes=modes,function_comparison=comparison,largest_secant_identity_error=max(errors),
        old_gradient_embedding_error=embedding_error,parameter_space='full_feedback_with_geometry' if full_embedding is not None else 'protected_residual',
        goal_complete=False))
    receipt(stage,dict(status='completed_raw_xy_physical_constraint_credit',modes=modes,
        new_physics_ticks=0,new_actor_decisions=0,wall_s=time.perf_counter()-started))
    print('XY_PHYSICAL_CREDIT',g.shape,modes,max(errors),flush=True)


if __name__=='__main__':main()
