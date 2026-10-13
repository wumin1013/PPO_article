"""Use isolated observable support to learn a substantive outgoing-shape correction."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog
from joint_teacher_state_policy import features,state_pair
from observable_memory_joint_policy import dynamic_context,future_bend_context
from consolidate_joint_teacher_policy import collect_data
from audit_context_restoration_scaling import solve_correction
from duration_joint_metrics import guards,penalty
from shape_event_credit import shape_event_learning_guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--midturn-support',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_decoded_joint_step_learning_20260920'
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)=='4b356283a0641c9aadba71f8c603fa02b8669592c607e520d159f48b987ac6b9'
    md=torch.load(parent_file,map_location='cpu',weights_only=False)
    data=read(source/'baseline.json')
    data['baseline']=read(source/'results.json')['candidates'][0]
    baseline=data['baseline']['records'];selected=[r for r in baseline if r['path']==104]
    retained=[next(o['baseline_shape'] for o in data['retained'] if o['path']==104)]
    u0=np.asarray(md['coefficients']).reshape(3,7,7,3).copy();kernel=3
    response_source=BASE/'canonical_gate_band_supported_outgoing_joint_learning_20260920'
    stage=register('canonical_gate_band_midturn_supported_joint_learning_20260920' if args.midturn_support else
                   'canonical_gate_band_supported_outgoing_joint_learning_20260920',
        [__file__,parent_file,source/'receipt.json',source/'selection_review.json',source/'raw_result_audit.json',
         source/'intrinsic_curve_audit.json',BASE/'consolidate_joint_teacher_policy.py',
         BASE/'observable_memory_joint_policy.py',BASE/'shape_event_credit.py',
         BASE/'learn_intrinsic_joint_recovery.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py']+
         ([response_source/'receipt.json',response_source/'state_joint/results.json',
           response_source/'local_response.json',response_source/'local_capacity_audit.json',
           BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920/replayed_states.npz'] if args.midturn_support else []),
        dict(wall_s=240 if args.midturn_support else 180,workers=1,symmetric_vectors=18,maximum_joint_vectors=8,maximum_head_controls=2,
             finite_difference_steps=[.005,.01,.01],radii=[1.,4.,16.],
             required_predicted_exit_reduction_mm=.005,coefficient_bounds=[-1.,1.]),
        ('The measured9-dimensional pre-peak response permits only0.0000548mm '
         'linear predicted exit reduction under current guards. Add one compact '
         'radial center at the current104 training observation near320ms, after '
         'the earlier300ms support and before the332ms physical curvature peak. '
         'Choose its radius from distances to ALL current training states; '
         'time/path select the training center only, never enter the actor. '
         'Verify zero-extension decoded-action identity on the seven changed '
         'routes and all18 earlier response traces before reusing them. Learn '
         'nine new instantaneous/memory three-action weights and coordinate '
         'them jointly with the nine existing300ms weights. Collect only18 '
         'new symmetric complete vectors. All original acceptance requirements '
         'remain; this tests added observable temporal support, not a timed rule. '
         if args.midturn_support else '')+
        'The decoded16 research candidate passes all original and intrinsic '
        'guards but leaves substantial outgoing excursions. Kernel3 supplies '
        'strong visible pre-peak support on104 only, whereas the earlier280/360ms '
        'kernels are weak there. Replay all seven changed trajectories and combine '
        'the unchanged81-path audit with parent support to prove other87 routes '
        'are unchanged by this kernel. Learn nine constant-context weights over '
        'all three actions and instantaneous/two causal memories, relative to '
        'decoded16. Collect18 actual complete +/- responses, demand a predicted '
        'outgoing reduction of at least0.005mm under ALL current per-route '
        'guards, and test at most8 joint proposals/restorations. If feasible, '
        'measure steering-only and feed/preview-only learned-weight controls. '
        'Retain unchanged0:300ms physical prefix. No runtime path/time input, '
        'angle/speed table, rule deployment, PPO or sealed tests. Do not continue '
        'microscopic boundary repairs if substantial shape descent is infeasible.')
    started=time.perf_counter();deadline=started+(225 if args.midturn_support else 165);folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data)
    choice=dict(paths=[104],variable_kernels=[3]);write_json(stage/'support_audit.json',dict(selected=choice))
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    def score(records):return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def entry(records,u,tag):
        original=np.r_[old_guards(records),guards(records[0],selected[0])]
        extra=intrinsic_guards(records,selected)
        return dict(tag=tag,coefficients=np.asarray(u).reshape(-1).tolist(),records=records,
            score=score(records),measured_inequalities=original.tolist(),
            augmented_measured_inequalities=np.r_[original,extra].tolist(),
            augmented_learning_inequalities=np.r_[shape_event_learning_guards(records,selected,original),extra].tolist(),
            actual_rerun_paths=[104],exact_zero_support_paths=[])
    zero=entry(copy.deepcopy(selected),u0,'reused_current');history=[zero];md['reused_candidates']=1
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['augmented_measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    def violation(c):return float(np.linalg.norm(np.minimum(c['augmented_measured_inequalities'],0.)))
    def outgoing(c):return c['records'][0]['shape']['out_edge_outward_mm']
    pool=None;blocks=[]
    try:
        replay=collect_data(md,md['parent'],[r for r in baseline if r['path'] in [14,104,105,204,205,304,305]],deadline=deadline)
        f=features(md,replay['observations'],replay['innovations']);mask=np.any(f[:,kernel]!=0.,axis=1)
        assert sorted(np.unique(replay['path_ids'][mask]).tolist())==[104]
        assert (replay['decisions'][mask]==15).all()
        parent_support=read(BASE/'canonical_gate_band_preemptive_joint_credit_learning_20260920/support_audit.json')
        assert parent_support['phase_paths'][3]==[104]
        assert read(source/'raw_result_audit.json')['unchanged_other_paths_bitwise']
        kernels=[3]
        def extend_coefficients(old_coefficients):
            old_coefficients=np.asarray(old_coefficients).reshape(3,7,7,3)
            extended=np.zeros((3,7,8,3));extended[:,:,:7]=old_coefficients
            return extended
        if args.midturn_support:
            with np.load(BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920/replayed_states.npz') as z:
                old_replay={k:z[k].copy() for k in ('observations','innovations','path_ids','decisions')}
            unchanged=~np.isin(old_replay['path_ids'],[14,104,105,204,205,304,305])
            all_replay={k:np.concatenate([old_replay[k][unchanged],replay[k]]) for k in old_replay}
            a,b=state_pair(all_replay['observations'],all_replay['innovations'])
            mean,std=np.asarray(md['mean']),np.asarray(md['std'])
            a,b=(a-mean)/std,(b-mean)/std
            center_id=np.flatnonzero((all_replay['path_ids']==104)&(all_replay['decisions']==16)).item()
            center=a[center_id]
            distances=np.minimum(np.sqrt(np.mean((a-center)**2,axis=1)),np.sqrt(np.mean((b-center)**2,axis=1)))
            other=np.arange(len(a))!=center_id
            radius=.5*float(distances[other].min());assert radius>1e-6
            x=all_replay['observations'][center_id];bend=float(future_bend_context(x))
            for key,value in [('centers',center),('per_center_mean',mean),('per_center_std',std),
                ('per_center_context_mean',dynamic_context(x)),('per_center_context_scale',np.array([5.,100.]))]:
                md[key]=np.concatenate([np.asarray(md[key]),np.asarray(value)[None]])
            for key,value in [('radii',radius),('per_center_geometry_mean',bend),('per_center_geometry_scale',.02),
                ('observed_bend_metric_centers',bend),('observed_bend_metric_weights',0.)]:
                md[key]=np.r_[md[key],value]
            u0=extend_coefficients(u0);md['coefficients']=u0.reshape(-1,3)
            zero['coefficients']=u0.reshape(-1).tolist();kernels=[3,7]
            f_new=features(md,all_replay['observations'],all_replay['innovations'])
            touched=np.any(f_new[:,7]!=0.,axis=1)
            assert np.flatnonzero(touched).tolist()==[center_id]
            identity=collect_data(md,md['parent'],[r for r in baseline if r['path'] in [14,104,105,204,205,304,305]],deadline=deadline)
            previous=read(response_source/'state_joint/results.json')['candidates'];pairs_old=read(response_source/'local_response.json')['pairs']
            for c in previous[1:]:
                trial=copy.deepcopy(md);trial['coefficients']=extend_coefficients(c['coefficients']).reshape(-1,3)
                collect_data(trial,trial['parent'],c['records'],deadline=deadline)
                c=copy.deepcopy(c);c['coefficients']=trial['coefficients'].reshape(-1).tolist();c['tag']='reused_'+c['tag'];history.append(c)
            md['reused_candidates']=len(history)
            write_json(stage/'zero_extension_identity.json',dict(parent_decisions=len(identity['path_ids']),
                reused_physical_vectors=len(history)-1,all_decoded_actions_bitwise=True,
                new_center_training_selector=dict(path=104,decision=16),radius=radius,
                support_on_all_current_training_states=dict(paths=[104],decisions_ms=[320]),
                no_runtime_path_time_angle_input=True,new_physical_runs=0))
            torch.save(md,stage/'initial_policy.pt')
        write_json(stage/'zero_support_audit.json',dict(replayed_paths=7,replayed_decisions=len(replay['decisions']),
            actual_kernel_paths=[104],other81_parent_rows_bitwise_unchanged=True,
            all_other87_new_policy_trajectories_exact_by_zero_support=True,
            policy_uses_observable_state_only=True))
        with np.load(selected[0]['trace']) as z:prefix=z['rows'][:300].copy()
        pool=mp.get_context('spawn').Pool(1,worker_initialize,(md,selected,folder,deadline))
        cache={tuple(np.round(c['coefficients'],12)):c for c in history}
        def evaluate(u,kind):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.) and len(history)-md['reused_candidates']<28
                tag=f'outgoing_joint_{len(history)-md["reused_candidates"]}'
                records=pool.map(worker_evaluate,[(104,u,tag)])
                with np.load(records[0]['trace']) as z:assert np.array_equal(z['rows'][:300],prefix)
                c=entry(records,u,tag);c['kind']=kind;history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('OUTGOING_JOINT',tag,kind,outgoing(c),c['score'],violation(c),flush=True)
            return cache[key]
        steps=np.tile([.005,.01,.01],3*len(kernels));dimensions=len(steps)
        def expand(v):
            out=np.zeros_like(u0);values=np.asarray(v)*steps
            for i,k in enumerate(kernels):out[:,0,k]=values[9*i:9*i+9].reshape(3,3)
            return out.reshape(-1)
        size=len(zero['augmented_learning_inequalities']);jac=np.zeros((size,dimensions));g=np.zeros(dimensions);out_grad=np.zeros(dimensions);pairs=[]
        first_coordinate=0
        if args.midturn_support:
            measured=read(response_source/'local_response.json')
            jac[:,:9]=measured['jacobian'];g[:9]=measured['loss_gradient'];out_grad[:9]=measured['outgoing_gradient']
            pairs=[dict(p,plus='reused_'+p['plus'],minus='reused_'+p['minus']) for p in pairs_old]
            first_coordinate=9
        for j in range(first_coordinate,dimensions):
            e=np.eye(dimensions)[j];plus=evaluate(u0.reshape(-1)+expand(e),'axis_plus');minus=evaluate(u0.reshape(-1)-expand(e),'axis_minus')
            jac[:,j]=.5*(np.asarray(plus['augmented_learning_inequalities'])-minus['augmented_learning_inequalities'])
            g[j]=.5*(plus['score']-minus['score']);out_grad[j]=.5*(outgoing(plus)-outgoing(minus))
            pairs.append(dict(coordinate=j,plus=plus['tag'],minus=minus['tag']))
        write_json(stage/'local_response.json',dict(pairs=pairs,steps=steps.tolist(),jacobian=jac.tolist(),
            loss_gradient=g.tolist(),outgoing_gradient=out_grad.tolist()))
        proposal=None;values=np.asarray(zero['augmented_learning_inequalities'])
        for radius in (1.,4.,16.):
            a=np.vstack([-jac,out_grad]);b=np.r_[values,-.005]
            scale=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
            result=linprog(g,A_ub=a/scale[:,None],b_ub=b/scale,bounds=[(-radius,radius)]*dimensions,method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            blocks.append(dict(radius=radius,success=bool(result.success),status=str(result.message)))
            if result.success:proposal=result.x;break
        if proposal is not None:
            current=evaluate(u0.reshape(-1)+expand(proposal),'joint_substantial')
            blocks[-1].update(tag=current['tag'],predicted_outgoing_change=float(out_grad@proposal))
            jac+=np.outer(np.asarray(current['augmented_learning_inequalities'])-values-jac@proposal,proposal)/(proposal@proposal)
            g+=(current['score']-zero['score']-g@proposal)*proposal/(proposal@proposal)
            out_grad+=(outgoing(current)-outgoing(zero)-out_grad@proposal)*proposal/(proposal@proposal)
            for iteration in range(7):
                if feasible(current):break
                # Restore protections while keeping the requested substantial exit improvement.
                matrix=np.vstack([jac,-out_grad]);values=np.r_[current['augmented_learning_inequalities'],outgoing(zero)-.005-outgoing(current)]
                solved=solve_correction(matrix,g,values,zero['score']-.001-current['score'],radius=4.)
                blocks.append(dict(iteration=iteration,anchor=current['tag'],**solved))
                if not solved['success']:break
                step=np.asarray(solved['scaled_parameter_step'])
                c=evaluate(np.asarray(current['coefficients'])+expand(step),'joint_restoration')
                blocks[-1].update(tag=c['tag'],violation=violation(c))
                jac+=np.outer(np.asarray(c['augmented_learning_inequalities'])-current['augmented_learning_inequalities']-jac@step,step)/(step@step)
                g+=(c['score']-current['score']-g@step)*step/(step@step)
                out_grad+=(outgoing(c)-outgoing(current)-out_grad@step)*step/(step@step)
                if violation(c)<violation(current):current=c
                write_json(stage/'recovery.json',dict(blocks=blocks,nearest=current['tag']))
            if feasible(current) and outgoing(current)<=outgoing(zero)-.005:
                delta=np.asarray(current['coefficients']).reshape(u0.shape)-u0
                steering=delta.copy();steering[...,1:]=0.
                feed_preview=delta.copy();feed_preview[...,0]=0.
                for label,d in [('steering_only',steering),('feed_preview_only',feed_preview)]:
                    c=evaluate(u0+d,label);blocks.append(dict(control=label,tag=c['tag'],feasible=feasible(c),outgoing_mm=outgoing(c)))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_supported_outgoing_learning',error=str(exc),wall_s=time.perf_counter()-started,
            actual_new_candidates=len(history)-md['reused_candidates'],adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
