"""Bounded fresh-student pilot with physical recovery data and rebuilt feature maps."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from strong_policy_v4 import StrongPolicyV4,tensor_hash,load_strong_policy
from train_strong_fit_v4 import equivalent_loss

OUT=ROOT/'results/recovery_training_v6'
REG=ROOT/'evidence/recovery_training_candidate_v6.json'
PRIOR=ROOT/'results/physical_teacher_v5/collection_001'
MODEL=ROOT/'baseline_records/supplementary_reproduction/models/main.pt'
SEED=101
PERTURBATIONS=[(.8,.02),(.8,-.02),(.6,.01),(.6,-.01)]
CASES=['train_box','train_sine','train_bends','train_bezier','validation_trapezoid','validation_zigzag','dense']
SOURCES=['recovery_training_v6.py','strong_policy_v4.py','train_strong_fit_v4.py','physical_teacher_v5.py',
         'frozen_policy.py','runtime_physical_time_v5.py','runtime_resolution_fixed_v4e.py',
         'terminal_completion_v4d.py','runtime_no_target_override_v4b.py','replay_no_target_override_v4b.py',
         'runtime_bootstrap.py','runtime_configuration.py','corridor_v3.py','execution_settings.py',
         'evaluate.py','metrics.py']


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def arrays(path):
    with np.load(path) as data:
        return {key:data[key].copy() for key in data.files}


def register():
    if REG.exists() or OUT.exists():
        raise FileExistsError('Never overwrite a registered pilot')
    previous=read(PRIOR/'completion_validation.json')
    assert previous['status']=='passed'
    source_cases=read(ROOT/'results/historical_strong_v5/registration.json')['cases']
    rows=[next(x for x in source_cases if x['id']==name) for name in CASES]
    base=arrays(PRIOR/'base_teacher_records.npz')
    selected=[]
    for case in ('original','dense'):
        group=np.flatnonzero(base['case']==case)
        selected.extend(group[np.rint(np.linspace(0,len(group)-1,128)).astype(int)].tolist())
    assert len(set(selected))==256
    pilot=[selected[i] for i in (0,64,128,192)]
    ordered=pilot+[i for i in selected if i not in pilot]
    OUT.mkdir(parents=True)
    reg=dict(created_utc=utc(),version='fresh_recovery_wendland_v6_1',seed=SEED,
        identity='Complete fixed-budget fresh-student supervised fit conditional on a shared historical teacher; not independent historical RL training',
        teacher_sha256=sha(MODEL),teacher_trainable_weights_used_at_student_inference=False,
        teacher_source='Available frozen c16d45 model; no missing historical training assets required by this pilot',
        source_sha256={name:sha(ROOT/'src'/name) for name in SOURCES},
        runtime_snapshot_sha256=sha(ROOT/'runtime/source_manifest.json'),
        data_inputs={name:sha(PRIOR/name) for name in ('base_teacher_records.npz','branch_teacher_records.npz','completion_validation.json','registration.json')},
        selected_base_ids=selected,ordered_base_ids=ordered,pilot_base_ids=pilot,
        physical_perturbations=[dict(feed_factor=f,steering_offset_rad=t,preview_change_mm=0.) for f,t in PERTURBATIONS],
        branch_rule='Decode teacher action first, multiply physical feed and add physical steering, clip to the same action bounds, then invert the common decoder. Hold the same request for at most two 20 ms decisions; stop at actual termination.',
        branch_count=1024,maximum_20ms_commands=2048,collection_wall_limit_s=300.,
        failure_retention='Every branch/command outcome, exception and partial trace retained. Supervision uses only the recorded nonterminal in-corridor endpoint; terminal labels are retained but not trained.',
        label_rule='Same physical state and unchanged frozen teacher for both arms; fixed 3 mm steering uses wrap(beta(d)+theta-beta(3)); retain all clipping flags and direction errors.',
        feature_rule='Same14-64-64-3 network and96/192 compact-support Wendland heads. Rebuild all centers/scales/radii from registered training observations only; no historical feature center or trained parameter is copied.',
        feature_scale='Equal-weight mean and variance across original/dense x prior/recovery groups; coordinate standard deviation floored at0.03.',
        centers='Deterministic farthest-point selection in scaled14D observations, first center nearest the balanced mean; first96 for preview and all192 for joint.',
        radius='Twice the maximum training distance to its nearest selected center, separately at96/192. Output scales fixed0.01/0.08/0.4 as the existing architecture.',
        arms=['adaptive','fixed'],fixed_preview_mm=3.,same_allocated_architecture=True,
        paired_initialization='Identical seed101 fresh base tensors and zero corrections; fixed interface masks preview output.',
        sampler='128 uniform-with-replacement rows from each of four case x data-source groups per update; same sampler stream in both arms.',
        offline_iterations=24000,minibatch=512,learning_rate=2e-4,lr_multipliers={'8000':.3,'16000':.3},
        gradient_norm_limit=1.,loss_scales=[.015,.15,.3],boundary_margin=.02,
        fit_wall_limit_s_per_arm=300.,checkpoints=[0,8000,16000,24000],selection='Fixed last24000 checkpoint only for both arms.',
        new_ppo_updates=0,cases=rows,
        evaluation_runs=[dict(case=name,dt_s=.001,wall_s=120.) for name in CASES]+[dict(case='dense',dt_s=.0005,wall_s=120.)],
        evaluation_scope='Six prior training/validation programs and dense at both resolutions. The long original is not rerun in this bounded pilot; its old failure remains in existing evidence.',
        no_holdout_access=True,no_five_seed_expansion=True,no_figure8_or_manuscript_replacement=True,
        prospective_step='Only a successful pilot could motivate a separately registered training-path PPO stage; no such budget is authorized here.')
    write_json(REG,reg)
    shutil.copy2(REG,OUT/'registration.json')
    return verify_registration()


def verify_registration():
    reg=read(REG)
    assert sha(REG)==sha(OUT/'registration.json')
    assert all(sha(ROOT/'src'/name)==digest for name,digest in reg['source_sha256'].items())
    assert all(sha(PRIOR/name)==digest for name,digest in reg['data_inputs'].items())
    assert sha(MODEL)==reg['teacher_sha256']
    assert sha(ROOT/'runtime/source_manifest.json')==reg['runtime_snapshot_sha256']
    return reg


def collect():
    total_started,total_cpu=time.perf_counter(),time.process_time()
    reg=verify_registration()
    target=OUT/'collection';target.mkdir(exist_ok=False)
    import physical_teacher_v5 as ph
    from runtime_physical_time_v5 import install
    from frozen_policy import FrozenPolicy
    torch.set_num_threads(1)
    teacher=FrozenPolicy(MODEL)
    base=arrays(PRIOR/'base_teacher_records.npz')
    envs={}
    for case in ('original','dense'):
        trace,_,_,tol,_=ph.source_case(case)
        envs[case]=ph.environment(trace['reference'],tol,install)
    archive=ph.BranchArchive(target)
    starts=[];labels=[];branch_ends=[];pilot=[];ticks=0;completed=0;error=None
    preparation_wall=time.perf_counter()-total_started;preparation_cpu=time.process_time()-total_cpu
    started,cpu=time.perf_counter(),time.process_time()
    try:
        for order,base_id in enumerate(reg['ordered_base_ids']):
            case=str(base['case'][base_id]);env=envs[case]
            state={k:base[k][base_id].copy() if isinstance(base[k][base_id],np.ndarray) else base[k][base_id].item() for k in ph.SOURCE_KEYS}
            old=base['teacher_physical_action'][base_id].astype(float)
            for k,(factor,offset) in enumerate(PERTURBATIONS):
                branch_id=order*4+k
                desired=old.copy();desired[0]=np.clip(old[0]+offset,-np.pi/2+1e-10,np.pi/2-1e-10);desired[1]=np.clip(old[1]*factor,0.,100.)
                action=np.array([desired[0]/(np.pi/2),desired[1]/50.-1.,(desired[2]-6.4)/5.6])
                decoded=np.array([action[0]*np.pi/2,50*(1+action[1]),6.4+5.6*action[2]])
                assert np.max(abs(decoded-desired))<1e-12 and old[1]-desired[1]>1.
                current=copy.deepcopy(state);command_count=0
                for hold in range(2):
                    if time.perf_counter()-started>reg['collection_wall_limit_s']:
                        raise TimeoutError('Registered physical recovery collection wall cap')
                    before=copy.deepcopy(current);event_id=2*branch_id+hold
                    rows,positions,outcome=ph.branch(env,current,action)
                    starts.append(dict(**before,event_id=event_id))
                    labelled=None;label_error=None
                    if len(rows):
                        try:
                            labelled=ph.labels(env,teacher)
                            labelled.update(ph.capture_state(env))
                            labelled.update(event_id=event_id,branch_id=branch_id,base_id=base_id,hold=hold,case=case,
                                eligible_nonterminal_state=outcome['eligible_nonterminal_state'],terminal=outcome['done'])
                        except Exception as exc:
                            label_error=type(exc).__name__+': '+str(exc)
                    if labelled is not None:labels.append(labelled)
                    record=dict(**outcome,branch_id=event_id,physical_branch_id=branch_id,hold=hold,base_id=base_id,case=case,
                        start_state_index=len(starts)-1,requested_physical=desired.tolist(),base_teacher_physical=old.tolist(),
                        physical_feed_change_mm_s=float(desired[1]-old[1]),physical_steering_change_rad=float(desired[0]-old[0]),
                        feed_factor=factor,steering_offset_rad=offset,action_raw=action.tolist(),
                        label_record_index=len(labels)-1 if labelled is not None else -1,label_error=label_error)
                    archive.append(record,rows,positions)
                    ticks+=len(rows);command_count+=1
                    if base_id in reg['pilot_base_ids']:
                        assert outcome['kinematic_pass'] and label_error is None
                        pilot.append(dict(branch_id=branch_id,hold=hold,base_id=base_id,physical_request_error=float(np.max(abs(decoded-desired))),
                            feed_changed=bool(abs(desired[1]-old[1])>1.),outcome=outcome['executor_reason']))
                    if outcome['done'] or outcome['error'] or not outcome['corridor_inside'] or label_error:
                        break
                    current=ph.capture_state(env)
                completed+=1
                branch_ends.append(dict(branch_id=branch_id,base_id=base_id,case=case,commands=command_count,
                    done=outcome['done'],success=outcome['success'],error=outcome['error'],label_error=label_error))
            if order==3:
                write_json(target/'pilot_validation.json',dict(status='passed',physical_branches=16,commands=len(pilot),checks=pilot))
                print('RECOVERY_PILOT_PHYSICAL_REQUESTS_PASSED',len(pilot),flush=True)
            if (order+1)%32==0:
                print('RECOVERY_COLLECTION',order+1,completed,ticks,flush=True)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        archive.close();ph.pack_records(target/'start_states.npz',starts);ph.pack_records(target/'recovery_labels.npz',labels)
        write_json(target/'branch_outcomes.json',branch_ends)
    receipt=dict(created_utc=utc(),status='completed' if error is None and completed==1024 else 'incomplete',error=error,
        physical_branches=completed,command_events=len(starts),label_rows=len(labels),physics_ticks=ticks,
        eligible_label_rows=sum(bool(x['eligible_nonterminal_state']) for x in labels),
        fixed_label_clipped=sum(bool(x['fixed_steering_clipped']) for x in labels),
        collection_wall_s=time.perf_counter()-started,collection_cpu_s=time.process_time()-cpu,
        preparation_wall_s=preparation_wall,preparation_cpu_s=preparation_cpu,
        total_wall_s=time.perf_counter()-total_started,total_cpu_s=time.process_time()-total_cpu,
        registration_sha256=sha(REG),output_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file()},
        scope='Local physical recovery states only; branch survival does not certify future teacher rollout feasibility.')
    write_json(target/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def audit_collection():
    started,cpu=time.perf_counter(),time.process_time()
    reg=verify_registration();folder=OUT/'collection';receipt=read(folder/'receipt.json')
    assert receipt['status']=='completed'
    assert all(sha(folder/name)==digest for name,digest in receipt['output_sha256'].items())
    records=[json.loads(s) for s in (folder/'branch_records.jsonl').read_text('utf-8').splitlines()]
    starts=arrays(folder/'start_states.npz');labels=arrays(folder/'recovery_labels.npz')
    from runtime_bootstrap import setup
    setup()
    from geometry_spec import Corridor,GateTracker
    import physical_teacher_v5 as ph
    corridors={}
    for case in ('original','dense'):
        trace,_,_,tol,_=ph.source_case(case)
        corridors[case]=Corridor(trace['reference'],epsilon=tol,closed=True)
    batches={};total=0;failures=0;maximum=np.zeros(3)
    for record in records:
        name=record['trace_file']
        if name not in batches:batches[name]=arrays(folder/name)
        data=batches[name];i=record['start_state_index'];p0=record['position_offset'];r0=record['row_offset'];n=record['row_count']
        pos=data['positions'][p0:p0+n+1];rows=data['rows'][r0:r0+n]
        assert sha(folder/name)==record['trace_sha256']
        assert np.array_equal(pos[0],starts['position'][i]) and np.array_equal(pos[1:],rows[:,:2])
        tracker=GateTracker(corridors[record['case']],pos[0])
        for key in ('frontier','pending_negative','segment_count','start_error','inside'):
            setattr(tracker,key,starts[key][i].item())
        for point in pos[1:]:tracker.update(point)
        assert tracker.frontier==record['frontier_after'] and tracker.inside==record['corridor_inside']
        if n:
            v=np.diff(pos,axis=0)/.001;a=np.diff(np.vstack([starts['velocity'][i],v]),axis=0)/.001;j=np.diff(np.vstack([starts['acceleration'][i],a]),axis=0)/.001
            peaks=np.array([np.linalg.norm(value,axis=1).max() for value in (v,a,j)])
            assert np.array_equal(peaks,record['finite_command_vector_peaks']);maximum=np.maximum(maximum,peaks)
        expected=np.array(record['base_teacher_physical']);expected[0]=np.clip(expected[0]+record['steering_offset_rad'],-np.pi/2+1e-10,np.pi/2-1e-10);expected[1]*=record['feed_factor']
        assert np.array_equal(expected,record['requested_physical']) and record['physical_feed_change_mm_s']<0.
        if record['label_record_index']>=0:
            label=record['label_record_index'];assert labels['event_id'][label]==record['branch_id']
            assert np.array_equal(labels['position'][label],pos[-1])
        total+=n;failures+=bool(record['error'] or (record['done'] and not record['success']))
    theta=np.clip(labels['targets_three_raw'][:,0],-1+1e-10,1-1e-10)*np.pi/2
    expected=ph.wrap(labels['teacher_guide_angle']+theta-labels['fixed_guide_angle'])
    assert np.max(abs(expected-labels['fixed_theta_unclipped']))<1e-14
    assert np.array_equal(labels['targets_three_raw'][:,1],labels['targets_two_raw'][:,1])
    errors=abs(labels['fixed_direction_error_rad'][~labels['fixed_steering_clipped']])
    assert errors.max(initial=0)<1e-10
    assert total==receipt['physics_ticks'] and len({r['physical_branch_id'] for r in records})==1024
    result=dict(created_utc=utc(),status='passed',command_events=len(records),physics_ticks=total,
        every_command_interval_and_ordered_gate_recomputed=True,physical_feed_changed_events=len(records),
        failed_or_exception_events=failures,maximum_vector_peaks=maximum.tolist(),
        maximum_unclipped_fixed_direction_error_rad=float(errors.max(initial=0)),
        fixed_steering_clipped=int(labels['fixed_steering_clipped'].sum()),receipt_sha256=sha(folder/'receipt.json'))
    result.update(audit_wall_s=time.perf_counter()-started,audit_cpu_s=time.process_time()-cpu)
    write_json(folder/'completion_validation.json',result);print(json.dumps(result),flush=True)


def prepare_training():
    started,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(1)
    reg=verify_registration();target=OUT/'training_data';target.mkdir(exist_ok=False)
    assert read(OUT/'collection/completion_validation.json')['status']=='passed'
    old=arrays(PRIOR/'base_teacher_records.npz');near=arrays(PRIOR/'branch_teacher_records.npz');new=arrays(OUT/'collection/recovery_labels.npz')
    im=near['eligible_nonterminal_state'].astype(bool);nm=new['eligible_nonterminal_state'].astype(bool)
    x=np.vstack([old['observations'],near['observations'][im],new['observations'][nm]]).astype(np.float32)
    y3=np.vstack([old['targets_three'],near['targets_three'][im],new['targets_three'][nm]]).astype(np.float32)
    y2=np.vstack([old['targets_two'],near['targets_two'][im],new['targets_two'][nm]]).astype(np.float32)
    cases=np.r_[old['case'],near['case'][im],new['case'][nm]]
    source=np.r_[np.zeros(len(old['observations'])+int(im.sum()),int),np.ones(int(nm.sum()),int)]
    group=(cases=='dense').astype(int)*2+source
    groups=[np.flatnonzero(group==i) for i in range(4)];assert all(len(g)>0 for g in groups)
    weights=np.zeros(len(x))
    for indices in groups:weights[indices]=.25/len(indices)
    mean=(weights[:,None]*x).sum(0);scale=np.sqrt((weights[:,None]*(x-mean)**2).sum(0)).clip(.03)
    z=(x.astype(float)-mean)/scale
    index=int(np.argmin((z*z).sum(1)));chosen=[];distance=np.full(len(x),np.inf);radii={}
    for i in range(192):
        chosen.append(index);distance=np.minimum(distance,((z-z[index])**2).sum(1))
        if i in (95,191):radii[i+1]=max(.1,2.*float(np.sqrt(distance.max())))
        distance[np.array(chosen)]=-1.;index=int(np.argmax(distance))
    assets=dict(format='training_rebuilt_wendland_v6',tensors={
        'preview_centers':torch.tensor(x[chosen[:96]]),'joint_centers':torch.tensor(x[chosen]),
        'preview_scale':torch.tensor(scale,dtype=torch.float32),'joint_scale':torch.tensor(scale,dtype=torch.float32),
        'preview_radius':torch.tensor(radii[96],dtype=torch.float32),'joint_radius':torch.tensor(radii[192],dtype=torch.float32),
        'output_scale':torch.tensor([.01,.08,.4])},contains_historical_trainable_tensors=False,
        contains_historical_feature_tensors=False,teacher_identity=reg['teacher_sha256'])
    assets['tensor_sha256']=tensor_hash(assets['tensors'])
    torch.save(assets,target/'rebuilt_static_assets.pt')
    np.savez_compressed(target/'supervision.npz',observations=x,targets_three=y3,targets_two=y2,case=cases,group=group,
        selected_feature_center_rows=np.array(chosen),base_center=mean,base_scale=scale)
    policy=StrongPolicyV4(SEED,assets,'adaptive');coverage={}
    with torch.no_grad():
        features=[policy.actor.features(part) for part in torch.tensor(x).split(256)]
        for j,name in enumerate(['preview','joint']):
            f=torch.cat([a[j] for a in features]);m=f.max(1).values
            assert bool((m>.18).all())
            coverage[name]=dict(zero_support_rows=int((m==0).sum()),minimum_maximum_activation=float(m.min()),radius=radii[96 if j==0 else 192])
    receipt=dict(created_utc=utc(),rows=len(x),group_counts=[len(g) for g in groups],recovery_active_rows=int(nm.sum()),
        excluded_recovery_terminal_or_failed_rows=int((~nm).sum()),centers_from_training_only=True,center_rows=chosen,
        feature_coverage=coverage,feature_tensor_sha256=assets['tensor_sha256'],
        files={p.name:sha(p) for p in target.iterdir() if p.is_file()},
        registration_sha256=sha(REG),collection_validation_sha256=sha(OUT/'collection/completion_validation.json'))
    receipt.update(preparation_wall_s=time.perf_counter()-started,preparation_cpu_s=time.process_time()-cpu)
    write_json(target/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def fit(arm):
    total_started,total_cpu=time.perf_counter(),time.process_time()
    reg=verify_registration();target=OUT/('seed101_'+arm);target.mkdir(exist_ok=False)
    data_receipt=read(OUT/'training_data/receipt.json')
    assert all(sha(OUT/'training_data'/name)==digest for name,digest in data_receipt['files'].items())
    data=arrays(OUT/'training_data/supervision.npz');assets=torch.load(OUT/'training_data/rebuilt_static_assets.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    policy=StrongPolicyV4(SEED,assets,arm);x=torch.tensor(data['observations']);y=torch.tensor(data['targets_three' if arm=='adaptive' else 'targets_two'])
    groups=[np.flatnonzero(data['group']==i) for i in range(4)];rng=np.random.default_rng(SEED)
    with torch.no_grad():
        f=[policy.actor.features(part) for part in x.split(256)];preview=torch.cat([a[0] for a in f]);joint=torch.cat([a[1] for a in f])
    initial_hash=tensor_hash(dict(policy.named_parameters()));optimizer=torch.optim.Adam(policy.actor.parameters(),lr=reg['learning_rate'])
    scales=torch.tensor(reg['loss_scales'][:policy.action_dim]);history=[];sample_hash=hashlib.sha256()
    localreg=dict(parent_registration_sha256=sha(REG),arm=arm,seed=SEED,initial_trainable_tensor_sha256=initial_hash,
        training_data_sha256=sha(OUT/'training_data/supervision.npz'),static_asset_sha256=sha(OUT/'training_data/rebuilt_static_assets.pt'),
        allocation_parameters=sum(p.numel() for p in policy.parameters()),final_fixed_iterations=24000)
    write_json(target/'registration.json',localreg)
    def predict(ids):return policy.actor.from_features(x[ids],preview[ids],joint[ids])[...,:policy.action_dim]
    def save(path,step):
        torch.save(dict(format='strong_policy_v4',policy=policy.state_dict(),optimizer=optimizer.state_dict(),
            seed=SEED,arm=arm,fixed_preview_mm=3.,version=reg['version'],offline_iterations=step,new_ppo_updates=0,
            registration_sha256=sha(REG),trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),
            sampling_rng_state=rng.bit_generator.state),path)
    save(target/'initial.pt',0);step=0;error=None
    preparation_wall=time.perf_counter()-total_started;preparation_cpu=time.process_time()-total_cpu
    started,cpu=time.perf_counter(),time.process_time()
    try:
        with (target/'losses.jsonl').open('x',encoding='utf-8') as stream:
            for iteration in range(1,24001):
                if time.perf_counter()-started>reg['fit_wall_limit_s_per_arm']:raise TimeoutError('Fixed fitting wall cap')
                ids=torch.from_numpy(np.concatenate([g[rng.integers(len(g),size=128)] for g in groups]))
                loss=equivalent_loss(predict(ids),y[ids],scales);assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);assert torch.isfinite(gradient)
                optimizer.step();step=iteration;sample_hash.update(ids.numpy().tobytes())
                if step in (8000,16000):
                    for group in optimizer.param_groups:group['lr']*=.3
                stream.write(json.dumps(dict(iteration=step,loss=float(loss.detach())))+'\n')
                if step%2000==0:
                    with torch.no_grad():
                        output=torch.cat([predict(i) for i in torch.arange(len(x)).split(512)]);delta=output.clamp(-1,1)-y
                    record=dict(iteration=step,rmse=delta.square().mean(0).sqrt().tolist(),max_error=delta.abs().max(0).values.tolist(),wall_s=time.perf_counter()-started)
                    history.append(record);write_json(target/'fit_log.json',history);stream.flush();print('RECOVERY_FIT',arm,step,record['rmse'],flush=True)
                if step in (8000,16000,24000):save(target/f'fit_{step:05d}.pt',step)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    if error is None and step==24000:shutil.copy2(target/'fit_24000.pt',target/'final.pt')
    else:save(target/'interrupted.pt',step)
    final=target/('final.pt' if error is None else 'interrupted.pt');loaded,_=load_strong_policy(final)
    with torch.no_grad():assert torch.equal(loaded.mean(x[:512]),policy.mean(x[:512]))
    receipt=dict(created_utc=utc(),status='completed' if error is None and step==24000 else 'incomplete',error=error,
        arm=arm,seed=SEED,iterations=step,supervised_sample_presentations=step*512,model_sha256=sha(final),
        initial_trainable_tensor_sha256=initial_hash,final_fit=history[-1] if history else None,
        fitting_wall_s=time.perf_counter()-started,fitting_cpu_s=time.process_time()-cpu,
        preparation_wall_s=preparation_wall,preparation_cpu_s=preparation_cpu,
        total_wall_s=time.perf_counter()-total_started,total_cpu_s=time.process_time()-total_cpu,
        sampled_row_stream_sha256=sample_hash.hexdigest(),
        feature_tensor_sha256=assets['tensor_sha256'],same_shared_data_and_map_for_both_arms=True,
        historical_actor_weights_copied=False,historical_feature_maps_copied=False,new_ppo_updates=0)
    write_json(target/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def evaluate(arm):
    reg=verify_registration();folder=OUT/('seed101_'+arm);receipt=read(folder/'receipt.json')
    assert receipt['status']=='completed' and receipt['iterations']==24000
    from runtime_physical_time_v5 import install,VERSION
    from evaluate import rollout
    model=folder/'final.pt';target=folder/'evaluation';target.mkdir(exist_ok=False);results=[];torch.set_num_threads(1)
    write_json(target/'registration.json',dict(parent_registration_sha256=sha(REG),model_sha256=sha(model),arm=arm,
        runs=reg['evaluation_runs'],fixed_checkpoint=24000,runtime_version=VERSION))
    for run in reg['evaluation_runs']:
        case=next(x for x in reg['cases'] if x['id']==run['case']);reference=np.loadtxt(ROOT/case['reference'],delimiter=',',skiprows=1)
        assert sha(ROOT/case['reference'])==case['reference_sha256']
        install(dt=run['dt_s'],tolerance=case['evaluation_tolerance_mm']);policy,_=load_strong_policy(model)
        output=target/(run['case']+f"_dt{round(run['dt_s']*1e6)}us.npz")
        result=rollout(policy,reference,output,dt=run['dt_s'],tolerance=case['evaluation_tolerance_mm'],
            fixed_d=3. if arm=='fixed' else None,wall_s=run['wall_s'],runtime_version='v3')
        result.update(path=run['case'],arm=arm,seed=SEED,model_sha256=sha(model),
            reference_sha256=case['reference_sha256'],runtime_version=VERSION,scope='Registered bounded development pilot',
            nominal_tolerance_mm=case['nominal_tolerance_mm'],arc_chord_error_bound_mm=case['arc_chord_error_bound_mm'],
            method='fresh_recovery_wendland_v6_'+arm,registration_sha256=sha(REG))
        write_json(output.with_suffix('.json'),result);results.append(result)
    write_json(target/'summary.json',dict(created_utc=utc(),arm=arm,registered_runs=8,successful_runs=sum(x['success'] for x in results),
        results=[{k:x[k] for k in ('path','dt_s','success','complete_time_s','elapsed_prefix_s','reason')} for x in results],
        all_failures_retained=True,no_independent_final_test_claim=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['register','collect','audit','prepare','fit','evaluate']);parser.add_argument('--arm',choices=['adaptive','fixed'])
    args=parser.parse_args()
    if args.mode=='register':print(json.dumps(register(),indent=2))
    elif args.mode=='collect':collect()
    elif args.mode=='audit':audit_collection()
    elif args.mode=='prepare':prepare_training()
    elif args.mode=='fit':assert args.arm;fit(args.arm)
    else:assert args.arm;evaluate(args.arm)
