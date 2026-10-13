"""Build a minimal earlier/middle state basis around the protected research policy."""
import time
import argparse
import numpy as np
import torch
from joint_teacher_state_policy import features,state_pair,INDICES
from consolidate_joint_teacher_policy import collect_data
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-stage',default='canonical_gate_band_decoded_joint_step_learning_20260920')
    parser.add_argument('--output-stem',default='earlier_geometry')
    args=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter()
    target=BASE/'canonical_gate_band_direct_stable_joint_learning_20260920'
    parent_file=BASE/args.parent_stage/'research_candidate.pt'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    template=torch.load(BASE/'canonical_gate_band_midturn_supported_joint_learning_20260920/initial_policy.pt',map_location='cpu',weights_only=False)
    baseline=read(BASE/args.parent_stage/'results.json')['candidates'][0]['records']
    changed=[14,104,105,204,205,304,305]
    current=collect_data(parent,parent,[r for r in baseline if r['path'] in changed],deadline=started+55)
    assert not np.any(current['labels'])
    with np.load(BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920/replayed_states.npz') as z:
        old={k:z[k].copy() for k in ('observations','innovations','path_ids','decisions')}
    keep=~np.isin(old['path_ids'],changed)
    replay={k:np.concatenate([old[k][keep],current[k]]) for k in old}
    raw,_=state_pair(replay['observations'],replay['innovations'])
    source_kernels=[6,3,7];training_decisions=[5,15,16]
    mean=np.asarray(template['per_center_mean'])[source_kernels]
    std=np.asarray(template['per_center_std'])[source_kernels]
    centers=[]
    for k,d in enumerate(training_decisions):
        ix=np.flatnonzero((replay['path_ids']==104)&(replay['decisions']==d)).item()
        centers.append((raw[ix]-mean[k])/std[k])
    weights=np.ones((3,raw.shape[1]));stable=set([11,12,15]+list(range(22,72)))
    dynamic=[i for i,k in enumerate(INDICES) if k not in stable]+list(range(len(INDICES),weights.shape[1]))
    weights[1:,dynamic]=0.
    md=dict(architecture='observable_memory_joint_radial',parent=parent,centers=np.asarray(centers),
        mean=template['mean'],std=template['std'],per_center_mean=mean,per_center_std=std,
        radii=np.asarray(template['radii'])[source_kernels],per_center_distance_weights=weights,
        memory_decays=[.5,.8],coefficients=np.zeros((9,3)),accepted=False,
        learning_method='earlier_geometry_supported_complete_joint_parameter_learning',
        action_interval_ms=20,physical_tick_ms=1)
    f=features(md,replay['observations'],replay['innovations']);records=[]
    for k in range(3):
        active=np.any(f[:,k]!=0.,axis=1);paths=sorted(np.unique(replay['path_ids'][active]).tolist())
        row=dict(kernel=k,training_selector_ms=training_decisions[k]*20,paths=paths,
            activations=[dict(path=int(p),decision_ms=int(d)*20,feature=float(abs(v).max()))
                for p,d,v in zip(replay['path_ids'][active],replay['decisions'][active],f[active,k])])
        records.append(row);print('EARLY_GEOMETRY_SUPPORT',row,flush=True)
    affected=sorted(set(p for r in records for p in r['paths']))
    assert all('shape' in r for r in baseline if r['path'] in affected)
    # Parent + exactly zero outer weights is bitwise unchanged, including float32 output.
    selected=[r for r in baseline if r['path'] in affected]
    active_replay=collect_data(md,parent,selected,include_latents=True,deadline=started+55)
    actions=[]
    for r in selected:
        with np.load(r['trace']) as z:actions.append(z['actions'].copy())
    active_replay['held_actions']=np.vstack(actions)
    assert len(active_replay['held_actions'])==len(active_replay['path_ids'])
    torch.save(md,target/(args.output_stem+'_initial_policy.pt'))
    np.savez_compressed(target/(args.output_stem+'_training_states.npz'),**replay)
    np.savez_compressed(target/(args.output_stem+'_affected_replay.npz'),**active_replay)
    write_json(target/(args.output_stem+'_support_audit.json'),dict(records=records,affected_paths=affected,
        source_model_sha256=sha(parent_file),training_paths=88,training_decisions=len(replay['path_ids']),
        zero_outer_policy_decoded_actions_bitwise=True,new_physical_runs=0,wall_s=time.perf_counter()-started,
        interpretation='Three centers selected from observable training states. The earlier '
        'center preserves full-state distance; later centers retain reference geometry '
        'while allowing motion changes caused by earlier actions. All27 weights across '
        'three actions and causal memories are learnable. No runtime time/path inputs. '
        'Preparation only; this initial policy has not been optimized or adopted.'))


if __name__=='__main__':main()
