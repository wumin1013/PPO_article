"""Saved-trajectory identity and unsmoothed XY checks for explicitly unadopted training actors."""
import argparse
import copy
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from saved_memory_action_identity import SavedMemoryActionIdentity
from gate_band_state import GateBandRuntime
from audit_saved_lateral_motion import lateral_motion
from multi_corner_raw_metrics import augment as augment_multi
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True)
    parser.add_argument('--arm')
    args=parser.parse_args();torch.set_num_threads(1)
    stage=BASE/args.stage
    if args.arm:
        model_file=stage/(args.arm+'_training_policy.pt')
        result=next(r for r in read(stage/'training_arm_results.json')['arms'] if r['arm']==args.arm)
        prefix=args.arm
    else:
        model_file=stage/'shadow_training_policy.pt'
        result=read(stage/'shadow_training_result.json');prefix='shadow'
    model=torch.load(model_file,map_location='cpu',weights_only=False)
    assert model['training_branch_only'] and not model['accepted']
    support=read(stage/'support_audit.json')
    paths=support.get('evaluation_paths',support['selected']['paths'])
    records=[r for r in result['records'] if r['path'] in paths]
    replay=collect_data(model,model['parent'],records,include_latents=True)
    held=[]
    for r in records:
        with np.load(r['trace']) as z:held.append(z['actions'].copy())
    replay['held_actions']=np.vstack(held)
    proof=SavedMemoryActionIdentity(model,replay,GateBandRuntime.decode_action)
    assert all(proof.unchanged(r['path'],model['coefficients']) for r in records)
    old={r['path']:r for r in read(stage/'baseline.json')['baseline']['records']}
    rows=[]
    for r in records:
        if 'shape' not in r:
            fresh=copy.deepcopy(r);fresh.pop('multi_corner_windows',None)
            augment_multi(fresh)
            assert fresh['multi_corner_windows']==r['multi_corner_windows']
            rows.append(dict(path=r['path'],multicorner_raw_windows_recomputed_exactly=True,
                window_count=len(r['multi_corner_windows']),physical=r['audit'],
                cruise99_s=r['motion']['cruise_ge99_s'],complete_time_s=r['complete_time_s']))
            continue
        a=lateral_motion(r['trace']);b=lateral_motion(old[r['path']]['trace'])
        assert abs(a['any_reverse_lateral_total_mm']-r['shape']['reverse_lateral_total_mm'])<1e-12
        assert abs(a['significant_reverse_lateral_total_mm']-r['shape']['reverse_lateral_significant_mm'])<1e-12
        assert a['significant_reverse_lateral_duration_ms']==round(1000*r['shape']['reverse_lateral_significant_duration_s'])
        rows.append(dict(path=r['path'],before=b,after=a,physical=r['audit'],
            first_turn_inside=r['target_motion']['first_turn_inside'],exit_mm=r['shape']['out_edge_outward_mm'],
            reverse_deg=r['shape']['full_physical_reverse_deg'],cruise99_s=r['motion']['cruise_ge99_s'],
            complete_time_s=r['complete_time_s']))
    np.savez_compressed(stage/(prefix+'_actor_replay.npz'),**replay)
    write_json(stage/(prefix+'_raw_identity_audit.json'),dict(model_sha256=sha(model_file),paths=paths,
        decisions=len(replay['decisions']),actual_actor_and_saved_decoded_actions_bitwise=True,
        lateral_metrics_recomputed_exactly=True,records=rows,new_physical_runs=0,
        adopted=False,ideal_tangency_accepted=False,final_guard_pass=result['original_all_guard_pass']))
    print('TRAINING_SHADOW_IDENTITY',prefix,len(replay['decisions']),sha(model_file),'not_adopted',flush=True)


if __name__=='__main__':main()
