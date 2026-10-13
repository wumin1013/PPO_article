"""Saved-data audit; no new simulation, training, or model mutation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
R5 = ROOT/'direct14_speed_deficit_ppo_seed922302_round5'
R6 = ROOT/'direct14_all34_context_ppo_seed922302_round6'
R7 = ROOT/'direct14_balanced_failure_ppo_seed922302_round7'
HIST = ROOT.parent/'all34_dominance_research/joint_transit_retained_seed922222_full'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def batch_audit():
    files=[p/'batch_0001.npz' for p in (R6,R7)]
    if not all(p.exists() for p in files):
        return None
    with np.load(files[0]) as old, np.load(files[1]) as new:
        result={key:bool(np.array_equal(old[key],new[key])) for key in old.files}
        changed=np.flatnonzero(old['reward']!=new['reward'])
        result=dict(array_equal=result,changed_reward_indices=changed.tolist(),
                    source_sha256={str(p):sha(p) for p in files},
                    new_training_or_simulation=False)
    write(R7/'first_batch_matched_audit.json',result)
    return result


def trajectory_audit():
    current=read(R5/'eval_0024_original_metrics.json')
    historical=read(HIST/'full_metrics.json')
    historic_corners={x['id']:x for x in historical['corners']}
    path=R5/'eval_0024_original.npz'
    with np.load(path) as data:
        rows=data['rows'];actions=data['executed_actions_1ms']
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    records=[]
    for c in current['corners']:
        h=historic_corners[c['id']]
        sl=slice(c['first_index'],c['last_index']+1)
        sv=speed[sl];request=actions[sl];rr=rows[sl]
        records.append(dict(id=c['id'],angle_deg=c['included_angle_deg'],
            deficit_mm=c['speed_deficit_mm'],historical_deficit_mm=h['speed_deficit_mm'],
            deficit_change_mm=c['speed_deficit_mm']-h['speed_deficit_mm'],
            requested_feed_min=float(request[:,1].min()),
            requested_feed_mean=float(request[:,1].mean()),
            requested_feed_below99_fraction=float((request[:,1]<99).mean()),
            requested_feed_deficit_mm=float(np.maximum(100-request[:,1],0).sum()*.001),
            speed_deficit_while_request100_mm=float(np.maximum(100-sv,0)[request[:,1]>=100-1e-7].sum()*.001),
            preview_range_mm=[float(request[:,2].min()),float(request[:,2].max())],
            minimum_speed=float(sv.min()),maximum_acceleration=float(np.linalg.norm(rr[:,4:6],axis=1).max()),
            outgoing_overshoot_mm=c['outgoing_overshoot_mm']))
    result=dict(source_sha256={str(p):sha(p) for p in [path,R5/'eval_0024_original_metrics.json',HIST/'full_metrics.json']},
        new_training_or_simulation=False,corners=records,
        total_deficit_change_mm=sum(x['deficit_change_mm'] for x in records),
        counts_by_angle={str(a):dict(count=sum(x['angle_deg']==a for x in records),
            nonworse=sum(x['angle_deg']==a and x['deficit_change_mm']<=1e-7 for x in records),
            deficit_change_mm=sum(x['deficit_change_mm'] for x in records if x['angle_deg']==a)) for a in sorted(set(x['angle_deg'] for x in records))})
    write(ROOT/'retained_round5_corner_request_audit.json',result)
    return result


def update_retained_record():
    status_path=ROOT/'compact14_verified_status_20260922.json'
    status=read(status_path)
    checkpoint=R5/'checkpoint_0024.pt'
    if status['compact_candidate']['checkpoint']==str(checkpoint):
        return
    nominal=read(R5/'eval_0024_dense.json')
    original=read(R5/'eval_0024_original.json')
    half_path=ROOT/(R5.name+'_half_ms')/'result.json'
    half=read(half_path)
    assert nominal['success'] and original['success'] and half['success'] and read(R5/'result.json')['adopted']
    joint=next(x for x in read(R5/'joint_acceptance_audit.json')['candidates'] if x['update']==24)
    now=datetime.now(timezone.utc)
    backup=ROOT/'state_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True,exist_ok=False)
    (backup/status_path.name).write_bytes(status_path.read_bytes())
    write(backup/'receipt.json',dict(original=str(status_path),backup=str(backup/status_path.name),sha256=sha(status_path)))
    status.setdefault('compact_candidate_history',[]).append(status['compact_candidate'])
    status['compact_candidate']=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),
        observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,
        inherited_controller_at_execution=False,minimum_sufficient_dimension_proven=False,
        training_kind='Offline initialization and two explicit offline refits, with16+24inherited actual PPOupdates',
        ppo_updates_in_ancestry=40,latest_stage_ppo_updates=24,
        original_time_s=original['complete_time_s'],dense_time_1ms_s=nominal['complete_time_s'],
        dense_time_half_ms_s=half['complete_time_s'],
        original_all34_vs_historical_not_worse_count=joint['original34_comparisons']['historical_main']['corners_not_worse'],
        adopted_as_main=False,source_evaluations=[str(R5/'eval_0024_original.json'),str(R5/'eval_0024_dense.json'),str(half_path)])
    status.update(updated_utc=now.isoformat(),goal_complete=False,method_frozen=False,
        next_step='Round7 matched failure-credit PPO is active; final joint goal remains unmet.',
        latest_ppo_round_report=str(R5/'joint_acceptance_audit.json'),latest_ppo_round_adopted=True,
        latest_compact_candidate_audit=str(R5/'joint_acceptance_audit.json'),
        latest_state_backup_receipt=str(backup/'receipt.json'))
    status['alternative_candidate']=dict(checkpoint=str(ROOT/'direct14_saturation_aware_fit_seed922302_round4/fit_12000.pt'),
        original_time_s=444.353,dense_time_1ms_s=15.114,original_all34_vs_historical_not_worse_count=19,
        role='Different-metric research alternative; never combine its corner count with retained candidate timing')
    write(status_path,status)


if __name__=='__main__':
    batch=batch_audit();trajectory=trajectory_audit();update_retained_record()
    print(json.dumps(dict(first_batch=batch,counts_by_angle=trajectory['counts_by_angle'],
        worst_corners=sorted(trajectory['corners'],key=lambda x:-x['deficit_change_mm'])[:6]),ensure_ascii=False))
