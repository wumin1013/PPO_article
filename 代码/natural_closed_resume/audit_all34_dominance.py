"""Reopen effective acceptance and audit existing full-lap evidence, no simulation."""
from pathlib import Path
import sys, json, csv, hashlib, shutil, datetime
import numpy as np
import torch

BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT = Path(__file__).resolve().parent
STAGE = OUT / 'all34_dominance_research'
sys.path[:0] = [str(BASE), str(OUT)]
from deep_joint_credit_policy import load_credit_policy
from arc_cruise_policy import build_policy

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    STAGE.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    states = [BASE / '当前执行状态.json', OUT / 'live_research_summary.json']
    if not (STAGE / 'reopened_state_receipt.json').exists():
        history = STAGE / 'state_history' / stamp
        history.mkdir(parents=True)
        for p in states:
            shutil.copy2(p, history / p.name)
            data = json.loads(p.read_text(encoding='utf-8'))
            data.update(updated_utc=stamp, current='新逐角验收标准下研究进行中：34角速度亏损均不高于S2和强Fixed；保留完整成功、三动作机制及两分辨率完整时间收益。',
                        goal_complete=False, method_frozen=False, research_acceptance_passed=False,
                        final_manuscript_accepted=False, previous_acceptance_scope='历史完整时间和稿件核验，不满足新增逐角支配标准',
                        current_research_directory=str(STAGE), latest_revision_history=str(history))
            data['current_acceptance'] = dict(all34_deficit='Each registered +/-10mm corner window <= both baselines; zero permits numerical tie',
                                             unchanged_full_success=True, unchanged_full_time_margin=True,
                                             learned_three_action_mechanism=True, corner_count=34)
            data['next_research_if_requested'] = []
            write(p, data)
        write(STAGE / 'reopened_state_receipt.json', dict(history=str(history), immutable_binding_unchanged=True,
             binding_sha256=sha(BASE / 'paper_evidence_review/verified_results/paper_result_binding.json')))
    evidence = BASE / 'paper_evidence_review/verified_results'
    table = list(csv.DictReader((evidence / 'all34_corner_statistics.csv').open(encoding='utf-8-sig')))
    print('METHODS', sorted({r['method'] for r in table}), flush=True)
    by = {(r['method'], r['id']): r for r in table}
    z = np.load(OUT / 'joint_actual_context_feedback_training/exact_continuation/full_trace.npz')
    records = []
    for r in table:
        if r['method'] != 'Joint':
            continue
        s, f = by['S2', r['id']], by['Fixed', r['id']]
        lo, hi = int(r['first_index']) // 20, int(r['last_index']) // 20 + 1
        aa, xx = z['actions'][lo:hi], z['observations'][lo:hi]
        dirs = xx[:, 42:62].reshape(-1, 10, 2)
        ang = np.unwrap(np.arctan2(dirs[:, :, 1], dirs[:, :, 0]), axis=1)
        bend = np.max(np.abs(ang - ang[:, :1]) * xx[:, 62:72], axis=1)
        rec = dict(id=r['id'], angle=float(r['included_angle_deg']),
                   joint=float(r['speed_deficit_mm']), s2=float(s['speed_deficit_mm']), fixed=float(f['speed_deficit_mm']),
                   minimum_speed=float(r['minimum_speed_mm_s']), feed_min=float(aa[:, 1].min()),
                   feed_max=float(aa[:, 1].max()), preview_min=float(aa[:, 2].min()), preview_max=float(aa[:, 2].max()),
                   bend_max=float(bend.max()), curvature_max=float(r['peak_curvature_mm_inv']))
        rec.update(gap_s2=rec['joint']-rec['s2'], gap_fixed=rec['joint']-rec['fixed'])
        records.append(rec)
    tolerance = 1e-8  # floating-point report only; final tolerance requires resolution evidence
    counts = {m: dict(worse=sum(r['gap_'+m] > tolerance for r in records),
                       better=sum(r['gap_'+m] < -tolerance for r in records),
                       tied=sum(abs(r['gap_'+m]) <= tolerance for r in records)) for m in ('s2', 'fixed')}
    x = z['observations']
    constants = np.flatnonzero(np.ptp(x, axis=0) < 1e-9).tolist()
    identical = [(i,j) for i in range(72) for j in range(i+1,72)
                 if np.ptp(x[:,i])>1e-9 and np.max(np.abs(x[:,i]-x[:,j]))<1e-9]
    selection = json.loads((OUT / 'joint_actual_context_feedback_training/exact_continuation/selected_policy.json').read_text(encoding='utf-8'))
    policy = build_policy(load_credit_policy(torch.load(selection['model'], map_location='cpu', weights_only=False)), selection['spec'])
    layers = []
    while policy is not None:
        layers.append(dict(type=type(policy).__name__, module=type(policy).__module__,
                           has_memory=hasattr(policy,'memory'), fields=list(vars(policy).keys())))
        policy = getattr(policy, 'base', None)
    result = dict(goal_complete=False, counts=counts, records=records, current_route_observation_audit=dict(
        samples=len(x), dimension=72, constant_indices=constants, identical_nonconstant_pairs=identical,
        min=x.min(0).tolist(), max=x.max(0).tolist(), std=x.std(0).tolist(),
        interpretation='Route-specific redundancy only. No dimension optimality or retraining comparison established.'), policy_layers=layers,
        input_csv_sha256=sha(evidence/'all34_corner_statistics.csv'), ppo_updates=0, physics_ticks=0)
    write(STAGE / 'existing_evidence_audit.json', result)
    print('COUNTS', counts, 'CONSTANTS', constants, 'IDENTICAL', identical, flush=True)
    print('LAYERS', [(r['type'], r['has_memory']) for r in layers], flush=True)
    for r in records:
        print('{id} angle={angle:.3f} deficit J/S/F={joint:.6f}/{s2:.6f}/{fixed:.6f} gap={gap_s2:+.6f}/{gap_fixed:+.6f} feed={feed_min:.2f}..{feed_max:.2f} preview={preview_min:.2f}..{preview_max:.2f}'.format(**r), flush=True)

if __name__ == '__main__':
    main()
