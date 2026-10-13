"""Preserve the previous live state before updating current research evidence."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil

OUT = Path(__file__).resolve().parent
R = OUT / 'all34_dominance_research'
BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
J = R / 'joint_bend_transit_fit_full'
F = R / 'fixed_actual_failure_recovery_full'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    history = R / 'state_history' / stamp
    history.mkdir(parents=True, exist_ok=False)
    paths = [BASE / '当前执行状态.json', OUT / 'live_research_summary.json', R / 'active_research_checkpoint.json']
    for path in paths:
        if path.exists():
            shutil.copy2(path, history / path.name)
    nominal = read(J / 'full_result.json')
    half = read(J / 'half_ms/full_result.json') if (J / 'half_ms/full_result.json').exists() else None
    probes = read(J / 'channel_interventions/receipt.json') if (J / 'channel_interventions/receipt.json').exists() else None
    pair_path = J / ('paired_acceptance_' + F.name) / 'assessment.json'
    pair = read(pair_path) if pair_path.exists() else None
    current = 'J3在1ms完整成功444.302s、全34角名义达标；比S2快883ms，比强Fixed快120ms。等待双分辨率数值验收及当前候选干预审阅，未冻结。'
    if half:
        current = f'J3在1ms为444.302s，0.5ms为{half["complete_time_s"]}s；双分辨率与机制最终审阅中，未冻结。'
    if pair and not pair['quantitative_core_passed']:
        current = 'J3双分辨率完整成功且34角均达标；对Fixed收益为120/100ms，细分辨率恰等100ms门槛，严格验收未通过。两项干预均几何失败且动力学通过；继续在J3真实上下文优化，未冻结。'
    active = {}
    if half is None:
        active['joint_half_ms'] = 28303
    if probes is None:
        active['joint_channel_interventions'] = 48989
    if (R/'joint_bend_transit_refit/registration.json').exists() and not (R/'joint_bend_transit_refit/receipt.json').exists():
        active['joint_transit_refit'] = 32056
    state = read(paths[0])
    state.update(updated_utc=stamp, current=current, goal_complete=False, method_frozen=False,
                 research_acceptance_passed=False, final_manuscript_accepted=False,
                 authoritative_acceptance=None, latest_revision_history=str(history),
                 current_candidate_complete_times_s=dict(Joint_1ms=nominal['complete_time_s'],
                     Joint_half_ms=half['complete_time_s'] if half else None,
                     Fixed_1ms=444.422, Fixed_half_ms=444.4205, S2_1ms=445.185, S2_half_ms=445.185),
                 current_candidate_bindings=dict(Joint=str(J/'full_result.json'),
                     Joint_half_ms=str(J/'half_ms/full_result.json'), Fixed=str(F/'full_result.json'),
                     Fixed_half_ms=str(F/'half_ms/full_result.json')),
                 current_candidate_all34_nominal_passed=True,
                 current_candidate_all34_passed=bool(pair and all(v['passed'] for v in pair['corner_comparisons'].values())),
                 current_candidate_resolution_verified=bool(pair and pair['quantitative_core_passed']),
                 current_candidate_nominal_mechanism=dict(status='candidate_specific_geometry_visual_and_interventions_reviewed' if probes else 'candidate_specific_geometry_and_visual_review_done; interventions_pending',
                     cruise99_s=nominal['motion']['cruise_ge99_s'],
                     maximum_corridor_error_mm=nominal['analytic']['maximum_evaluated_distance_mm'],
                     geometry_audit=str(J/'actual_turn_geometry/audit.json'),
                     visual_review=str(J/'mechanism_diagnostic/visual_review.json')),
                 active_validation_sessions=active,
                 next_active_research=['在J3真实上下文继续20分钟局部通过时间参数优化',
                     '有收益候选完成整圈、细分辨率和自身机制验证',
                     '核心条件通过后才冻结，从18页原稿对应源码修订20–25页论文'])
    write(paths[0], state)
    live = read(paths[1]) if paths[1].exists() else {}
    live.update(updated_utc=stamp, current=current, goal_complete=False, method_frozen=False,
                active_sessions=active, candidate=str(J), fixed=str(F), state_history=str(history),
                current_candidate_complete_times_s=state['current_candidate_complete_times_s'])
    write(paths[1], live)
    write(paths[2], dict(updated_utc=stamp, current=current, goal_complete=False, method_frozen=False,
        active_sessions=active, running_candidate=str(J), current_fixed=str(F),
        completed_sessions=[13130,34987,61723,29084,36806,68301],
        state_history=str(history), next_steps=state['next_active_research']))
    print(json.dumps(dict(history=str(history),active=active,half_result=bool(half),probes=bool(probes)),ensure_ascii=False))

if __name__ == '__main__':
    main()
