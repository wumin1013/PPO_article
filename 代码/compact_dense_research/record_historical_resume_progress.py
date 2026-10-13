"""Preserve and update effective research state from actual stage receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
STATES = [BASE / '当前执行状态.json', OUT / 'live_research_summary.json',
          OUT / 'all34_dominance_research/active_research_checkpoint.json', ROOT / 'current_progress.json']
NAMES = ['historical_best_dense_initial', 'historical_best_dense_local_arc_support',
         'historical_dense_cluster_preview_probe', 'historical_dense_cluster_feed_probe',
         'historical_dense_cluster_joint_feedback_probe', 'historical_residual_ppo_seed922301_pilot',
         'historical_dense_global_triplet_probe']


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verified-active-session', type=int)
    parser.add_argument('--verified-audit-session', type=int)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    backup = ROOT / 'state_history' / now.strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True, exist_ok=False)
    log = ROOT / 'research_rounds_20260922.md'
    backups = []
    for path in [*STATES, log]:
        raw = path.read_bytes()
        (backup / path.name).write_bytes(raw)
        backups.append(dict(original=str(path), path=str(backup / path.name), sha256=hashlib.sha256(raw).hexdigest()))
    records = []
    for name in NAMES:
        folder = ROOT / name
        record = dict(directory=str(folder))
        for key in ['registration', 'receipt', 'result', 'results', 'interface_validation', 'live']:
            value = read(folder / f'{key}.json')
            if value is not None:
                record[key] = value
        records.append(record)
    pilot = next(record for record in records if Path(record['directory']).name == 'historical_residual_ppo_seed922301_pilot')
    active = []
    if 'receipt' not in pilot and args.verified_active_session is not None:
        active = [pilot['directory']]
    if 'receipt' not in pilot and not active:
        raise RuntimeError('No terminal receipt or verified live handle; inspect the existing process before updating status')
    current = ('历史最佳方法已核验源码并在密集案例复现；局部规则探针均未完整通过，现正进行继承该方法的三动作残差PPO训练。'
               if active else '历史最佳方法已核验并完成密集案例复现、局部规则探针和三动作残差PPO先导；实际结果见绑定记录，尚未通过研究验收。')
    sparse_stage = ROOT / 'historical_sparse_memory_full_recorded_replay'
    sparse_result = read(sparse_stage / 'result.json')
    offline = (dict(directory=str(sparse_stage), exec_session=args.verified_audit_session,
                    verification_utc=now.isoformat(), kind='Recorded trajectory replay; no new physics')
               if sparse_result is None and args.verified_audit_session is not None else None)
    for path in STATES:
        state = read(path)
        state.update(updated_utc=now.isoformat(), current=current,
            active_stages=active, goal_complete=False, method_frozen=False,
            research_acceptance_passed=False, final_manuscript_accepted=False, final_pdf=None,
            latest_historical_dense_stages=records,
            latest_historical_source_audit=str(ROOT / 'historical_dense_source_and_arc_audit.json'),
            verified_active_exec_session=args.verified_active_session if active else None,
            active_offline_audit=offline,
            sparse_memory_recorded_replay=sparse_result,
            sparse_memory_coefficient_audit=str(ROOT / 'historical_memory_coefficient_audit.json'),
            baseline_effort_policy='Primary effort goes to main-method feasibility and efficiency. Fixed-preview comparison receives a reasonable predeclared budget, not unlimited optimization to become best. Core ablation remains genuinely retrained two-output steering/feed.',
            latest_state_backup_receipt=str(backup / 'receipt.json'),
            next_steps=[
                'Inspect residual PPO complete evaluations and failure causes; do not repeat ineffective local coefficient scans.',
                'Finish sparse-memory action-identity verification, then validate closed-loop behavior before adopting the smaller stored state.',
                'Keep the historical original-case result and all34 evidence intact; validate any adopted modification on both cases.',
                'Use bounded fair two-output training and selection; do not optimize a fixed-preview method indefinitely.',
                'Resolve three-action mechanism,5percent dense-case gains,independent repetition and resolution checks before method freeze.',
                'Only then finalize experiments on the unique18-page source and preserve its first3 figures.'])
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    report = []
    for record in records:
        name = Path(record['directory']).name
        if 'results' in record:
            rr = record['results']
            report.append(f"- {name}: {len(rr)}个参数候选，完整成功{sum(bool(x['success']) for x in rr)}，最多{max(x['gates_passed'] for x in rr)} /84门。")
        elif 'result' in record:
            rr = record['result']
            if 'success' in rr:
                report.append(f"- {name}: success={rr['success']}，完整时间={rr['complete_time_s']}，{rr['geometry']['gates_passed']}/84门；前缀不能作为完整效率。")
            else:
                report.append(f"- {name}: {rr.get('ppo_updates_this_round')}次真实PPO更新，新增{rr.get('new_decisions')}决策；best={rr.get('best')}。")
        else:
            report.append(f"- {name}: 本次以exec会话{args.verified_active_session}核验仍在运行。")
    with log.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write('\n\n## '+now.isoformat()+'：历史最佳方法密集案例研究\n\n')
        stream.write('\n'.join(report)+'\n\n')
        stream.write('历史候选60个源码及快照、模型和选择参数哈希均一致。发现密集离散转角会误触发历史圆弧进给补偿；关闭该局部补偿后通过首个锐角，但还未完整成功。前瞻下限、单独几何进给限制和共同误差系数探针均保留，不继续扩大同类无收益搜索。\n\n')
        stream.write('PPO先导仅训练新增三动作修正；17维是该修正网络输入，整个控制器仍保留历史72维和474个因果记忆标量，不宣称降维或独立从零训练。零修正轨迹与动作逐元素完全一致，状态恢复后的下一步及观测也逐元素一致；训练复用既有PPO采样、更新与检查点循环。\n\n')
        stream.write('用户补充：不以让固定前瞻达到最好为目标。资源优先用于主方法；对照给予合理、预先限定的公平机会。上述新增复现、参数探针与PPO均属于主方法研究，S2和固定前瞻未改动。原论文及前三图未改动；未冻结、未完成。\n')
        stream.write('独立记忆结构审计发现474个历史记忆值仅154个对应非零输出系数；新稀疏实现保留原求和布局以核验输出一致性。原始前驱特征缓存237值，新实现按活跃衰减槽存154值；这些与72维观测分开计数。当前全记录回放结果：'+json.dumps(sparse_result if sparse_result is not None else offline,ensure_ascii=False)+'。它不替代新闭环运行。\n')
    receipt = dict(updated_utc=now.isoformat(), backups=backups, active_stages=active,
        goal_complete=False, method_frozen=False, source_or_manuscript_edit=False)
    (backup / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for path in STATES:
        raw = path.read_bytes()
        assert not raw.startswith(b'\xef\xbb\xbf')
        json.loads(raw.decode('utf-8'))
    print(json.dumps(dict(active_stages=active, backup_receipt=str(backup / 'receipt.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
