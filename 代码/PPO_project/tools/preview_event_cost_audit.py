"""Audit saved event-study training costs without executing any policy."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import preview_event_study as study


def main():
    destination = study.OUT.parent / 'training_cost_audit.json'
    if destination.exists():
        raise FileExistsError('Preserve the existing cost snapshot')
    rows = []
    for stage in ('smoke', 'pilot', 'formal'):
        for path in sorted((study.OUT / stage).glob('*/training_summary.json')):
            row = study.read(path)
            run = path.parent
            actual = study.sha(run / 'checkpoint.pt')
            if actual != row['checkpoint_sha256']:
                raise ValueError(f'Model/summary mismatch: {run}')
            rows.append(dict(
                stage=stage, run=run.relative_to(study.ROOT).as_posix(),
                variant=row['variant'], seed=row['seed'], steps=row['steps'],
                inherited_pretraining_steps=0,
                reported_training_wall_s=row['training_wall_s'],
                wall_scope=('Last saved invocation; resumed smoke duration is incomplete'
                            if stage == 'smoke' else 'Complete training invocation'),
                summary_sha256=study.sha(path), checkpoint_sha256=actual,
                config_sha256=study.sha(run / 'config.yaml'),
                source_sha256=study.read(run / 'provenance.json')['source_sha256']))
    # Reconstruction repeated learning transitions, so its compute is additional.
    # The restored 512-step copy is not another trained run and is not counted twice.
    recovery = study.OUT / 'smoke/hold50_seed901/recovery_512'
    proof = recovery / 'recovery_report.json'
    recovered = recovery / 'isolated/论文项目/revision_20260909_event_preview/study/smoke/hold50_seed901'
    row = study.read(recovered / 'training_summary.json')
    if row['steps'] != 1024 or not study.read(proof)['continued_1024_full_state_exact_except_wall_time']:
        raise ValueError('Reconstruction evidence is incomplete')
    rows.append(dict(stage='reconstruction', run=recovered.relative_to(study.ROOT).as_posix(),
        variant=row['variant'], seed=row['seed'], steps=row['steps'], inherited_pretraining_steps=0,
        reported_training_wall_s=row['training_wall_s'],
        wall_scope='Last reconstruction invocation; initial reconstruction duration excluded',
        summary_sha256=study.sha(recovered / 'training_summary.json'), proof_sha256=study.sha(proof),
        checkpoint_sha256=study.sha(recovered / 'checkpoint.pt')))
    groups = defaultdict(lambda: dict(runs=0, steps=0, reported_training_wall_s_sum=0.))
    for row in rows:
        group = groups[row['stage']]
        group['runs'] += 1
        group['steps'] += row['steps']
        group['reported_training_wall_s_sum'] += row['reported_training_wall_s']
    if groups['formal']['runs'] != 20 or groups['formal']['steps'] != 20 * study.FORMAL_STEPS:
        raise ValueError('Formal budget incomplete')
    if groups['pilot']['runs'] != 19 or groups['pilot']['steps'] != 19 * study.PILOT_STEPS:
        raise ValueError('Pilot budget incomplete')
    prior = study.ROOT / '论文项目/revision_20260908/training_cost_audit.json'
    earlier = study.read(prior)
    batches = []
    for stage in ('pilot', 'formal'):
        for path in sorted(study.OUT.glob(f'batch_{stage}_*.json')):
            item = study.read(path)
            batches.append(dict(path=path.relative_to(study.ROOT).as_posix(), sha256=study.sha(path),
                stage=stage, workers=item['workers'], wall_s=item['wall_s'],
                completed=len(item['completed']), errors=item['errors']))
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        command='python ' + ' '.join(sys.argv), script_sha256=study.sha(__file__),
        scope='Recorded learning transitions; evaluation, replay-only transitions and unit-test toy updates excluded',
        groups=dict(groups), runs=rows, batch_wall_times=batches,
        current_recorded_learning_steps=sum(r['steps'] for r in rows),
        prior_audit_path=prior.relative_to(study.ROOT).as_posix(), prior_audit_sha256=study.sha(prior),
        prior_recorded_learning_steps=earlier['actual_executed_learning_steps_in_saved_records'],
        cumulative_recorded_learning_steps=earlier['actual_executed_learning_steps_in_saved_records'] + sum(r['steps'] for r in rows),
        caveats=['Process duration sums are not elapsed batch time.',
                 'Smoke and reconstruction summaries do not recover all earlier invocation wall time.',
                 'Historical training lacking run metadata remains unquantified.',
                 'Auxiliary NNC transfer has zero new training; original NNC training is in the prior audit.',
                 'Budget completion does not establish policy convergence.'])
    study.write(destination, result)
    lines = ['# 本轮训练成本', '', '仅按保存的实际训练摘要计步；同一模型恢复前后的累计步数只计一次。', '',
             '|阶段|记录数|训练步数|记录的进程秒数之和|', '|---|---:|---:|---:|']
    for stage, group in groups.items():
        lines.append(f"|{stage}|{group['runs']}|{group['steps']:,}|{group['reported_training_wall_s_sum']:.3f}|")
    lines += ['', f"本轮有记录的训练合计 {result['current_recorded_learning_steps']:,} 步。"
              f"加上前轮已归档审计 {result['prior_recorded_learning_steps']:,} 步，累计 {result['cumulative_recorded_learning_steps']:,} 步。",
              '', '重建512步后续至1024步实际重复了1024步训练，单独计入；重建模型的文件复制不重复计步。',
              'smoke和重建表中的秒数仅为最后调用的已保存值，缺失的历史历时不能填零。正式/pilot完整调用按实际日志记录；并行进程时间之和不等于批次历时。',
              'NNC迁移没有新增训练；原NNC训练已在前轮审计中。历史权重缺少元数据的训练成本仍未知。',
              '', '逐运行来源、模型/config/源码哈希、批次实际历时和排除项见 training_cost_audit.json。']
    destination.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('groups', 'current_recorded_learning_steps', 'cumulative_recorded_learning_steps')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
