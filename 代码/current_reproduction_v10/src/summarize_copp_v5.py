"""Regenerate COPP summaries from every registered complete development run."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['copp_frozen_v4e_complement', 'copp_v5_preflight', 'copp_v5b_preflight',
        'copp_v5b_complete_development', 'copp_v5c_reserve_validation',
        'copp_v5c_complete_development', 'copp_v5c_observed_additional',
        'copp_v5d_complete_development']


def read(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    all_versions = []
    lines = ['# COPP完整开发矩阵审计', '',
        '所有路径均为已经观察过的开发程序；本记录不构成独立最终测试。固定第三轮为候选，前两轮、求解失败及所有已生成完整轨迹均保留。', '']
    for directory in RUNS:
        folder = ROOT/'results'/directory
        registration = read(folder/'registration.json')
        summary_path = folder/'summary.json'
        if not summary_path.exists():
            lines += ['## '+directory, '', '运行尚未完整结束。', '']
            continue
        records = read(summary_path)
        version = dict(directory=directory, version=registration['version'], cases=[],
            source_hash_match=all(sha(ROOT/'src'/name) == value for name, value in registration['source_sha256'].items()),
            generated_command_runs=0, successful_command_runs=0, fixed_third_complete_cases=0,
            total_attempted_cases=len(records), trace_hash_matches=True)
        lines += ['## '+directory, '', '| 路径 | 状态 | 第三轮1 ms (s) | 第三轮0.5 ms (s) | 站点 | 总墙钟 (s) |', '|---|---|---:|---:|---:|---:|']
        for record in records:
            iterations = record.get('iterations', [])
            final = next((row for row in iterations if row['iteration'] == 3), None)
            all_runs = [run for row in iterations for run in row['runs']]
            version['generated_command_runs'] += len(all_runs)
            version['successful_command_runs'] += sum(run['metrics']['success'] for run in all_runs)
            for run in all_runs:
                version['trace_hash_matches'] &= sha(folder/run['trace']) == run['trace_sha256']
            row = dict(path_id=record['path_id'], planning_status=record['status'], error=record.get('error'),
                generated_runs=len(all_runs), successful_runs=sum(r['metrics']['success'] for r in all_runs),
                final_iteration=3, completed_fixed_candidate=bool(final and len(final['runs']) == 2 and all(r['metrics']['success'] for r in final['runs'])),
                stations=record.get('planner', {}).get('station_count'), wall_s=record['total_wall_s'],
                planning_clocks=record.get('planner', {}).get('clocks'),
                solve_wall_s=sum(r['solver_wall_s'] for r in iterations),
                optimizer_rerun=not any(r.get('solver_rerun') is False for r in iterations),
                nominal_tolerance_mm=record.get('nominal_tolerance_mm', 1.25),
                reference_tolerance_mm=record['tolerance_mm'],
                third_iteration_runs=[dict(dt_s=r['dt_s'], metrics=r['metrics'], execution=r['execution'],
                    trace=str(folder/r['trace']), trace_sha256=r['trace_sha256']) for r in final['runs']] if final else [])
            version['fixed_third_complete_cases'] += row['completed_fixed_candidate']
            version['cases'].append(row)
            times = {r['dt_s']: r['metrics']['complete_time_s'] for r in final['runs']} if final else {}
            values = [f'{times[x]:.4f}' if times.get(x) is not None else '—' for x in [.001, .0005]]
            state = '通过' if row['completed_fixed_candidate'] else '失败：'+str(row['error'] or '共同验收未通过')
            lines.append(f"| {row['path_id']} | {state} | {values[0]} | {values[1]} | {row['stations'] or '—'} | {row['wall_s']:.3f} |")
        lines += ['', f"固定第三轮双分辨率通过：{version['fixed_third_complete_cases']}/{len(records)}程序；已生成轨迹通过：{version['successful_command_runs']}/{version['generated_command_runs']}。", '']
        all_versions.append(version)
    output = ROOT/'audit/copp_development_summary_v5'
    output.with_suffix('.json').write_text(json.dumps(dict(scope='Development only', versions=all_versions), indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(output.with_suffix('.md'))
    for version in all_versions:
        print(version['directory'], version['fixed_third_complete_cases'], '/', version['total_attempted_cases'],
              'traces', version['trace_hash_matches'], 'sources', version['source_hash_match'])


if __name__ == '__main__':
    main()
