"""Regenerate complete v5d evidence, including reused failures and new costs."""
from pathlib import Path
import json
import numpy as np
from copp_path_adapter_v4 import sha, write

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'results/copp_v5d_complete_development'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    registration = read(FOLDER/'registration.json')
    assert all(sha(ROOT/'src'/name) == value for name,value in registration['source_sha256'].items())
    for row in registration['inputs']:
        path = Path(row['folder'])
        assert sha(path/'registration.json') == row['registration_sha256']
        assert sha(path/'summary.json') == row['summary_sha256']
    inputs = {row['id']:row for row in registration['cases']}
    records = read(FOLDER/'summary.json')
    cases = []
    for record in records:
        name = record['path_id']
        source = inputs[name]
        assert record == read(FOLDER/(name+'.json'))
        assert sha(source['source_record']) == source['source_record_sha256']
        for profile in source['profiles']:
            assert sha(profile['path']) == profile['sha256']
        row = dict(path_id=name, final_candidate_runs=[], all_candidates=[],
            newly_sampled_command_runs=0, reused_candidate_zero_runs=0,
            trace_verification_wall_s=0., new_sampling_audit_save_wall_s=0.,
            new_geometry_reconstruction_wall_s=record.get('new_geometry_reconstruction_wall_s',0.),
            recorded_case_wall_s=record['total_wall_s'],
            inherited_three_socp_solve_wall_s=sum(item['solver_wall_s'] for item in record['iterations']),
            inherited_geometry_grid_seed_clocks=record['planner'].get('clocks'),
            repaired_profiles=0)
        for item in record['iterations']:
            previous = None
            expected_candidate = 0
            for candidate in item['projection_candidates']:
                assert candidate['candidate_index'] == expected_candidate
                expected_candidate += 1
                assert expected_candidate <= registration['maximum_candidates_including_initial']
                assert {run['dt_s'] for run in candidate['runs']} == {.001,.0005}
                assert all(run['execution']['time_dilation'] == candidate['time_dilation'] for run in candidate['runs'])
                if previous is not None:
                    ratio = np.max([run['metrics']['vector_limit_ratios'] for run in previous['runs']],axis=0)
                    expected = (1+registration['correction_reserve'])*max(1.,ratio[0],np.sqrt(ratio[1]),np.cbrt(ratio[2]))
                    assert candidate['correction_factor'] == expected
                    assert candidate['time_dilation'] == previous['time_dilation']*expected
                row['trace_verification_wall_s'] += candidate.get('trace_and_contract_verification_wall_s',0.)
                for run in candidate['runs']:
                    path = FOLDER/run['trace']
                    assert sha(path) == run['trace_sha256']
                    assert run['metrics']['tolerance_mm'] == record['tolerance_mm']
                    assert run['metrics']['limits'] == [100.,2000.,62500.]
                    row['newly_sampled_command_runs'] += bool(run['new_sampling_performed'])
                    row['reused_candidate_zero_runs'] += bool(run['trace_reused'])
                    row['new_sampling_audit_save_wall_s'] += run['sample_audit_save_wall_s']
                    row['all_candidates'].append(dict(iteration=item['iteration'],candidate_index=candidate['candidate_index'],
                        dt_s=run['dt_s'],success=run['metrics']['success'],reason=run['metrics']['reason'],
                        complete_time_s=run['metrics']['complete_time_s'],time_dilation=candidate['time_dilation'],
                        vector_peaks=run['metrics']['vector_peaks'],vector_limit_ratios=run['metrics']['vector_limit_ratios'],
                        reused=run['trace_reused'],trace=str(path),trace_sha256=run['trace_sha256']))
                previous = candidate
            assert item['runs'] == item['projection_candidates'][-1]['runs']
            assert item['selected_projection_candidate'] == item['projection_candidates'][-1]['candidate_index']
            row['repaired_profiles'] += item['selected_projection_candidate'] > 0
            row.setdefault('selected_runs',[]).extend(item['runs'])
            if item['iteration'] == 3:
                row['final_candidate_runs'] = item['runs']
        row['fixed_third_joint_success'] = len(row['final_candidate_runs']) == 2 and all(run['metrics']['success'] for run in row['final_candidate_runs'])
        row['all_selected_runs_pass'] = len(row['selected_runs']) == 6 and all(run['metrics']['success'] for run in row['selected_runs'])
        cases.append(row)
    candidates = [run for row in cases for run in row['all_candidates']]
    selected = [run for row in cases for run in row['selected_runs']]
    result = dict(scope='All eighteen development programs, three SOCP iterates and two command periods; no final holdout claim',
        version=registration['version'],source_input_profile_trace_hashes_match=True,
        profile_count=sum(len(record['iterations']) for record in records),
        fixed_third_successful_programs=sum(row['fixed_third_joint_success'] for row in cases),
        programs=len(cases),selected_command_runs=len(selected),selected_successful_runs=sum(run['metrics']['success'] for run in selected),
        all_attempts_including_reused_failures=len(candidates),
        all_successful_attempts=sum(run['success'] for run in candidates),
        prior_failed_attempts_retained=sum(not run['success'] and run['reused'] for run in candidates),
        newly_sampled_command_runs=sum(row['newly_sampled_command_runs'] for row in cases),
        reused_candidate_zero_runs=sum(row['reused_candidate_zero_runs'] for row in cases),
        repaired_profiles=sum(row['repaired_profiles'] for row in cases),
        maximum_candidates_used=max(run['candidate_index']+1 for run in candidates),
        costs=dict(new_trace_verification_wall_s=sum(row['trace_verification_wall_s'] for row in cases),
            new_geometry_reconstruction_wall_s=sum(row['new_geometry_reconstruction_wall_s'] for row in cases),
            new_sampling_audit_save_wall_s=sum(row['new_sampling_audit_save_wall_s'] for row in cases),
            recorded_case_wall_s=sum(row['recorded_case_wall_s'] for row in cases),
            inherited_three_socp_solve_wall_s=sum(row['inherited_three_socp_solve_wall_s'] for row in cases),
            interpretation='New case wall time excludes initial import/source-registration work. Reused solver, sampling and acceptance costs are not new computation. No hard-real-time claim.'),
        cases=cases)
    output = ROOT/'audit/copp_v5d_complete_development_audit'
    write(output.with_suffix('.json'),result)
    lines=['# COPP v5d完整开发矩阵审计','',
        f"18条完整开发程序、54个SOCP剖面、双时间分辨率共{result['selected_successful_runs']}/{result['selected_command_runs']}个最终执行通过；固定第三轮为主候选，{result['fixed_third_successful_programs']}/18程序通过。",'',
        f"候选0共复用{result['reused_candidate_zero_runs']}条已核验轨迹，包含{result['prior_failed_attempts_retained']}条保留的旧失败；仅{result['repaired_profiles']}个剖面需要统一规则追加候选，共{result['newly_sampled_command_runs']}条新轨迹。全部{len(candidates)}个尝试记录保留，未挑选其他SOCP轮次。",'',
        '| 程序 | 第三轮1 ms (s) | 第三轮0.5 ms (s) | 追加投影的剖面数 |',
        '|---|---:|---:|---:|']
    for row in cases:
        runs={run['dt_s']:run for run in row['final_candidate_runs']}
        times=[runs[dt]['metrics']['complete_time_s'] for dt in (.001,.0005)]
        values=[f'{value:.4f}' if value is not None else '失败' for value in times]
        lines.append(f"| {row['path_id']} | {values[0]} | {values[1]} | {row['repaired_profiles']} |")
    lines += ['', '修复仅统一拉伸全时轴，原几何、单侧公差及圆弧误差预算、初末状态、有序门和完整命令验收不变。速度、加速度及jerk有限差分重新计算；固定dt重采样改变相位，故比例公式不替代完整验收。', '',
        'v5c通过者完全复用原数值；v5d只补充必要的可行性修复，不额外放慢已通过的竞争基线。新计算成本和继承成本分别保存在JSON。此处仍为开发证据，不是独立保留测试、连续伺服动力学保证或硬实时验证。', '']
    output.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key != 'cases'},indent=2))


if __name__ == '__main__':
    main()
