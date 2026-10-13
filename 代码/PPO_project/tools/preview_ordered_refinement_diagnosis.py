"""Read-only witness-density sensitivity on 18 saved pilots plus one diagnostic."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'PPO_project'))
from src.utils.revision_metrics import ordered_projection

BASE=ROOT/'论文项目/revision_20260908'
OUT=BASE/'preview_direct_gain/refinement_diagnostic'
VARIANTS=('fixed_short','fixed_mid','fixed_center','fixed_long','direct','direct_no_anchor')
FAMILIES=('mixed_lengths','dense_corners','line_curve')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={}
    items=[(variant,family,BASE/f'preview_direct_gain/pilot/{variant}_seed101/evaluation_validation_nominal/{family}_validation')
           for variant in VARIANTS for family in FAMILIES]
    items.append(('constant_action_D3','mixed_lengths',BASE/'preview_v2/guidance_gain_probe_extension_D3/blend_0.015_D3_mixed_lengths_validation'))
    rows=[]
    for variant,family,stem in items:
        paths={suffix:Path(str(stem)+suffix) for suffix in ['.json','.npz']}
        reference_path=ROOT/f'PPO_project/data/preview_v2/paths/{family}_validation.csv'
        for path in [*paths.values(),reference_path]:sources[path.relative_to(ROOT).as_posix()]=sha(path)
        original=json.loads(paths['.json'].read_text(encoding='utf-8'))
        z=np.load(paths['.npz']);columns=dict(zip(z['columns'].tolist(),z['data'].T))
        xy=np.c_[columns['x_mm'],columns['y_mm']]
        reference=np.loadtxt(reference_path,delimiter=',',skiprows=1)
        coarse=ordered_projection(reference,xy,tolerance_mm=1.25)
        assert np.isclose(coarse['ordered_progress'][-1],original['ordered_progress'],atol=1e-12)
        fine=ordered_projection(reference,xy,tolerance_mm=1.25,witness_spacing_mm=.01)
        complete=bool(fine['ordered_coverage_complete'] and fine['start_error_mm']<=1.25 and fine['endpoint_error_mm']<=1.25)
        physical=bool(original['geometry_feasible'] and all(c['satisfied'] is not False for c in original['constraints'].values()))
        row=dict(variant=variant,path_id=family+'_validation',split='validation',
            source_json=paths['.json'].relative_to(ROOT).as_posix(),source_trace=paths['.npz'].relative_to(ROOT).as_posix(),
            original_success=original['success'],original_path_completed=original['path_completed'],
            original_termination_reason=original['termination_reason'],original_failure_reasons=original['failure_reasons'],
            original_ordered_progress=original['ordered_progress'],coarse_covered_witnesses=coarse['covered_witness_count'],
            coarse_witness_count=coarse['witness_count'],fine_covered_witnesses=fine['covered_witness_count'],fine_witness_count=fine['witness_count'],
            diagnostic_fine_coverage_complete=fine['ordered_coverage_complete'],diagnostic_fine_path_completed=complete,
            diagnostic_fine_ordered_progress=float(fine['ordered_progress'][-1]),
            unchanged_physical_constraints_and_geometry_feasible=physical,
            diagnostic_saved_trace_fine_completion_and_physical_feasibility=bool(complete and physical),
            original_contour_error_max_mm=original['contour_error_max_mm'],original_constraints=original['constraints'],
            original_elapsed_time_s=original['elapsed_time_s'],labels_mutated=False)
        rows.append(row)
        print(variant,row['path_id'],'old',row['original_path_completed'],'fine',complete,'physical',physical,flush=True)
    main_rows=rows[:18]
    by_method=[dict(variant=v,n=3,original_completions=sum(r['original_path_completed'] for r in main_rows if r['variant']==v),
        fine_diagnostic_completions=sum(r['diagnostic_fine_path_completed'] for r in main_rows if r['variant']==v),
        fine_diagnostic_completion_and_feasibility=sum(r['diagnostic_saved_trace_fine_completion_and_physical_feasibility'] for r in main_rows if r['variant']==v)) for v in VARIANTS]
    summary=dict(pilot_trace_count=len(main_rows),original_completions=sum(r['original_path_completed'] for r in main_rows),
        fine_diagnostic_completions=sum(r['diagnostic_fine_path_completed'] for r in main_rows),
        fine_diagnostic_completion_and_feasibility=sum(r['diagnostic_saved_trace_fine_completion_and_physical_feasibility'] for r in main_rows),by_method=by_method)
    for file in ['PPO_project/src/utils/revision_metrics.py','PPO_project/src/utils/ordered_coverage.py']:
        sources[file]=sha(ROOT/file)
    payload=dict(scope='Read-only certificate-density sensitivity; no policy, training or test evaluation; original labels preserved',
        command='python PPO_project/tools/preview_ordered_refinement_diagnosis.py',generator_sha256=sha(Path(__file__)),
        one_sided_tolerance_mm=1.25,original_spacing_mm=.25,original_radius_mm=1.125,
        diagnostic_spacing_mm=.01,diagnostic_radius_mm=1.245,
        interpretation='Finer witnesses reduce the sufficient coverage certificate margin at the same physical tolerance. A finer-certificate pass does not relabel the frozen experiment or represent a newly trained policy. All unchanged physical feasibility flags remain; no failed stopping time establishes efficiency.',
        source_hashes=sources,summary=summary,rows=rows)
    (OUT/'summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fields=[key for key in rows[0] if key!='original_constraints']
    with (OUT/'rows.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# 顺序覆盖见证点加密：只读敏感性诊断','','同一批已保存的 18 条 direct-gain pilot 验证轨迹，另加常动作 D3 mixed 轨迹。保持单侧物理允差 1.25 mm、原始轨迹和全部运动约束判据不变，仅离线将见证点间隔从 0.25 mm 加密至 0.01 mm；相应保守匹配半径由 1.125 mm 变为 1.245 mm。原始结果文件和失败标签均未改写。','',
        '| 方法 | 原完成数/3 | 细证书完成数/3 | 细证书完成且原物理约束满足/3 |','|---|---:|---:|---:|']
    for item in by_method:lines.append(f"| {item['variant']} | {item['original_completions']} | {item['fine_diagnostic_completions']} | {item['fine_diagnostic_completion_and_feasibility']} |")
    extra=rows[-1]
    lines+=['',f"18 条 pilot 原完成 {summary['original_completions']}，细证书诊断完成 {summary['fine_diagnostic_completions']}，其中原物理约束及全局轮廓允差同时满足 {summary['fine_diagnostic_completion_and_feasibility']}。",
        f"常动作 D3 mixed：原完成 {extra['original_path_completed']}，细证书诊断完成 {extra['diagnostic_fine_path_completed']}，原物理可行性 {extra['unchanged_physical_constraints_and_geometry_feasible']}。",'',
        '粗证书失败、细证书通过表示同一物理允差下离散覆盖充分条件的保守程度不同，不能据此把原冻结实验悄然改为成功。该读数也不是新奖励或重训方法的运行结果；原训练过程已受粗证书进度与终止信号影响。若采用加密方案，应建立新版本，对固定和可学习前瞻共享修正、重训并预先冻结评价，保持未见测试路径独立。','']
    (OUT/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(OUT/'summary.json')


if __name__=='__main__':main()
