"""Read-only attribution of saved position-difference noise, never a relabeler.

The registered evaluator and its tolerance remain authoritative. Alternatives
isolate timestamp spacing and position/heading quantization at the same t[k].
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'PPO_project'))
from src.utils.revision_metrics import independent_kinematics

DEFAULT_TRACE = ROOT / '论文项目/revision_20260908/preview_direct_explore/formal/selected_fixed_seed11/evaluation_regression_nominal/square.npz'


def backward(values, dt):
    values = np.asarray(values)
    result = np.full(values.shape, np.nan, dtype=values.dtype)
    result[1:] = np.diff(values, axis=0) / dt
    return result


def angular_from_heading(heading, dt, omega0=0., alpha0=0.):
    heading = np.unwrap(np.asarray(heading))
    omega = backward(heading, dt)
    omega[0] = omega0
    alpha = backward(omega, dt)
    alpha[0] = alpha0
    return {'heading': heading, 'omega': omega, 'alpha': alpha, 'jerk': backward(alpha, dt)}


def angular_from_positions(positions, dt, heading0, omega0=0., alpha0=0.):
    p = np.asarray(positions)
    displacement = np.diff(p, axis=0)
    heading = np.empty(len(p), dtype=p.dtype)
    heading[0] = heading0
    for k, step in enumerate(displacement, start=1):
        heading[k] = np.arctan2(step[1], step[0]) if np.linalg.norm(step) > 1e-12 else heading[k-1]
    return angular_from_heading(heading, dt, omega0, alpha0)


def stats(values, limit, rtol):
    finite = np.asarray(values)[np.isfinite(values)]
    return {'peak_abs': float(np.max(np.abs(finite))),
            'violating_samples': int(np.count_nonzero(np.abs(finite) > limit * (1. + rtol))),
            'checked_samples': int(len(finite))}


def finite_peak(values):
    return float(np.nanmax(np.abs(values)))


def diagnose(trace_path, output):
    trace_path, output = Path(trace_path).resolve(), Path(output).resolve()
    metrics_path = trace_path.with_suffix('.json')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (trace_path, metrics_path)}
    archive = np.load(trace_path, allow_pickle=False)
    values = {str(k): archive['data'][:, i] for i, k in enumerate(archive['columns'])}
    t = values['timestamp_s']
    p = np.column_stack((values['x_mm'], values['y_mm']))
    theta, feed = values['heading_rad'], values['v_mm_s']
    omega, alpha, internal_jerk = values['omega_rad_s'], values['alpha_rad_s2'], values['angular_jerk_rad_s3']
    sample_dt = float(np.nanmedian(values['reward_components__dt_s']))
    initial_velocity = feed[0] * np.array([math.cos(theta[0]), math.sin(theta[0])])
    initial_acceleration = values['a_mm_s2'][0] * np.array([math.cos(theta[0]), math.sin(theta[0])])
    exact = independent_kinematics(p, t, initial_velocity=initial_velocity,
        initial_acceleration=initial_acceleration, initial_heading=theta[0],
        initial_angular_velocity=omega[0], initial_angular_acceleration=alpha[0])
    methods = {
        'registered_position_float64_saved_timestamps': {
            'heading': exact['heading_rad'], 'omega': exact['angular_velocity_rad_s'],
            'alpha': exact['angular_acceleration_rad_s2'], 'jerk': exact['angular_jerk_rad_s3']},
        'position_float64_uniform_dt': angular_from_positions(p, sample_dt, theta[0], omega[0], alpha[0]),
        'position_longdouble_uniform_dt': angular_from_positions(p.astype(np.longdouble), np.longdouble(sample_dt), theta[0], omega[0], alpha[0]),
        'saved_heading_float64_saved_timestamps': angular_from_heading(theta, np.diff(t), omega[0], alpha[0]),
        'saved_heading_float64_uniform_dt': angular_from_heading(theta, sample_dt, omega[0], alpha[0]),
    }
    # Recreate only each one-step addition, using saved executed speed/heading;
    # the original positions, not new accumulated positions, anchor each step.
    intended = np.array([feed[k] * sample_dt * np.array([math.cos(theta[k]), math.sin(theta[k])]) for k in range(1, len(t))])
    replayed = p[:-1] + intended
    displacement_residual = np.diff(p, axis=0) - intended
    limit = float(metrics['constraints']['angular_jerk_rad_s3']['limit'])
    rtol = float(metrics['constraint_relative_tolerance'])
    registered = methods['registered_position_float64_saved_timestamps']
    registered_stats = stats(registered['jerk'], limit, rtol)
    if registered_stats['violating_samples'] != metrics['constraints']['angular_jerk_rad_s3']['violating_samples']:
        raise ValueError('Saved registered violation count did not reproduce')
    if not np.isclose(registered_stats['peak_abs'], metrics['constraints']['angular_jerk_rad_s3']['peak'], rtol=0, atol=1e-10):
        raise ValueError('Saved registered peak did not reproduce')
    bad = np.flatnonzero(np.abs(registered['jerk']) > limit*(1+rtol))
    # Since consecutive nonzero coordinates are close and same-sign, subtraction
    # is frequently exact (Sterbenz); position accumulation has already rounded.
    a, b = p[:-1], p[1:]
    nonzero = (a != 0) & (b != 0) & (np.signbit(a) == np.signbit(b))
    ratio = np.divide(np.abs(a), np.abs(b), out=np.zeros_like(a), where=nonzero)
    sterbenz = nonzero & (ratio >= .5) & (ratio <= 2)
    heading_error = registered['heading'] - np.unwrap(theta)
    aligned = {}
    for name, data in methods.items():
        aligned[name] = {**stats(data['jerk'], limit, rtol),
            'max_abs_omega_minus_internal_rad_s': finite_peak(data['omega']-omega),
            'max_abs_alpha_minus_internal_rad_s2': finite_peak(data['alpha']-alpha),
            'max_abs_jerk_minus_internal_rad_s3': finite_peak(data['jerk']-internal_jerk)}
    # A bounded heading perturbation e becomes at most 8 max|e| / dt^3 in
    # the interior third backward difference at a constant sample interval.
    noise_jerk = angular_from_heading(heading_error, sample_dt, 0., 0.)['jerk']
    source = Path(__file__).resolve()
    summary = {
        'schema_version': 'read-only-roundoff-attribution-v1',
        'trace': str(trace_path.relative_to(ROOT)),
        'authority': 'Original JSON success/failure labels and relative tolerance are unchanged; alternatives are attribution diagnostics only.',
        'registered_result': {k: metrics[k] for k in ('success','failure_reasons','termination_reason','elapsed_time_s','ordered_progress')},
        'limit_rad_s3': limit, 'registered_relative_tolerance': rtol,
        'sample_count': len(t), 'sample_dt_s': sample_dt,
        'max_abs_saved_dt_minus_nominal_s': finite_peak(np.diff(t)-sample_dt),
        'float64_mantissa_bits': np.finfo(np.float64).nmant,
        'longdouble_mantissa_bits': np.finfo(np.longdouble).nmant,
        'longdouble_is_extended_on_this_platform': bool(np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant),
        'internal_saved_jerk': stats(internal_jerk[1:], limit, rtol),
        'aligned_methods': aligned,
        'one_step_replay_max_abs_position_residual_mm': finite_peak(replayed-p[1:]),
        'position_difference_minus_intended_step_max_abs_mm': finite_peak(displacement_residual),
        'max_abs_coordinate_mm': finite_peak(p),
        'median_command_step_mm': float(np.median(np.linalg.norm(np.diff(p, axis=0),axis=1))),
        'sterbenz_exact_subtraction_fraction_of_coordinate_pairs': float(np.mean(sterbenz)),
        'max_abs_position_derived_heading_minus_saved_heading_rad': finite_peak(heading_error),
        'third_difference_of_heading_discrepancy_peak_rad_s3': finite_peak(noise_jerk),
        'interior_heading_noise_third_difference_bound_rad_s3': 8*finite_peak(heading_error)/sample_dt**3,
        'uniform_position_minus_uniform_heading_jerk_peak_rad_s3': finite_peak(methods['position_float64_uniform_dt']['jerk']-methods['saved_heading_float64_uniform_dt']['jerk']),
        'violation_indices_at_original_threshold': bad.tolist(),
        'alignment': 'Every array index k represents t[k]. Position displacement uses executed v[k] and heading[k]; three backward differences of unwrapped heading are compared to saved internal jerk[k]. Initial omega/alpha are supplied from the trace.',
        'interpretation_boundary': 'Exact one-step replay and tiny displacement/heading residuals identify finite-precision coordinate accumulation and subsequent differentiation as an explanation for edge excess. Higher precision arithmetic on already rounded coordinates cannot recover missing position bits. This does not establish continuous motion or Cartesian-axis constraint guarantees, and cannot remove the independent timeout/incomplete-path failure.',
    }
    output.mkdir(parents=True,exist_ok=True)
    raw = {'timestamp_s':t,'position_mm':p,'saved_heading_rad':theta,'internal_omega_rad_s':omega,
           'internal_alpha_rad_s2':alpha,'internal_jerk_rad_s3':internal_jerk,
           'intended_displacement_mm':intended,'displacement_rounding_residual_mm':displacement_residual,
           'position_minus_saved_heading_rad':heading_error,'violation_indices':bad}
    raw.update({f'{name}__{key}':value for name,data in methods.items() for key,value in data.items()})
    np.savez_compressed(output/'aligned_raw_arrays.npz',**raw)
    rows = [{'index':int(k),'timestamp_s':float(t[k]),'x_mm':float(p[k,0]),'y_mm':float(p[k,1]),
             'internal_jerk_rad_s3':float(internal_jerk[k]),
             'position_heading_discrepancy_rad':float(heading_error[k]),
             **{f'{name}_jerk_rad_s3':float(data['jerk'][k]) for name,data in methods.items()}} for k in bad]
    if rows:
        with (output/'original_violations.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table = '\n'.join(f"| {name} | {item['peak_abs']:.12f} | {item['violating_samples']} |" for name,item in aligned.items())
    note = f'''# 保存轨迹的有限精度差分诊断

本诊断只读取 `{trace_path.relative_to(ROOT).as_posix()}` 及对应 JSON。原评价仍为失败，原约束相对容差 `{rtol:g}`、原角捷度违规 `{len(bad)}` 次及全部失败原因均保留。替代重算用于解释数值来源，不用于重标或选择更有利统计口径。

轨迹每个位置 p[k] 使用当前步速度 v[k] 与航向 theta[k] 积分；所有后向差分对齐到 t[k]。内部角速度、角加速度、角捷度也比较同一 k。已提供保存的初始角速度和角加速度，未补造首样本捷度。

| 重算途径 | 角捷度绝对峰值 (rad/s³) | 原阈值下违规数 |
| --- | ---: | ---: |
| 内部保存角捷度 | {summary['internal_saved_jerk']['peak_abs']:.12f} | {summary['internal_saved_jerk']['violating_samples']} |
{table}

原位置重算峰值和违规数与原 JSON 完全复现。使用均匀 dt 后仍有 {aligned['position_float64_uniform_dt']['violating_samples']} 次违规，因此保存时间戳的浮点差值不是主要解释。当前平台 `np.longdouble` 与 float64 同为 {np.finfo(np.float64).nmant} 位显式尾数，longdouble 对照没有提供更高精度；将已舍入的位置转换类型也不能恢复已丢失位。

依据保存 v[k]、theta[k] 和 dt，逐步重放原位置加法，与保存位置的最大残差为 {summary['one_step_replay_max_abs_position_residual_mm']:.3g} mm。保存位置差分与其理论步位移的最大分量偏差为 {summary['position_difference_minus_intended_step_max_abs_mm']:.6g} mm；坐标量级约 {summary['max_abs_coordinate_mm']:.6g} mm，而位移中位数为 {summary['median_command_step_mm']:.6g} mm。{100*summary['sterbenz_exact_subtraction_fraction_of_coordinate_pairs']:.3f}% 的相邻坐标对满足 Sterbenz 条件，其减法本身可精确；主要问题是之前位置累加已经发生舍入，随后由短位移恢复航向，再做高阶差分。

位置恢复航向与保存航向之差最大为 {summary['max_abs_position_derived_heading_minus_saved_heading_rad']:.6g} rad。此差异的三阶后向差分峰值为 {summary['third_difference_of_heading_discrepancy_peak_rad_s3']:.9g} rad/s³，与“位置重算减去保存航向重算”的均匀 dt 捷度差峰值一致。内部样本的三阶差分系数为 1,-3,3,-1，其绝对系数和为 8；由 8 max|航向差|/dt³ 给出的界为 {summary['interior_heading_noise_third_difference_bound_rad_s3']:.9g} rad/s³，足以覆盖所观察边缘超限。

这些证据支持有限精度位置累加及三阶差分放大解释该轨迹的边缘超限，但不证明连续机床运动或笛卡尔各轴约束。该轨迹还因 {metrics['elapsed_time_s']:g} s 达到 max_steps 且有序覆盖不足而失败，数值诊断不影响该独立失败原因。实机侧仍需采样、量化、控制器重规划与反馈测量信息。

`aligned_raw_arrays.npz` 保存全部对齐序列；`original_violations.csv` 保留原违规样本和各重算值；`summary.json` 保存数值摘要；`hash_manifest.json` 绑定输入、当前评估源文件、诊断脚本和输出，并验证原 NPZ/JSON 在诊断前后哈希不变。
'''
    (output/'README.md').write_text(note,encoding='utf-8')
    shutil.copyfile(source,output/source.name)
    after = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (trace_path,metrics_path)}
    if before != after: raise RuntimeError('Read-only source trace changed during diagnostic')
    inputs=[trace_path,metrics_path,source,ROOT/'PPO_project/src/utils/revision_metrics.py',ROOT/'PPO_project/src/environment/cnc_env.py']
    inputs.extend(p for p in (trace_path.parent/'effective_config.yaml',trace_path.parent.parent/'config.yaml',trace_path.parent.parent/'source.zip') if p.exists())
    manifest={'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name!='hash_manifest.json'},
        'source_trace_and_metric_unchanged':True,
        'command':f'python PPO_project/tools/{source.name} --trace "{trace_path.relative_to(ROOT).as_posix()}" --output "{output.relative_to(ROOT).as_posix()}"'}
    (output/'hash_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=DEFAULT_TRACE)
    parser.add_argument('--output',type=Path,default=ROOT/'论文项目/revision_20260908/preview_direct_explore/numerical_roundoff_diagnostic')
    args=parser.parse_args()
    print(json.dumps(diagnose(args.trace,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
