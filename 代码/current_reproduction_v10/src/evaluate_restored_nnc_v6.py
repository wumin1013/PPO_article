"""Registered complete-task evaluation of the recovered protected project NNC adapter."""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from common import ROOT, sha, utc, write_json

OUT = ROOT / 'results/restored_nnc_protected_v6'
MODEL_SHA = '24d263a0e07fe471e7995ebae2d0a0ee7a2ee03e2feec80ea50aa5b94b0cfe6b'
POLICY_DT = .001


def read(p):
    return json.loads(Path(p).read_text('utf-8'))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def register():
    assert not OUT.exists(), 'Use a new version for another evaluation; never overwrite runs.'
    recovered = read(ROOT / 'audit/nnc_recovered_inputs_v5.json')
    assert recovered['checkpoint_sha256'] == MODEL_SHA and recovered['all_six_registered_sources_exact']
    archive = Path(recovered['historical_project_root']) / 'revision_20260908/historical/source/PPO_project'
    OUT.mkdir(parents=True)
    copies = []
    for source in sorted(archive.rglob('*.py')):
        target = OUT / 'source/PPO_project' / source.relative_to(archive)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        copies.append(dict(original=str(source), local=target.relative_to(OUT).as_posix(), sha256=sha(target)))
    for source, relative in ((Path(recovered['checkpoint']), 'checkpoint.pth'),
            (Path(recovered['historical_project_root']) / 'revision_20260908/historical/baseline_policy/baseline_policy.yaml', 'baseline_policy.yaml')):
        target = OUT / relative
        shutil.copy2(source, target)
        copies.append(dict(original=str(source), local=relative, sha256=sha(target)))
    specs = []
    for case in ('original', 'dense'):
        folder = ROOT / 'baseline_records/supplementary_reproduction/cases' / case
        for name in ('reference.csv', 'geometry.json'):
            target = OUT / 'programs' / case / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / name, target)
            copies.append(dict(original=str(folder / name), local=target.relative_to(OUT).as_posix(), sha256=sha(target)))
        geometry = read(folder / 'geometry.json')
        chord = max([p.get('chord_error_bound_mm', 0.) for p in geometry['primitives']] + [0.])
        for dt in (.001, .0005):
            specs.append(dict(id=f'{case}_NNC_dt{round(dt*1e6)}us', case=case, dt_s=dt,
                              nominal_tolerance_mm=1.25, chord_reserve_mm=chord,
                              tolerance_mm=1.25-chord, wall_budget_s=240.))
    write_json(OUT / 'registration.json', dict(created_utc=utc(), model_sha256=MODEL_SHA,
        source_files=copies, driver_sha256=sha(Path(__file__)), runs=specs,
        method='Recovered project PPO-based NNC adapter with original project feed protections restored',
        published_method_reproduction=False, new_training_updates=0, policy_dt_s=POLICY_DT,
        protections='Use archived Env directly; do not import OriginalNNCPlanarEnv. Preserve angular speed/acceleration/jerk feed caps, turn prebraking, endpoint braking and cap release logic.',
        kcm='Preserve registered enable_kcm=false; do not add the J-NNC vector projector or other KCM.',
        resolution='Policy requests sampled every 1 ms and held over two 0.5 ms commands. Per-step cap release and hysteresis/stall durations are rescaled to retain their nominal physical rates/durations.',
        geometry='epsilon=2*(1.25-source chord error bound). The shared GateTracker audits every entire command interval, full ordered traversal and endpoint.',
        initial_state='Archived reset(random_start=False), requiring exactly zero velocity and acceleration.',
        completion='All common ordered gates, valid geometry, vector limits, and >=20 ms final rest. Native done is logged but is not common completion; continue the same controller if still feasible.',
        stopping='Stop on first infeasible actual command (retain it), complete common task, simulation horizon 5+length/5 seconds, exception, or registered wall budget. No failure is assigned completion time; all four runs retained.',
        numerical_tolerance='Common vector acceptance ratio <=1+2e-5; geometry uses the unchanged current corridor implementation.',
        preflight='Verify archived source/model hashes, exact class method origin, observation/action dimensions, stationary reset, epsilon interpretation and policy period without executing any command.'))


def verify():
    reg = read(OUT / 'registration.json')
    assert sha(OUT / 'checkpoint.pth') == MODEL_SHA
    assert sha(Path(__file__)) == reg['driver_sha256']
    for row in reg['source_files']:
        assert sha(OUT / row['local']) == row['sha256'], row['local']
    return reg


def build(spec):
    project = OUT / 'source/PPO_project'
    sys.path.insert(0, str(project))
    from src.environment.cnc_env import Env
    from tools.phase32_export_best_trajectories import _make_agent
    assert Path(inspect.getfile(Env)).resolve() == (project / 'src/environment/cnc_env.py').resolve()
    assert Env._compute_p4_pre_step_status.__qualname__ == 'Env._compute_p4_pre_step_status'
    cfg = yaml.safe_load((OUT / 'baseline_policy.yaml').read_text('utf-8'))
    assert cfg['experiment']['enable_kcm'] is False
    cfg = copy.deepcopy(cfg)
    dt = spec['dt_s']
    scale = dt / POLICY_DT
    rw = cfg['reward_weights']
    for key in ('vcap_rate_up', 'vcap_rate_down'):
        rw['p8'][key] *= scale
    rw['p8']['corner_exit_hold_steps'] = round(rw['p8']['corner_exit_hold_steps'] / scale)
    rw['p4']['stall_steps'] = round(rw['p4']['stall_steps'] / scale)
    reference = np.loadtxt(OUT / 'programs' / spec['case'] / 'reference.csv', delimiter=',', skiprows=1)
    horizon = 5. + np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()/5.
    e = cfg['environment']
    e.update(epsilon=2.*spec['tolerance_mm'], interpolation_period=dt, max_steps=int(np.ceil(horizon/dt)))
    env = Env(device=torch.device('cpu'), epsilon=e['epsilon'], interpolation_period=dt,
        **cfg['kinematic_constraints'], Pm=[p.copy() for p in reference], max_steps=e['max_steps'],
        lookahead_points=e.get('lookahead_points', 5), lookahead_obs_enabled=e.get('lookahead_obs_enabled', True),
        lookahead_obs_scales=e.get('lookahead_obs_scales', [1.]), reward_weights=rw,
        curvature_observation=e.get('curvature_observation'), return_normalized_obs=True,
        path_name=spec['case'], disable_kcm=True)
    env.debug_collector.dump_dir = OUT / 'diagnostic_dumps' / spec['id']
    torch.set_num_threads(1)
    agent = _make_agent(cfg, env, torch.device('cpu'))
    payload = torch.load(OUT / 'checkpoint.pth', map_location='cpu', weights_only=False)
    agent.actor.load_state_dict(payload['actor'], strict=True)
    agent.critic.load_state_dict(payload['critic'], strict=True)
    agent.actor.eval(); agent.critic.eval()
    obs = env.reset(random_start=False)
    assert obs.shape == (14,) and env.action_space_dim == 2
    assert env.velocity == env.acceleration == env.angular_vel == env.angular_acc == 0.
    assert np.array_equal(env.current_position, reference[0])
    assert env.half_epsilon == spec['tolerance_mm']
    assert round(POLICY_DT / dt) * dt == POLICY_DT
    return env, agent, obs, reference, cfg


def worker(run_id, preflight=False):
    reg = verify()
    spec = next(r for r in reg['runs'] if r['id'] == run_id)
    torch.manual_seed(42); np.random.seed(42)
    started, cpu = time.perf_counter(), time.process_time()
    env, agent, obs, reference, cfg = build(spec)
    if preflight:
        with torch.no_grad():
            mu, _ = agent.actor(torch.as_tensor(obs, dtype=torch.float).reshape(1, 14))
        record = dict(run_id=run_id, passed=True, commands_executed=0, observation_dimension=len(obs),
                      actor_output_dimension=int(mu.numel()), original_caps_method=True,
                      disable_kcm=env.disable_kcm, half_epsilon=env.half_epsilon,
                      configuration=cfg, wall_s=time.perf_counter()-started)
        write_json(OUT / f'{run_id}_preflight.json', clean(record))
        return
    assert read(OUT / f'{run_id}_preflight.json')['passed']
    output = OUT / run_id
    assert not output.with_suffix('.json').exists() and not output.with_suffix('.npz').exists()
    # The shared geometry audit is standalone; importing the J-NNC executor
    # would collide with the archived NNC package's own src namespace.
    from runtime_bootstrap import setup
    setup()
    from geometry_spec import Corridor, GateTracker
    from metrics_exact_index_v5 import metrics
    dt = spec['dt_s']
    corridor = Corridor(reference, epsilon=spec['tolerance_mm'], closed=True)
    tracker = GateTracker(corridor, reference[0])
    positions = [reference[0].copy()]; observations = []; actions = []; native_done = []
    p4_records = []; first_failure = None; error = None; success = False; rest_ticks = 0
    retained_exception_commands = 0
    v_prev = np.zeros(2); a_prev = np.zeros(2); reason = 'simulation_horizon_exhausted'; action = None
    control_steps = round(POLICY_DT/dt)
    try:
        with torch.no_grad():
            for tick in range(cfg['environment']['max_steps']):
                if time.perf_counter()-started > spec['wall_budget_s']:
                    reason = 'wall_budget_exhausted'; break
                if tick % control_steps == 0:
                    observations.append(np.asarray(obs).copy())
                    mu, _ = agent.actor(torch.as_tensor(obs, dtype=torch.float).reshape(1, 14))
                    action = np.clip(mu.squeeze(0).numpy().astype(float), env.action_space.low, env.action_space.high)
                    actions.append(action.copy())
                obs, _, done, info = env.step(action)
                point = np.asarray(env.current_position, dtype=float).copy()
                v = (point-positions[-1])/dt; a = (v-v_prev)/dt; j = (a-a_prev)/dt
                positions.append(point); tracker.update(point)
                peaks = np.array([np.linalg.norm(x) for x in (v, a, j)])
                p4_records.append(clean(info.get('p4_status', {})))
                if done: native_done.append(dict(tick=tick+1, reason=str(info.get('done_reason', ''))))
                rest_ticks = rest_ticks+1 if peaks[0] <= .001 and peaks[1] <= .01 else 0
                bad = [key for key, ratio in zip(('speed', 'acceleration', 'jerk'), peaks/[100., 2000., 62500.]) if ratio > 1.+2e-5]
                if not tracker.inside: bad.append('command_interval_corridor')
                if bad:
                    first_failure = dict(tick=tick+1, time_s=(tick+1)*dt, reasons=bad, vector_values=peaks.tolist())
                    reason = 'first_infeasible_command'; break
                if tracker.snapshot(endpoint_tolerance=spec['tolerance_mm'])['geometry_complete'] and rest_ticks*dt >= .02-1e-10:
                    success = True; reason = 'success'; break
                v_prev, a_prev = v, a
    except Exception:
        error = traceback.format_exc(); reason = 'execution_exception'
        # Archived Env appends the actual position before perception/reward.
        # Preserve a moved command even if a later part of step raises.
        actual = np.asarray(env.trajectory, dtype=float)
        assert len(actual) >= len(positions)
        assert np.array_equal(actual[:len(positions)], np.asarray(positions))
        for point in actual[len(positions):]:
            positions.append(point.copy()); tracker.update(point)
            retained_exception_commands += 1
    positions = np.asarray(positions)
    np.savez_compressed(output, positions=positions, reference=reference,
                        observations=np.asarray(observations), actions=np.asarray(actions),
                        timestamps=np.arange(len(positions))*dt)
    audit = metrics(reference, positions, dt=dt, tolerance=spec['tolerance_mm'], declared_success=success, reason=reason)
    audit.update(method='NNC', path=spec['case'], runtime_version='restored_project_NNC_protected_v6',
        model_sha256=MODEL_SHA, registration_sha256=sha(OUT/'registration.json'),
        trace_sha256=sha(output.with_suffix('.npz')), first_failure=first_failure, error=error,
        status='complete' if success else 'infeasible' if first_failure else 'error' if error else 'incomplete_budget',
        retained_exception_commands=retained_exception_commands,
        new_training_updates=0, native_done_events=native_done, actor_calls=len(actions),
        policy_dt_s=POLICY_DT, all_peaks_scope='full_completed_program' if success else 'actual_executed_failure_or_unfinished_prefix',
        initial_rest=True, original_project_caps_preserved=True, original_enable_kcm=False,
        source_identity='Recovered project PPO NNC adapter; not a verified reimplementation of the published DDPG NNC.',
        wall_s=time.perf_counter()-started, process_cpu_s=time.process_time()-cpu)
    write_json(output.with_suffix('.json'), clean(audit))
    write_json(OUT/f'{run_id}_execution_details.json', clean(dict(configuration=cfg, p4_status=p4_records, gate_events=tracker.events)))
    verify()
    print('RESTORED_NNC', run_id, audit['status'], audit['complete_time_s'], audit.get('elapsed_prefix_s'), flush=True)


def matrix(preflight):
    reg = verify()
    for spec in reg['runs']:
        suffix = '_preflight.log' if preflight else '.log'
        with (OUT/(spec['id']+suffix)).open('w', encoding='utf-8') as log:
            cmd = [sys.executable, '-B', str(Path(__file__)), 'worker', '--run', spec['id']]
            if preflight: cmd.append('--preflight')
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=spec['wall_budget_s']+120.)
        print('NNC_SUBPROCESS', spec['id'], result.returncode, flush=True)
        if result.returncode:
            print((OUT/(spec['id']+suffix)).read_text('utf-8')[-4000:], flush=True)
            if preflight: raise RuntimeError('Preflight failed; no commands executed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'preflight', 'run', 'worker'))
    parser.add_argument('--run'); parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.mode == 'register': register()
    elif args.mode == 'worker': worker(args.run, args.preflight)
    else: matrix(args.mode == 'preflight')
