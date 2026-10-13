"""Independently recheck actions and vector differences after all eight replays."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
import sys
import numpy as np
import torch
import run_unified as run

sys.path.insert(0,str(run.ROOT.parent/'terminal_research'))
from audit_launch_effect import independent_actor
torch.set_num_threads(1)


def main():
    target = run.ROOT/'actor_and_vector_audit.json'
    assert not target.exists(), 'Preserve earlier audit receipt.'
    reg = run.verify()
    runtime_sources = run.read(run.BASE/'runtime/source_manifest.json')
    for source in runtime_sources:
        assert run.sha(run.BASE/'runtime'/source['relative']) == source['sha256'], source['relative']
    summary = run.read(run.ROOT/'trace_summary.json')
    assert len(summary['rows']) == 8
    old_root = run.BASE/'evidence/frozen_main_v6'
    old_manifest = run.read(old_root/'manifest.json')
    old_cells = {cell['id']:cell for cell in old_manifest['runs']}
    models = {m:torch.load(run.ROOT/row['file'],map_location='cpu',weights_only=False)
              for m,row in reg['models'].items()}
    rows = []
    for cell in summary['rows']:
        record = run.read(cell['record'])
        assert run.sha(cell['record']) == cell['record_sha256']
        assert run.sha(cell['trace']) == cell['trace_sha256']
        with np.load(cell['trace'],allow_pickle=False) as z:
            observations, actions, commands, positions = (z[k] for k in ('observations','actions','rows','positions'))
        reproduced = np.asarray([independent_actor(obs,models[cell['method']]) for obs in observations])
        assert np.array_equal(reproduced,actions)
        assert len(actions) == record['actor_calls'] == record['submitted_actions']
        assert record['guide_calls'] == record['guide_interventions'] == 0
        dt = cell['dt_s']
        terminal = np.flatnonzero(commands[:,12])
        active_ticks = int(terminal[0]) if len(terminal) else len(commands)
        macro = round(.02/dt)
        feeds = 50.*(1.+np.clip(actions[:,1].astype(float),-1.+1e-10,1.-1e-10))
        requested = np.repeat(feeds,macro)[:active_ticks]
        feed_error = float(np.abs(requested-np.linalg.norm(commands[:active_ticks,8:10],axis=1)).max(initial=0.))
        assert feed_error < 1e-9
        v = np.diff(positions,axis=0)/dt
        a = np.diff(np.vstack((np.zeros(2),v)),axis=0)/dt
        j = np.diff(np.vstack((np.zeros(2),a)),axis=0)/dt
        peaks = np.asarray([np.linalg.norm(x,axis=1).max(initial=0.) for x in (v,a,j)])
        assert np.allclose(peaks,record['vector_peaks'],rtol=1e-12,atol=1e-8)
        rest = (np.linalg.norm(v,axis=1)<=.001)&(np.linalg.norm(a,axis=1)<=.01)
        nonrest = np.flatnonzero(~rest)
        rest_ticks = len(rest)-(int(nonrest[-1])+1 if len(nonrest) else 0)
        assert abs(rest_ticks*dt-record['terminal_rest_s']) < 1e-10
        settings = record['executor_setup']
        assert record['runtime_version'] == settings['version'] == 'v5_physical_time'
        assert settings['control_period_s'] == .02 and settings['heading_filter_tau_s'] == .012
        assert settings['guide_transition_width_mm'] == .1
        assert record['new_training_updates'] == 0
        old_cell = old_cells[cell['run_id']]
        old_path = old_root/old_cell['trace']
        assert run.sha(old_path) == old_cell['trace_sha256']
        with np.load(old_path,allow_pickle=False) as old:
            old_raw_key = 'latents' if cell['method']=='FIXED' else 'actions'
            old_raw = old[old_raw_key]
            assert old_raw.shape[1] == actions.shape[1]
            common = min(len(old_raw),len(actions))
            raw_mask = np.any(old_raw[:common]!=actions[:common],axis=1)
            raw_changed = np.flatnonzero(raw_mask)
            common_positions = min(len(old['positions']),len(positions))
            semantic_comparison = dict(old_trace_sha256=run.sha(old_path),
                old_raw_action_key=old_raw_key,new_raw_action_key='actions',
                old_raw_action_shape=list(old_raw.shape),new_raw_action_shape=list(actions.shape),
                all_raw_actions_bitwise_equal=bool(np.array_equal(old_raw,actions)),
                common_raw_action_rows_bitwise_equal=not bool(np.any(raw_mask)),
                first_differing_common_raw_action_index=int(raw_changed[0]) if len(raw_changed) else None,
                maximum_common_raw_action_abs_difference=float(np.abs(old_raw[:common]-actions[:common]).max(initial=0.)),
                maximum_common_position_abs_difference_mm=float(np.abs(old['positions'][:common_positions]-positions[:common_positions]).max(initial=0.)),
                schema_note='Historical FIXED actions contain 3 decoded held channels; its latents contain the 2 raw policy outputs. Only latents are semantically comparable with new actions.' if cell['method']=='FIXED' else None)
        rows.append(dict(run_id=cell['run_id'],success=record['success'],reason=record['reason'],
            complete_time_s=record['complete_time_s'],trace_sha256=cell['trace_sha256'],
            actor_calls=len(actions),every_actor_action_bitwise_equal=True,
            decoded_requested_feed_max_abs_error_mm_s=feed_error,
            guide_calls=0,guide_interventions=0,
            recomputed_vector_peaks=peaks.tolist(),recomputed_rest_s=rest_ticks*dt,
            historical_semantic_action_comparison=semantic_comparison,
            full_trajectory_geometry_scope='Existing worker full interval/gate audit retained; this second receipt rechecks actions and vector differences only.'))
        print(cell['run_id'],'ACTOR_AND_VECTOR_PASS',record['success'],flush=True)
    run.write(target,dict(status='passed',created_utc=run.utc(),generator_sha256=run.sha(__file__),
        independent_actor_expression_sha256=run.sha(run.ROOT.parent/'terminal_research/audit_launch_effect.py'),
        registration_sha256=run.sha(run.REG),trace_summary_sha256=run.sha(run.ROOT/'trace_summary.json'),
        runtime_source_files_rehashed_after_replay=len(runtime_sources),
        all_eight_cells_retained=True,additional_physical_rollouts=0,new_training_updates=0,
        all_actor_actions_bitwise_equal=True,success_count=sum(r['success'] for r in rows),
        failure_count=sum(not r['success'] for r in rows),rows=rows))
    print(json.dumps(dict(output=str(target),sha256=run.sha(target)),indent=2),flush=True)


if __name__=='__main__':
    main()
