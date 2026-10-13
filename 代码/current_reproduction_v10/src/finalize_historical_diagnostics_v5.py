"""Trace-identity receipts and exact runtime-decoder metadata correction."""
import copy
import json
import numpy as np
from common import ROOT, sha, utc, write_json
from replay_historical_strong_v5 import V4


def equality(case, current, previous, output):
    with np.load(current) as a, np.load(previous) as b:
        keys=set(a.files)==set(b.files)
        checks={key:bool(np.array_equal(a[key],b[key])) for key in a.files}
    receipt=dict(created_utc=utc(),case=case,current_trace=str(current),current_sha256=sha(current),
        previous_trace=str(previous),previous_sha256=sha(previous),same_array_keys=keys,
        exact_equal_arrays=checks,passed=keys and all(checks.values()),
        source_sha256=sha(__file__),scope='Every trace array compared elementwise; no result inferred from matching completion time')
    write_json(output,receipt)
    return receipt


def main():
    original=equality('original 1 ms',ROOT/'results/historical_strong_v5/original_dt1000us.npz',
        V4/'results/v4b_no_target_override/strong/original.npz',
        ROOT/'audit/historical_original_1ms_equality_v5.json')
    dense=equality('dense 0.5 ms',ROOT/'results/historical_strong_v5/dense_dt500us.npz',
        V4/'results/strong_checkpoint_v4/v4e_resolution_control/dense_dt500us.npz',
        ROOT/'audit/historical_dense_05ms_equality_v5.json')
    from runtime_physical_time_v5 import install
    install()
    from learned_action_pulse_policy import LearnedActionPulseRuntime
    source=ROOT/'audit/historical_action_diagnosis_exact_v5.json'
    report=copy.deepcopy(json.loads(source.read_text('utf-8')))
    checks=[]
    def physical(action):
        latent=np.arctanh(np.clip(np.asarray(action,dtype=float),-1.+1e-10,1.-1e-10))
        return LearnedActionPulseRuntime.decode_action(latent,None)
    for case in report['cases']:
        for row in case['all_records']:
            trace=ROOT/row['trace']
            if sha(trace)!=row['trace_sha256']:raise ValueError('Diagnostic trace hash changed')
            with np.load(trace) as data:
                original_action=physical(data['original_action'])
                branch_action=physical(data['branch_action'])
            angle_error=abs((branch_action[0]-original_action[0])-row['requested_steering_offset_radian'])
            feed_error=abs(branch_action[1]-original_action[1]*row['requested_physical_feed_factor'])
            preview_error=abs(branch_action[2]-original_action[2])
            if max(angle_error,feed_error,preview_error)>1e-10:raise AssertionError('Prescribed physical action not executed')
            row['original_physical_action']=original_action.tolist()
            row['branch_physical_action']=branch_action.tolist()
            row['physical_decode_verified']=True
            checks.append([angle_error,feed_error,preview_error])
        for state in case['states']:
            control=next(r for r in case['all_records'] if r['decision_index']==state['decision_index'] and r['original_control'])
            state['original_physical_action']=control['original_physical_action']
    report['metadata_correction']=dict(created_utc=utc(),superseded_metadata_source=str(source),
        superseded_metadata_sha256=sha(source),source_sha256=sha(__file__),
        reason='The diagnostic metadata decoder used the base JointRuntime preview formula (2.7+1.9u) rather than the actual inherited JointRuntimeV3 decoder (6.4+5.6u). Both use identical steering/feed formulas; normalized preview was preserved in every real branch. This correction changes physical-preview metadata only, not an action, trajectory or outcome.',
        actual_decoder='LearnedActionPulseRuntime.decode_action inherited from JointRuntimeV3',
        maximum_absolute_errors_steering_feed_preview=np.max(checks,axis=0).tolist(),
        all_180_prescribed_physical_actions_verified=True,all_original_trace_bytes_preserved=True,
        retained_earlier_float32_decode_diagnostic='results/historical_action_diagnosis_v5',
        authoritative_branch_traces='results/historical_action_diagnosis_exact_v5')
    write_json(ROOT/'audit/historical_action_diagnosis_corrected_v5.json',report)
    lines=['# Historical actor: exact-state action diagnosis','',
        '180 prescribed local branches; all 18 original-command controls match their archived rows exactly. These are local diagnostics, not complete-program successes.','',
        '| Program | Decision start (s) | Original steering (rad) | Original feed (mm/s) | Original preview (mm) | Feasible branches / 10 |',
        '|---|---:|---:|---:|---:|---:|']
    for case in report['cases']:
        for state in case['states']:
            angle,feed,preview=state['original_physical_action']
            lines.append(f"| {case['path_id']} | {state['start_time_s']:.3f} | {angle:.6f} | {feed:.6f} | {preview:.6f} | {state['feasible_branch_count']} |")
    lines.extend(['',
        'Steering offsets [-0.04,-0.02,0,0.02,0.04] rad and physical feed factors [0.8,1] preserve the actual preview. Their mapping was independently checked with the runtime subclass decoder (preview 0.8--12 mm).',
        'The final failed decision has no surviving tested branch in any of the three programs. All ten branches survive at each preceding inspected decision. Earlier intervention therefore has local feasible room, but this experiment does not establish downstream completion or prove that all possible braking maneuvers have become infeasible.',
        'The corrected JSON is authoritative for physical-action metadata. Earlier float32 diagnostic and base-decoder metadata remain archived; no original result or trace is overwritten.',''])
    (ROOT/'audit/historical_action_diagnosis_corrected_v5.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(original_all_arrays_exact=original['passed'],dense_half_ms_all_arrays_exact=dense['passed'],
        verified_branches=len(checks),physical_mapping_max_error=np.max(checks,axis=0).tolist()),indent=2))


if __name__=='__main__':main()
