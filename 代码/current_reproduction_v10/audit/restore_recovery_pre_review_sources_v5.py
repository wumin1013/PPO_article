"""Reconstruct pre-review source bytes from the recorded, reversible edits."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
archive = root/'archive/recoverable_v5_3_before_independent_review'
sources = root/'archive/recoverable_v5_3_before_metadata_correction/src'
reg = json.loads((archive/'registration.json').read_text('utf-8'))

def replace(text, old, new):
    assert text.count(old) == 1, old[:80]
    return text.replace(old, new)

name = 'recoverable_action_candidate_v5.py'
text = (sources/name).read_text('utf-8')
text = replace(text, 'import hashlib\n', '')
start = text.index('def stop_start_fingerprint(env):')
end = text.index('    core = env.env\n', start)
text = text[:start] + '''def execute_stop(env, positions, rows, retained_rows=None):
    """Execute every stored command, preserving all gates and ordinary rest rules."""
    if not np.array_equal(env.position, positions[0]):
        raise AssertionError('Stored stop starts at a different real position')
''' + text[end:]
text = replace(text, 'self.backup = (sr, sp, stop_start_fingerprint(env))', 'self.backup = (sr, sp)')
text = replace(text, 'stop_trace = (sr, sp, stop_start_fingerprint(self.clone))', 'stop_trace = (sr, sp)')
text = replace(text, 'rows = execute_stop(env, self.backup[1], self.backup[0], self.backup[2], self.actual_partial_rows)\n            action = None',
    'rows = execute_stop(env, self.backup[1], self.backup[0], self.actual_partial_rows)\n            action = raw.copy()  # Recorded only; the selected source is the stored stop.')
text = replace(text, 'submitted_action=None if action is None else action.tolist()', 'submitted_action=action.tolist()')
outputs = {name: text}

name = 'run_recoverable_candidate_v5.py'
text = (sources/name).read_text('utf-8')
text = replace(text, 'RecoverableActionCandidate, one_decision, execute_stop, stop_start_fingerprint)',
                     'RecoverableActionCandidate, one_decision, execute_stop)')
text = replace(text, 'stop_rows, stop_positions, stop_fingerprint = policy.backup', 'stop_rows, stop_positions = policy.backup')
start = text.index('    unchanged = state_bytes(env)\n')
end = text.index('    assert np.array_equal(executed, stop_rows)\n', start)
text = text[:start]+'    executed = execute_stop(env, stop_positions, stop_rows)\n'+text[end:]
text = replace(text, 'same_controller_resumed=True, stale_backup_rejected_before_integration=True, controller=summary,',
    'same_controller_resumed=True, controller=summary,')
text = replace(text, 'points, chunks, observations, actions, sources, action_decision_ids = [], [], [], [], [], []',
    'points, chunks, observations, actions, sources = [], [], [], [], []')
text = replace(text, '            if action is not None:\n                actions.append(action)\n                action_decision_ids.append(decision[\'decision_id\'])',
    '            actions.append(action)')
text = replace(text, '        action_decision_ids=np.asarray(action_decision_ids, dtype=np.int64),\n', '')
text = replace(text, "        action_array_definition='Only actually submitted model actions; action_decision_ids maps each row to the decision log. Certified stops have submitted_action=null and no actor action row.',\n", '')
outputs[name] = text

out = archive/'src'
out.mkdir(exist_ok=True)
checks = {}
for name, text in outputs.items():
    data = text.encode('utf-8')
    digest = hashlib.sha256(data).hexdigest()
    assert digest == reg['source_sha256'][name], (name, digest, reg['source_sha256'][name])
    (out/name).write_bytes(data)
    checks[name] = digest
(archive/'source_recovery_receipt.json').write_text(json.dumps(dict(
    source='Exact reverse of review changes, verified against the original pre-review registration',
    sources=checks), indent=2)+'\n', encoding='utf-8')
print(json.dumps(checks))
