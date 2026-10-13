"""Reuse the frozen complete-program runner with one registered action extension."""
import argparse
import copy
import json
import time
import torch
from common import ROOT, sha, utc, write_json
import run_recoverable_candidate_v5 as runner
from geometric_resumption_candidate_v5 import VERSION, GeometricResumptionCandidate

OUT = ROOT/'results/geometric_resumption_candidate_v5_4'


def register():
    file = OUT/'registration.json'
    if not file.exists():
        prior = ROOT/'results/recoverable_action_candidate_v5_3/registration.json'
        probe = ROOT/'results/recoverable_stall_probe_v5_1/report.json'
        reg = copy.deepcopy(json.loads(prior.read_text('utf-8')))
        reg.update(created_utc=utc(), version=VERSION,
            parent_candidate_registration_sha256=sha(prior),
            resumption_probe_sha256=sha(probe),
            changes=['When all original 25 actions are inadmissible, assess one gate-directed resumption action with the same one-decision and full-stop contract'],
            sole_algorithm_change='One additional action branch before the stored-stop fallback',
            resumption_action='Target the next unvisited gate (endpoint when all gates visited), preview 0.8 mm; steering compensates the common guide bearing and uses the unchanged pi/2 bound; feed 0 until filtered heading error <=0.05 rad, otherwise feed 20 mm/s.',
            no_state_reset='No heading, position, velocity, acceleration, tracker or gate reset',
            actual_command_sources=['raw model', 'finite-grid corrected model', 'geometric resumption action', 'stored certified cubic stop'],
            candidates='Original fixed 25 actions; one additional geometric action only when none passes',
            normal_action='Original raw action is preserved when its full contract passes',
            preview='Original raw third coordinate in the 25 original candidates; geometric resumption requests 0.8 mm through the unchanged action decoder',
            no_solution='Only when all 25 original actions and the geometric action fail, execute the previously certified stop',
            parameter_rule='One additional action definition fixed from the single retained-state probe; all three prior full programs, one pass each, no per-case tuning',
            implementation='Frozen runner and parent controller are reused in this process; an exact checked replacement extends only the empty-candidate branch in a subclass')
        reg['source_sha256'].update({name: sha(ROOT/'src'/name) for name in
            ['geometric_resumption_candidate_v5.py', 'run_geometric_resumption_v5.py']})
        write_json(file, reg)
    reg = json.loads(file.read_text('utf-8'))
    assert sha(runner.MODEL) == reg['model_sha256'] == runner.EXPECTED_MODEL
    assert all(sha(ROOT/'src'/name) == value for name, value in reg['source_sha256'].items())
    assert all(sha(ROOT/row['reference']) == row['reference_sha256'] for row in reg['cases'])
    return reg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    runner.OUT = OUT
    runner.VERSION = VERSION
    runner.RecoverableActionCandidate = GeometricResumptionCandidate
    reg = register()
    if args.audit:
        runner.unit_audit(reg)
    elif args.all:
        for case in runner.CASES:
            started_cpu = time.process_time()
            result = runner.run(case, reg)
            result['cost']['full_run_including_archive_and_metrics_process_cpu_s'] = time.process_time()-started_cpu
            result['action_source'] = 'Four logged sources: original model, bounded corrected model, geometric resumption action, stored certified cubic stop'
            result['action_array_definition'] = 'Only actually submitted action requests (original, corrected or geometric); action_decision_ids maps each row to its explicit decision source. Certified stops have submitted_action=null and no action row.'
            write_json(OUT/case/'actual.json', result)
        runner.summarize(reg)
    else:
        parser.error('Choose --audit or --all')
