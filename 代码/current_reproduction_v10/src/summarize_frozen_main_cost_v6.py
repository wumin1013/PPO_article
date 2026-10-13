"""Summarize recorded costs only; unavailable CPU and discovery times stay null."""
import json
from common import ROOT, sha, utc, write_json


def read(relative):
    return json.loads((ROOT / relative).read_text('utf-8'))


def main():
    recipe_path = 'evidence/main_model_inputs/training_recipe_inventory.json'
    historical_path = 'audit/historical_training_chain_inputs_v5.json'
    inventory = read(recipe_path)
    historical = read(historical_path)
    stages = []
    for name, records in inventory['historical_seed923202_stages'].items():
        receipt = records.get('replay_receipt.json', {})
        result = records.get('result.json', {})
        stages.append(dict(stage=name, wall_s=receipt.get('fields', {}).get('wall_s'),
            worker_wall_s=result.get('fields', {}).get('wall_s'), cpu_s=None,
            cpu_missing_reason='No measured CPU time in the retained receipt.',
            accounting='Outer stage receipt includes subprocess setup and scheduled checks. Worker wall time is nested and must not be added again.',
            receipt_archive_entry=receipt.get('archive_entry'), receipt_sha256=receipt.get('sha256'),
            worker_result_archive_entry=result.get('archive_entry'), worker_result_sha256=result.get('sha256')))
    final_fit = historical['archived_final_fit_receipts']['923202/replay_receipt.json']
    final = dict(stage='final_action_fit_12000', wall_s=final_fit['wall_s'], cpu_s=None,
        cpu_missing_reason='No measured CPU time in the retained receipt.',
        iterations=final_fit['offline_iterations'], source=historical_path,
        source_key='archived_final_fit_receipts/923202/replay_receipt.json',
        last_fit_log_wall_s=historical['last_main_fit_log']['wall_s'],
        accounting='The final fit log is nested in this receipt and must not be added again.')
    total = sum(x['wall_s'] for x in stages)
    if abs(total - inventory['historical_seed923202_receipt_total_wall_s']) > 1e-8:
        raise ValueError('Historical receipt sum changed')
    main_runs = []
    for case in ('original', 'dense'):
        for method in ('JNN', 'S2', 'S2R', 'SCMO'):
            relative = f'evidence/main_model_inputs/{case}_{method}.json'
            result = read(relative)
            cost = result.get('cost') or {}
            actor_calls = result.get('actor_calls')
            actor_time = cost.get('actor_wall_s')
            main_runs.append(dict(case=case, method=method, success=result['success'],
                complete_time_s=result['complete_time_s'], measured_total_wall_s=result.get('wall_s'),
                measured_cpu_s=result.get('cpu_s'), cost_fields=cost,
                actor_calls=actor_calls,
                mean_actor_wall_ms=(1000. * actor_time / actor_calls if actor_time is not None and actor_calls else None),
                source=relative, source_sha256=sha(ROOT / relative),
                scope='Registered command generation, trace export and independent audit; subfields retain their original measured scope.'))
    transfer_path = 'audit/frozen_main_generalization_v6_report.json'
    transfer = read(transfer_path)
    transfer_rows = [dict(run_id=x['run_id'], success=x['success'], wall_s=x['wall_s'], cpu_s=x['cpu_s'],
        cost=x['cost'], actor_calls=x['policy_calls'], source=x['source_record'],
        source_sha256=x['source_record_sha256']) for x in transfer['rows']]
    result = dict(created_utc=utc(), schema=1, model_id='jnnc_wendland96_192_c16d45e3',
        generator_sha256=sha(__file__),
        source_documents={p: sha(ROOT / p) for p in (recipe_path, historical_path, transfer_path)},
        historical_stage_receipts=stages, final_action_fit=final,
        historical_stage_wall_sum_s=total, final_action_fit_wall_s=final['wall_s'],
        registered_stage_plus_final_fit_wall_sum_s=total + final['wall_s'],
        historical_training_cpu_s=None, teacher_generation_wall_s=None, teacher_generation_cpu_s=None,
        feature_map_discovery_wall_s=None, feature_map_discovery_cpu_s=None,
        total_candidate_search_wall_s=None, total_candidate_search_cpu_s=None,
        missing_cost_scope='No complete measured receipts establish teacher generation, feature-map discovery or full candidate-search cost; null denotes unrecorded, not zero.',
        pure_optimizer_wall_s=None,
        pure_optimizer_scope='Retained stage wall times include scheduled checks and cannot be separated into optimizer-only cost.',
        standalone_online_planning_wall_s=None,
        online_scope='Actor and executor wall totals are recorded for simulated execution; there are no per-decision latency bounds or hard-deadline measurements.',
        historical_main_runs=main_runs, frozen_transfer_runs=transfer_rows,
        frozen_transfer_total_wall_s=sum(x['wall_s'] for x in transfer_rows),
        frozen_transfer_total_cpu_s=sum(x['cpu_s'] for x in transfer_rows),
        cost_caution='Wall and CPU totals use their recorded scopes and are not interchangeable. Runs share the host; no real-time guarantee follows from average costs.')
    write_json(ROOT / 'audit/frozen_main_costs_v6.json', result)
    lines = ['# Recorded costs of the frozen main controller', '',
        'All values below come from retained timing fields. Missing measurements remain null in the JSON record.', '',
        '| Stage | Outer wall time (s) | Nested worker wall time (s) | CPU time |', '|---|---:|---:|---|']
    for item in stages:
        worker = '--' if item['worker_wall_s'] is None else f"{item['worker_wall_s']:.3f}"
        lines.append(f"| {item['stage']} | {item['wall_s']:.3f} | {worker} | Unrecorded |")
    lines.extend(['', f'Registered historical stage total: {total:.6f} s.',
        f"Final 12,000-iteration fit: {final['wall_s']:.6f} s.",
        'These totals include each stage\'s scheduled checks and omit prior teacher generation, feature-map discovery and candidate search. Nested worker times are not added twice.', '',
        f"The ten new frozen-model executions used {result['frozen_transfer_total_wall_s']:.6f} s recorded rollout wall time and {result['frozen_transfer_total_cpu_s']:.6f} s CPU time; CPU additionally includes setup and actor construction.",
        'Per-run measured actor/executor fields and all eight main comparison runs are preserved in the JSON. No timing rerun or hardware test was performed.', ''])
    (ROOT / 'audit/frozen_main_costs_v6.md').write_text('\n'.join(lines), 'utf-8')
    destination = ROOT / 'shared_manuscript/frozen_main_v6'
    macros = [r'% Generated from recorded costs and all ten complete or retained-prefix results.']
    values = dict(FrozenTrainingStageWall=f'{total:.1f}', FrozenFinalFitWall=f"{final['wall_s']:.1f}",
        FrozenTransferNominalComplete=str(transfer['successful_by_resolution']['0.001']),
        FrozenTransferFineComplete=str(transfer['successful_by_resolution']['0.0005']),
        FrozenTransferCases='5', FrozenTransferCells='10',
        FrozenTransferCompleteCells=str(transfer['successful_cells']),
        FrozenTransferPairedComplete=str(transfer['paired_complete_cases']))
    for key, value in values.items():
        macros.append('\\newcommand{\\' + key + '}{' + value + '}')
    (destination / 'generated_frozen_macros.tex').write_text('\n'.join(macros) + '\n', 'utf-8')
    print(json.dumps({k: result[k] for k in ('historical_stage_wall_sum_s', 'final_action_fit_wall_s',
        'registered_stage_plus_final_fit_wall_sum_s', 'frozen_transfer_total_wall_s', 'frozen_transfer_total_cpu_s')}, indent=2))


if __name__ == '__main__':
    main()
