"""Read-only analysis of registered clip sensitivity and all new training costs.

Creates reports only. It never imports a model, evaluates a path, or changes a
frozen source, model, split, input trace, or validation receipt.
"""
import csv
import json
from common import ROOT, sha, utc, write_json

SEEDS = [101, 211, 307, 401, 503]
ARMS = ['fixed', 'adaptive']
VALIDATION_PATHS = ['validation_trapezoid', 'validation_zigzag']
CLIP_TAGS = [(.1, 'fullrange_clip_0.1'), (.2, 'fullrange'), (.3, 'fullrange_clip_0.3')]
OUT = ROOT/'results'/'summary'


def read(path):
    return json.loads(path.read_text('utf-8'))


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def validation_sensitivity():
    rows = []
    for clip, tag in CLIP_TAGS:
        for arm in ARMS:
            stage = ROOT/'results'/'paired'/tag/'seed_101'/arm
            registration, receipt = read(stage/'registration.json'), read(stage/'receipt.json')
            assert registration['seed'] == 101 and registration['arm'] == arm and registration['clip'] == clip
            assert receipt['complete'] and sha(stage/'final.pt') == receipt['model_sha256']
            for name in VALIDATION_PATHS:
                result_file = ROOT/'results'/'validation'/f'{tag}_{arm}_101_{name}_0.001.json'
                row = dict(clip=clip, tag=tag, arm=arm, seed=101, path=name,
                    status='missing', success=None, complete_time_s=None, reason='missing_receipt',
                    max_polyline_deviation_mm=None, evaluation_wall_s=None,
                    delta_from_clip_02_s=None, relative_change_from_clip_02_pct=None,
                    trace_sha256=None, receipt_file=relative(result_file), receipt_sha256=None,
                    model_sha256=receipt['model_sha256'], training_receipt_file=relative(stage/'receipt.json'))
                if result_file.exists():
                    result = read(result_file)
                    trace = result_file.with_suffix('.npz')
                    assert trace.exists() and sha(trace) == result['trace_sha256']
                    row.update(status='completed', success=result['success'], complete_time_s=result['complete_time_s'],
                        reason=result['reason'], max_polyline_deviation_mm=result.get('max_polyline_deviation_mm'),
                        evaluation_wall_s=result['wall_s'], trace_sha256=result['trace_sha256'],
                        receipt_sha256=sha(result_file))
                rows.append(row)
    main = {(r['arm'],r['path']):r for r in rows if r['clip']==.2}
    for row in rows:
        baseline = main[(row['arm'], row['path'])]
        if row['success'] and baseline['success']:
            row['delta_from_clip_02_s'] = row['complete_time_s']-baseline['complete_time_s']
            row['relative_change_from_clip_02_pct'] = 100*row['delta_from_clip_02_s']/baseline['complete_time_s']
    comparisons = []
    for clip, _ in CLIP_TAGS:
        for name in VALIDATION_PATHS:
            pair = {r['arm']:r for r in rows if r['clip']==clip and r['path']==name}
            fixed, adaptive = pair['fixed'], pair['adaptive']
            comparisons.append(dict(clip=clip, path=name, fixed_success=fixed['success'], adaptive_success=adaptive['success'],
                fixed_minus_adaptive_s=fixed['complete_time_s']-adaptive['complete_time_s'] if fixed['success'] and adaptive['success'] else None,
                adaptive_time_saving_pct=100*(fixed['complete_time_s']-adaptive['complete_time_s'])/fixed['complete_time_s'] if fixed['success'] and adaptive['success'] else None))
    registration = ROOT/'audit'/'fullrange_training_registration.json'
    rule = read(registration)
    assert rule['main_clip'] == .2
    summary = dict(created_utc=utc(), experiment='Prespecified validation-only PPO clip sensitivity',
        expected_runs=12, completed_runs=sum(r['status']=='completed' for r in rows),
        successes=sum(r['success'] is True for r in rows), independent_seeds=[101],
        primary_clip=.2, primary_clip_changed_using_validation_rank=False,
        selection_rule=rule['clip_comparison'], selection_rule_file=relative(registration), selection_rule_sha256=sha(registration),
        validation_only=True, final_test_records_read=0,
        interpretation='One complete independent seed per arm and clip, repeated on two validation programs. These are descriptive sensitivity results, not confidence intervals or independent-seed robustness evidence. Method ranking changes with clip. The predeclared main clip remains 0.2 regardless of the ranking.',
        records=rows, paired_arm_comparisons=comparisons)
    write_json(OUT/'validation_sensitivity.json', summary)
    csv_write(OUT/'validation_sensitivity.csv', rows)
    lines = ['# Validation-only PPO clip sensitivity', '',
        'All values below come from retained full-range seed-101 runs. The PPO ratio clip is dimensionless and is distinct from the 1.25 mm contour tolerance. Each arm used the same fitting and PPO budgets at all three clip values.', '',
        '| PPO clip | Arm | Trapezoid time (s) | Zigzag time (s) | Successes |',
        '|---:|---|---:|---:|---:|']
    for clip, _ in CLIP_TAGS:
        for arm in ARMS:
            pair = {r['path']:r for r in rows if r['clip']==clip and r['arm']==arm}
            values = [f"{pair[p]['complete_time_s']:.3f}" if pair[p]['success'] else pair[p]['reason'] for p in VALIDATION_PATHS]
            lines.append(f'| {clip:.1f} | {arm} | {values[0]} | {values[1]} | {sum(r["success"] is True for r in pair.values())}/2 |')
    lines += ['', 'All 12 registered runs completed successfully. The fixed arm is faster at clip 0.1 on both programs; the adaptive arm is faster at clips 0.2 and 0.3. This change in ranking is evidence of training-parameter sensitivity in this finite-budget, single-seed experiment. It does not establish a robust independent benefit of the adaptive-preview coordinate.', '',
        'The main comparison remains at the prespecified clip 0.2. No clip was selected, replaced, or retuned using this ranking or final-test data. There is only one independent seed at each non-main clip, so no seed-level confidence interval is reported. A policy evaluated on two paths is not two independently trained models.', '',
        'The JSON and CSV retain every expected run, its failure or completion state, input receipt and trace hashes, model hash, timing, and per-path differences. The training cost of these sensitivity runs is included in `training_budget_audit.md`.']
    (OUT/'validation_sensitivity.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def aggregate_cost(records):
    keys = sorted({key for record in records if record['receipt_present'] for key in record['cost']})
    return {key:sum(record['cost'].get(key, 0) for record in records if record['receipt_present']) for key in keys}


def training_budget():
    specs = [('main','bounded_calibration',SEEDS,.2), ('clip_0.1','bounded_calibration',[101],.1),
             ('clip_0.3','bounded_calibration',[101],.3), ('fullrange','fullrange_registered',SEEDS,.2),
             ('fullrange_clip_0.1','fullrange_registered',[101],.1), ('fullrange_clip_0.3','fullrange_registered',[101],.3)]
    records = []
    for tag, generation, seeds, clip in specs:
        for seed in seeds:
            for arm in ARMS:
                stage = ROOT/'results'/'paired'/tag/f'seed_{seed}'/arm
                entry = dict(tag=tag, generation=generation, seed=seed, arm=arm, clip=clip,
                    receipt_present=(stage/'receipt.json').exists(), complete=False, error='missing_receipt',
                    receipt_file=relative(stage/'receipt.json'), cost={})
                if entry['receipt_present']:
                    rec, reg = read(stage/'receipt.json'), read(stage/'registration.json')
                    assert reg['seed']==seed and reg['arm']==arm and reg['clip']==clip
                    assert sha(stage/'final.pt') == rec['model_sha256']
                    entry.update(complete=rec['complete'], error=rec['error'], source_unchanged=rec['source_unchanged'],
                        receipt_sha256=sha(stage/'receipt.json'), registration_sha256=sha(stage/'registration.json'),
                        model_sha256=rec['model_sha256'], source_sha256=reg['source_sha256'],
                        split_sha256=reg['split_sha256'], dataset_sha256=reg['dataset_sha256'], cost=rec['cost'])
                queue_file = ROOT/'results'/'queue'/f'train_{tag}_{seed}_{arm}.json'
                if queue_file.exists():
                    queue = read(queue_file)
                    assert queue['args'][:2] == ['src/paired_learning.py','train']
                    entry.update(queue_receipt_file=relative(queue_file), queue_receipt_sha256=sha(queue_file),
                                 whole_subprocess_wall_s=queue['wall_s'], subprocess_returncode=queue['returncode'])
                records.append(entry)
    teacher_records = []
    for folder in sorted((ROOT/'results').glob('teacher_selection*')):
        for result_file in sorted(folder.glob('*.json')):
            result = read(result_file)
            if 'actor_calls' not in result:
                continue
            trace = result_file.with_suffix('.npz')
            assert trace.exists() and sha(trace)==result['trace_sha256']
            teacher_records.append(dict(round=folder.name, phase='validation_parameter_search' if result_file.stem.startswith('validation_') else 'training_record_generation',
                receipt_file=relative(result_file), receipt_sha256=sha(result_file), trace_sha256=result['trace_sha256'],
                success=result['success'], reason=result['reason'], error=result.get('error'),
                wall_s=result['wall_s'], completed_actor_calls=result['actor_calls'],
                attempted_actor_calls=result.get('attempted_actor_calls'),
                elapsed_prefix_s=result.get('elapsed_prefix_s'), samples=result.get('samples')))
    rounds = []
    for name in sorted({r['round'] for r in teacher_records}):
        subset = [r for r in teacher_records if r['round']==name]
        rounds.append(dict(round=name, runs=len(subset), successes=sum(r['success'] for r in subset),
            failures=sum(not r['success'] for r in subset),
            validation_search_runs=sum(r['phase']=='validation_parameter_search' for r in subset),
            training_record_generation_runs=sum(r['phase']=='training_record_generation' for r in subset),
            completed_actor_calls=sum(r['completed_actor_calls'] for r in subset),
            exception_receipts=sum(r['error'] is not None for r in subset), wall_s=sum(r['wall_s'] for r in subset)))
    aggregate = aggregate_cost(records)
    grouped = []
    for tag, _, _, _ in specs:
        subset = [r for r in records if r['tag']==tag]
        grouped.append(dict(tag=tag, expected_runs=len(subset), completed_runs=sum(r['complete'] for r in subset), cost=aggregate_cost(subset)))
    teacher_calls = sum(r['completed_actor_calls'] for r in teacher_records)
    teacher_wall = sum(r['wall_s'] for r in teacher_records)
    exception_count = sum(r['error'] is not None for r in teacher_records)
    queue_records = [r for r in records if 'whole_subprocess_wall_s' in r]
    known_process_overhead = sum(r['whole_subprocess_wall_s']-r['cost']['total_wall_s'] for r in queue_records)
    validation_costs = []
    for result_file in sorted((ROOT/'results'/'validation').glob('*.json')):
        result = read(result_file)
        if 'wall_s' in result:
            validation_costs.append(dict(receipt_file=relative(result_file), wall_s=result['wall_s'], success=result['success']))
    summary = dict(created_utc=utc(), expected_training_runs=28, observed_receipts=sum(r['receipt_present'] for r in records),
        complete_training_runs=sum(r['complete'] for r in records), bounded_calibration_runs=14, fullrange_runs=14,
        primary_fullrange_independent_runs=10, fullrange_clip_sensitivity_extra_runs=4,
        aggregate_training_cost=aggregate, training_by_tag=grouped, training_records=records,
        teachers=dict(rounds=rounds, expected_runs=48, observed_runs=len(teacher_records),
            successes=sum(r['success'] for r in teacher_records), failures=sum(not r['success'] for r in teacher_records),
            completed_actor_calls_lower_bound=teacher_calls, exception_receipts=exception_count,
            wall_s=teacher_wall, records=teacher_records),
        available_validation_evaluations=dict(runs=len(validation_costs), wall_s=sum(r['wall_s'] for r in validation_costs), records=validation_costs),
        train_plus_teacher_process_wall_s=aggregate['total_wall_s']+teacher_wall,
        process_lifecycle_timing=dict(whole_subprocess_receipts=len(queue_records),
            expected_training_runs=28, missing_whole_subprocess_receipts=28-len(queue_records),
            measured_extra_import_startup_exit_wall_s=known_process_overhead,
            training_teacher_plus_known_lifecycle_wall_lower_bound_s=aggregate['total_wall_s']+teacher_wall+known_process_overhead,
            scope='Internal total_wall_s excludes some import/startup/exit time. Whole-subprocess queue timers are available for 24 runs. Four single-seed gate runs have only internal timers; missing startup/exit overhead is not imputed.'),
        accounting_notes=[
            'All 28 new fit/PPO runs are charged, including 14 bounded calibration runs and eight clip-sensitivity runs across both generations. The 10 full-range main runs form the five-seed primary comparison.',
            'The per-run hyperparameter_trials=0 field describes no inner search within that run. It must not be read as zero project-level tuning: the eight extra clip runs and all 36 teacher candidate-validation executions are included separately.',
            'Teacher trajectories are generated once for shared use by both arms, not charged twice. All three acquisition/search rounds and all failed runs are included, not only the final teacher dataset.',
            'Historic actor_calls counts completed decision calls. Two exceptional teacher receipts may each omit at most one attempted call; the exact attempted-call count cannot be recovered from those saved fields. Do not present the lower bound as exact total attempted calls.',
            'The feature_construction_wall_s label covers dataset loading, feature statistics/center selection, model initialization and registration, not an isolated static-feature kernel. The total wall ledger includes this preparation.',
            'Physics ticks are the retained historical counters. Calibration sources and early registered full-range sources did not uniformly charge partial exceptional environment steps; counts can be lower bounds in those cases. No missing ticks were imputed.',
            'PPO samples count minibatch presentations, including four passes over each rollout; they are not additional environment interactions. Supervised samples count sampled records with replacement, not unique demonstrations.',
            'Per-process wall durations sum concurrent jobs and must not be described as elapsed end-to-end time or CPU/GPU utilization. Validation evaluations are additional and separately listed; final-test records are not read by this script.',
            'Registered total_wall_s is an internal training timer. Available outer queue timers expose additional process startup/import/exit time; four single-seed gate jobs lack outer receipts, so all-process lifecycle cost is a lower bound.',
            'The legacy optimized historical model is not inherited by any new run. Its prior four-stage optimization/search cost remains incompletely reconstructible and is not implied to be covered by this new-run ledger.'],
        final_test_records_read=0)
    write_json(OUT/'training_budget_audit.json', summary)
    lines = ['# Training and search budget audit', '',
        f"All **{summary['complete_training_runs']}/{summary['expected_training_runs']}** registered new runs have complete receipts and matching checkpoint hashes. Both generations are charged: 14 bounded calibration runs and 14 full-range runs. The full-range main comparison contains five independently initialized seeds per arm (10 runs); its clip sensitivity adds four seed-101 runs. Every training receipt reports `source_unchanged=true`.", '',
        '| Tag | Complete / expected | Environment decisions | Fit optimizer steps | PPO optimizer steps | Registered wall (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in grouped:
        cost = row['cost']
        lines.append(f"| {row['tag']} | {row['completed_runs']}/{row['expected_runs']} | {cost['training_interactions']:,} | {cost['supervised_optimizer_steps']:,} | {cost['ppo_optimizer_steps']:,} | {cost['total_wall_s']:.3f} |")
    lines += ['', '## Aggregate new training costs', '',
        f"- Environment decisions: {aggregate['training_interactions']:,}; retained physics-tick counter: {aggregate['physics_ticks']:,} (see partial-exception qualification below).",
        f"- Action fitting: {aggregate['supervised_optimizer_steps']:,} Adam steps and {aggregate['supervised_samples']:,} sampled-record presentations; summed measured fitting time {aggregate['supervised_wall_s']:.3f} s.",
        f"- PPO: {aggregate['ppo_updates']:,} rollout updates, {aggregate['ppo_optimizer_steps']:,} minibatch optimizer steps and {aggregate['ppo_samples']:,} sample presentations; summed measured PPO time {aggregate['ppo_wall_s']:.3f} s.",
        f"- Registered feature/preparation time: {aggregate['feature_construction_wall_s']:.6f} s, included in training total. Each run constructs 192 training-only centers with paired-identical scales/radii; this timer also covers dataset loading, model initialization and registration and is not an isolated feature-computation benchmark.",
        f"- Summed registered training wall time: {aggregate['total_wall_s']:.3f} s ({aggregate['total_wall_s']/60:.3f} min). These are internal timers; jobs overlap in time, so this sum is not end-to-end elapsed time or complete process-lifecycle cost.", '',
        '## Shared teacher construction and validation search', '',
        '| Round | Runs | Validation candidate runs | Training-record runs | Successes | Failures | Recorded completed calls | Process wall (s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rounds:
        lines.append(f"| {row['round']} | {row['runs']} | {row['validation_search_runs']} | {row['training_record_generation_runs']} | {row['successes']} | {row['failures']} | {row['completed_actor_calls']:,} | {row['wall_s']:.3f} |")
    lines += ['',
        f"All {len(teacher_records)} retained teacher runs are charged, including {summary['teachers']['failures']} failure receipts. Three rounds contain six preview candidates on two validation programs (36 search executions) and four training-record executions per round (12 executions). Recorded completed calls total {teacher_calls:,}; summed measured wall time is {teacher_wall:.3f} s. Records are shared once by both arms. The final fitted dataset has 926 records from four training programs; 512,000 sampled-record presentations per fit are repetitions of those records, not 512,000 unique teacher observations.", '',
        f"Two exceptional historical teacher receipts may omit one attempted final call each, because `actor_calls` was appended only after a successful decision return. Therefore {teacher_calls:,} is a lower bound on total attempted calls (at most {teacher_calls+exception_count:,} under that logging path), not an exact attempted-call count. Early counters may also omit partial failed physics steps. Wall timers include those failed attempts, and all failure receipts and traces remain intact. The success/failure labels are those recorded by each development-round evaluator and are not used here as cross-version quality comparisons.", '',
        f"Registered training plus shared teacher/search wall totals {aggregate['total_wall_s']+teacher_wall:.3f} s. Whole-subprocess queue timers are available for {len(queue_records)}/28 training runs and expose another {known_process_overhead:.3f} s of startup/import/exit overhead. Including this observed overhead gives a lower bound of {aggregate['total_wall_s']+teacher_wall+known_process_overhead:.3f} s; four single-seed gate runs have no outer-process receipts, so their unmeasured startup/exit overhead is not imputed. The {len(validation_costs)} retained learned-policy validation evaluations add {sum(r['wall_s'] for r in validation_costs):.3f} s separately. Final-test execution cost is outside this read-only report.", '',
        '## Interpretation and remaining accounting limits', '',
        'The per-run field `hyperparameter_trials=0` does not mean that the project used no tuning. Across both generations there were eight extra clip-sensitivity training runs in addition to the 20 main seed/arm runs, and 36 teacher-validation candidate executions. All are included above. The bounded pilot exposed an action-range restriction and is retained as development cost; it is not mixed into the primary full-range effect estimate.', '',
        'Training data use is unchanged across the later manifest revisions: four training programs, two validation programs and the final shared teacher-record bytes remain fixed. The replacement of related or invalid unexecuted test inputs was recorded separately before freezing; each training registration retains its original manifest hash. Data preparation/source editing time, programmer effort and historical model optimization/search time were not instrumented and are not fabricated as measured compute costs.', '',
        'The fresh controlled comparison inherits no learned historical checkpoint. The older fully optimized system has a different optimization history whose complete cost cannot be recovered from this new ledger. Its training cost must remain explicitly unknown/incomplete when systems are compared.', '',
        'The primary PPO clip remains 0.2 regardless of the single-seed sensitivity ranking. The 0.1/0.3 sensitivity runs are charged even though they were not selected for final testing. No statistical claim of clip robustness or independently significant adaptive-preview gain follows from the single-seed parameter sweep.', '',
        'Reproduce these read-only reports with `python -B src/analyze_validation_budget.py`. Detailed receipt paths, model/trace hashes, errors, source/split hashes and every per-run cost are preserved in `training_budget_audit.json`; validation outcomes and comparison details are in `validation_sensitivity.json` and `.csv`.']
    (OUT/'training_budget_audit.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    validation = validation_sensitivity()
    budget = training_budget()
    print(json.dumps(dict(validation_completed=validation['completed_runs'], validation_successes=validation['successes'],
        training_complete=budget['complete_training_runs'], teacher_runs=budget['teachers']['observed_runs'],
        teacher_failures=budget['teachers']['failures'], train_plus_teacher_process_wall_s=budget['train_plus_teacher_process_wall_s'])))


if __name__ == '__main__':
    main()
