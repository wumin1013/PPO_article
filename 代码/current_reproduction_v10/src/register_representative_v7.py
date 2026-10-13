"""Register a complete comparator grid for two selected detailed case studies."""
import json
from common import ROOT, sha, utc, write_json


def main():
    destination = ROOT / 'evidence/new_representative_v7_registration.json'
    if destination.exists():
        raise FileExistsError(destination)
    original = ROOT / 'evidence/frozen_main_generalization_v6.json'
    study = json.loads(original.read_text('utf-8'))
    selected = ['piecewise_curvature_coupon', 'asymmetric_neck_contour']
    methods = ['JNN', 'Fixed', 'S2', 'S2R', 'SCMO', 'COPP', 'NNC']
    cases = [next(x for x in study['cases'] if x['id'] == name) for name in selected]
    traces = []
    for cell in study['grid']:
        if cell['case'] in selected:
            traces.append(dict(cell, trace_sha256=sha(ROOT / cell['trace']),
                               record_sha256=sha(ROOT / cell['record'])))
    record = dict(schema=1, registered_utc=utc(), study_id='new_representative_v7',
        origin='synthetic', methods=methods, dt_s=[.001, .0005], cases=cases,
        case_selection='Author-selected detailed comparisons after inspecting frozen-controller completion in the five-program v6 study; not a new prospectively sampled generalization test.',
        selection_date='2026-10-11',
        scope='Complete comparative evaluation on two previously generated full programs; all registered methods and both resolutions retained.',
        original_five_program_registration=original.relative_to(ROOT).as_posix(),
        original_five_program_registration_sha256=sha(original),
        original_ten_outcomes_retained=True, frozen_main_traces=traces,
        frozen_main_model=study['model'], frozen_main_model_sha256=study['model_sha256'],
        no_retraining=True, no_new_case_tuning=True,
        common_acceptance=study['completion'],
        planar_vector_limits=study['planar_vector_limits'],
        nominal_tolerance_mm=study['nominal_tolerance_mm'],
        arc_handling=study['arc_handling'],
        failure_rule=study['failure_rule'],
        result_rule='Completion times and time savings exist only for accepted complete executions. Retain every baseline result, including faster comparators and incomplete attempts. Two resolutions are not independent training seeds.',
        grid=[dict(case=case, method=method, dt_s=dt)
              for case in selected for method in methods for dt in (.001, .0005)])
    write_json(destination, record)
    print(json.dumps({'registration':str(destination),'sha256':sha(destination),
                      'grid_cells':len(record['grid'])},ensure_ascii=False))


if __name__ == '__main__':
    main()
