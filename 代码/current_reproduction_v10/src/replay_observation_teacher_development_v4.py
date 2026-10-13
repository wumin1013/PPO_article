"""Test a fixed development teacher on all five previously observed programs."""
import json
import numpy as np
from common import ROOT, sha, utc, write_json
from diagnose_teacher_latency_v4 import LatencyTeacher, CANDIDATES
from runtime_resolution_fixed_v4e import install, VERSION
from evaluate import rollout


def main():
    output = ROOT / 'results/observation_teacher_observed_development_v4'
    registration = output / 'registration.json'
    candidate = next(c for c in CANDIDATES if c['id'] == 'observation_latency070')
    cases = sorted((ROOT / 'development_imports/observed_v3_final').glob('final_*.csv'))
    sources = ['replay_observation_teacher_development_v4.py', 'diagnose_teacher_latency_v4.py',
               'diagnose_teacher_taper_v4.py', 'runtime_resolution_fixed_v4e.py',
               'terminal_completion_v4d.py', 'evaluate.py', 'metrics.py']
    if registration.exists():
        raise FileExistsError(registration)
    write_json(registration, dict(created_utc=utc(), scope='DEVELOPMENT ONLY',
        candidate=candidate, runtime_version=VERSION, dt_s=.001, fixed_preview_mm=3.,
        cases=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in cases],
        sources={s: sha(ROOT / 'src' / s) for s in sources}, expected_runs=len(cases),
        criterion='Run every one of the previously observed v3 final programs without tuning or replacement. These are no longer independent tests.',
        action_source='Fixed 14-observation geometric teacher, no trained checkpoint.'))
    records = []
    for case in cases:
        install()
        teacher = LatencyTeacher(candidate)
        result = rollout(teacher, np.loadtxt(case, delimiter=',', skiprows=1), output / (case.stem + '.npz'),
                         fixed_d=3., runtime_version='v3', wall_s=180.)
        result.update(scope='DEVELOPMENT ONLY', runtime_version=VERSION,
                      registration_sha256=sha(registration), reference_sha256=sha(case),
                      teacher=candidate, path=case.stem)
        write_json(output / (case.stem + '.json'), result)
        records.append(dict(path=case.stem, success=result['success'],
                            complete_time_s=result['complete_time_s'], reason=result['reason']))
    write_json(output / 'summary.json', dict(records=records, successes=sum(r['success'] for r in records)))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    main()
