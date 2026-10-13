"""Summarize already measured v8 learned costs; perform no new experiments."""
import json
from common import ROOT, sha, write_json


def main():
    registration_path=ROOT/'results/representative_scaled_v8/learned/registration.json'
    registration=json.loads(registration_path.read_text('utf-8'))
    rows=[]
    for spec in registration['runs']:
        if spec['method'] not in ('JNN','FIXED'):
            continue
        path=ROOT/spec['record'];record=json.loads(path.read_text('utf-8'))
        assert record['success'] and record['registration_sha256']==sha(registration_path)
        assert record['actor_calls']==record['completed_decisions']>0
        cost=record['cost']
        rows.append(dict(case=spec['case'],method=spec['method'],dt_s=spec['dt_s'],
            record=spec['record'],record_sha256=sha(path),actor_calls=record['actor_calls'],
            completed_decisions=record['completed_decisions'],wall_s=record['wall_s'],
            process_cpu_s=record['process_cpu_s'],cost=cost,
            mean_actor_ms=1000*cost['actor_wall_s']/record['actor_calls'],
            mean_executor_step_ms=1000*cost['executor_wall_s']/record['completed_decisions']))
    assert len(rows)==8
    output=ROOT/'audit/scaled_representative_v8'
    output.mkdir(parents=True,exist_ok=True)
    main_times=[row['mean_actor_ms'] for row in rows if row['method']=='JNN']
    scope=dict(actor='policy.act(observation, True): forward evaluation and tensor/NumPy conversion; observation construction excluded.',
        executor='Full env.step(action), including every held command, geometry and kinematic execution within the 20 ms decision.',
        wall='Environment initialization, observation/control loop, compressed trace save and independent metrics; excludes prior worker model loading/runtime installation and subsequent trace hashing/final JSON writes.',
        process_cpu='CPU clock around rollout, including rollout setup and its first output write; model load occurs earlier.',
        audit='No separate audit clock was recorded; the wall-time residual is not pure audit time.',
        statistics='Range across complete-program per-run means, not a tail-latency or worst-case bound.',
        excluded='Four NNC prefixes are retained in the execution study; failed first-command runs do not supply complete-program throughput estimates.',
        comparison='Traditional full-pipeline wall time and learned actor-only time have different boundaries and are not a timing race.')
    write_json(output/'learned_cost_summary.json',dict(registration_sha256=sha(registration_path),
        new_experiments=False,hard_realtime_demonstrated=False,scope=scope,rows=rows,
        jnn_actor_mean_range_ms=[min(main_times),max(main_times)]))
    prose=('On the shared CPU host, mean J-NNC actor evaluation took '
        f'{min(main_times):.2f}--{max(main_times):.2f} ms per decision across the two command periods; '
        'these measured averages do not establish hard real-time execution.\n')
    (output/'learned_cost_prose.tex').write_text(prose,encoding='utf-8')
    print(prose,end='')


if __name__=='__main__':
    main()
