"""Bounded counterexample and geometry coverage checks; no learned policy calls."""
import json
import math
from pathlib import Path
import numpy as np
from event_observation_candidate import factory, event_observation
from study_io import BASE, ROOT, register, receipt, write_json


def geometry_coverage(reference):
    lengths = np.linalg.norm(np.diff(reference, axis=0), axis=1)
    arcs = np.r_[0., np.cumsum(lengths)]
    rows = []
    # Midpoints are reference-space probes, not the occupancy of a learned policy.
    for distance in (3., 6., 12.):
        starts = arcs[:-1] + .5*lengths
        targets = np.minimum(np.searchsorted(arcs, starts+distance, side='right')-1, len(lengths)-1)
        crossed = targets-np.arange(len(lengths))
        rows.append(dict(distance_mm=distance,
            fraction_segment_midpoints_beyond_two_segments=float(np.mean(crossed>1)),
            arclength_weighted_fraction=float(np.sum(lengths*(crossed>1))/sum(lengths)),
            maximum_segments_ahead=int(max(crossed))))
    return dict(length_mm=float(arcs[-1]), segments=len(lengths), horizons=rows)


def main():
    refs = ROOT/'论文项目/revision_20260913_preview_efficiency_v2/geometry_design_v1/references'
    long_files = [refs/(name+'.csv') for name in
                  ('mixed_lengths_large','dense_corners_large','closed_line_arc_large')]
    sources = [Path(__file__), BASE/'event_observation_candidate.py',
        BASE/'compact_observation_candidate.py', BASE/'joint_runtime.py',
        BASE/'joint_runtime_v2.py', BASE/'joint_runtime_v3.py', BASE/'joint_runtime_v5.py',
        BASE/'terminal_candidate_runtime_v1.py', BASE/'geometry_spec.py',
        BASE/'learning_preparation_v4/paths.json', BASE/'event_horizon_information_probe_v1/failure.json', *long_files]
    stage = register('event_horizon_information_probe_v2', sources,
        dict(environment_constructions=81, physical_macros=2, physical_ticks_upper=4240,
             reference_geometry_paths=79, state_probes_upper=395, neural_calls=0, training_decisions=0),
        'Test immediate command aliasing beyond two events and measure train/development/known-long geometry coverage; no sealed test or performance evaluation.')
    references = [np.asarray([[-20.,0.],[0.,0.],[2.,0.],[4.,0.],[4.,sign*20.]]) for sign in (1.,-1.)]
    envs = [factory(ref) for ref in references]
    for env in envs:
        env.position[:] = [-.5,0.]
        env.tracker.position = env.position.copy()
        env.velocity[:] = [40.,0.]
    diffs = dict(events16=float(np.max(abs(event_observation(envs[0])-event_observation(envs[1])))),
        events_far18=float(np.max(abs(event_observation(envs[0],True)-event_observation(envs[1],True)))),
        original72=float(np.max(abs(envs[0].observation()-envs[1].observation()))))
    action = np.array([0., math.atanh(-.3), math.atanh((8.-6.4)/5.6)])
    steps = [env.step(action) for env in envs]
    rows = [step[3]['rows'] for step in steps]
    alias = dict(references=[ref.tolist() for ref in references], position=[-.5,0.],
        velocity=[40.,0.], decoded_action=envs[0].held.tolist(), observation_max_differences=diffs,
        first_requested_velocity_difference=float(np.linalg.norm(rows[0][0,8:10]-rows[1][0,8:10])),
        first_executed_velocity_difference=float(np.linalg.norm(rows[0][0,2:4]-rows[1][0,2:4])),
        endpoint_separation_mm=float(np.linalg.norm(envs[0].position-envs[1].position)),
        terminal_reasons=[env.reason for env in envs],
        interpretation='Equal events16 and equal allowed action yield different immediate requests and executed motion. Far18 resolves this pair only; no global sufficiency or different optimal action is proved.')
    assert diffs['events16']==0. and min(diffs['events_far18'],diffs['original72'])>.1
    assert alias['first_requested_velocity_difference']>0. and alias['endpoint_separation_mm']>0.
    data=json.loads((BASE/'learning_preparation_v4/paths.json').read_text(encoding='utf-8'))
    paths=[(split,str(i),np.asarray(ref)) for split in ('training','validation') for i,ref in enumerate(data[split])]
    paths += [('known_long_development',p.stem,np.loadtxt(p,delimiter=',',skiprows=1)) for p in long_files]
    records=[]
    for split,name,ref in paths:
        env=factory(ref)
        obs=[]
        for fraction in (.0,.25,.5,.75,.99):
            target=env.corridor.arcs[-1]*fraction
            index=min(int(np.searchsorted(env.corridor.arcs,target,side='right')-1),len(env.corridor.lengths)-1)
            env.position=env.corridor.starts[index]+(target-env.corridor.arcs[index])*env.corridor.tangents[index]
            env.tracker.position=env.position.copy()
            env.tracker.frontier=index
            env.heading=math.atan2(*env.corridor.tangents[index][::-1])
            obs.append(event_observation(env,True))
        values=np.stack(obs)
        assert np.isfinite(values).all()
        records.append(dict(split=split,path=name,**geometry_coverage(ref),
            probe_feature_min=values.min(axis=0).tolist(),probe_feature_max=values.max(axis=0).tolist()))
    write_json(stage/'results.json',dict(immediate_alias=alias,geometry=records,
        probe_semantics='Reference-space points with zero motion; not reachable-state or trained occupancy statistics.',
        choice='Do not advance events16 directly to learning: its two-event horizon is insufficient even for current allowed commands. Retain negative result; require physical-distance information in any next compact candidate.',
        training_decisions=0,goal_complete=False))
    receipt(stage,dict(status='completed_with_events16_immediate_command_alias',
        environment_constructions=len(paths)+2, physical_macros=2, physical_ticks=sum(len(r) for r in rows),
        geometry_paths=len(paths), state_probes=5*len(paths), neural_calls=0,training_decisions=0))
    print(json.dumps(alias,ensure_ascii=False))
    for split in ('training','validation','known_long_development'):
        selected=[r for r in records if r['split']==split]
        print(split, [(r['path'],r['segments'],r['horizons'][-1]['arclength_weighted_fraction']) for r in selected] if split=='known_long_development' else
              dict(paths=len(selected),mean_path_horizon12_beyond_two=float(np.mean([r['horizons'][-1]['arclength_weighted_fraction'] for r in selected]))))


if __name__=='__main__':
    main()
