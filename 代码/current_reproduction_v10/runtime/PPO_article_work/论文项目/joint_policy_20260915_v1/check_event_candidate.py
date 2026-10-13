"""New candidate checks; reuse completed sparse-window counterexample evidence."""
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json
from event_observation_candidate import factory


def main():
    stage = register('event_observation_contract_v1',
        [Path(__file__), BASE/'event_observation_candidate.py', BASE/'compact_observation_candidate.py',
         BASE/'compact_information_counterexamples_v1/results.json', BASE/'learning_preparation_v4/paths.json'],
        dict(environment_constructions=81, physical_macros=3, physical_ticks_upper=6360,
             neural_calls=0, training_decisions=0),
        'Check whether local event descriptors resolve the previously recorded sparse-window alias and preserve physical/reward semantics; finite schema and geometry coverage, not a sufficiency theorem.')
    old = json.loads((BASE/'compact_information_counterexamples_v1/results.json').read_text())
    envs = [factory(np.asarray(ref),representation='events16') for ref in old['geometry_alias']['references']]
    for env in envs:
        env.position[:] = [-.5,0.]
        env.tracker.position = env.position.copy()
        env.velocity[:] = [40.,0.]
    difference = float(np.max(abs(envs[0].observation()-envs[1].observation())))
    assert difference > .1
    paths = json.loads((BASE/'learning_preparation_v4/paths.json').read_text())
    representations = []
    for ref in paths['training']+paths['validation']:
        env = factory(np.asarray(ref),representation='events16')
        obs = env.observation()
        assert obs.shape == (16,) and np.isfinite(obs).all()
        representations.append(obs)
    # Separate reward-only equivalence and observation-only transition equality.
    ref = np.asarray(paths['validation'][1])
    original = factory(ref)
    compact = factory(ref,representation='events16')
    unshaped = factory(ref,reward_mode='unshaped')
    assert np.array_equal(original.observation(),unshaped.observation())
    action = np.array([.02,.1,-.7])
    before = original.potential()
    _,r0,d0,i0 = original.step(action)
    _,r1,d1,i1 = compact.step(action)
    _,r2,d2,i2 = unshaped.step(action)
    after = 0. if d0 else original.potential()
    assert np.array_equal(i0['rows'],i1['rows']) and np.array_equal(i0['rows'],i2['rows'])
    assert r0 == r1 and abs(r2-(r0-after+before)) < 1e-12 and d0==d1==d2
    result = dict(event16_resolves_recorded_sparse18_alias=True,
        observation_difference=difference, finite_schema_paths=len(representations),
        initial_feature_ranges=np.stack([np.min(representations,axis=0),np.max(representations,axis=0)]).tolist(),
        state_only_physical_and_reward_equal=True,reward_only_physical_and_observation_equal=True,
        reward_only_difference_equals_removed_potential=True,
        caveat='Only the recorded alias and registered path initial states were checked; distant geometry, fine arc discretization, and value aliasing still require learning/generalization evidence.',
        training_decisions=0,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_event_observation_contract_checks',
        environment_constructions=81, physical_macros=3,
        physics_ticks=sum(len(i['rows']) for i in (i0,i1,i2)),model_calls=0,training_decisions=0))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
