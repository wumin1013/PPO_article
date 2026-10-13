"""Fresh PPO with an explicit soft corridor-margin reward; physics is unchanged."""
import ast
import inspect
import json
import sys
from pathlib import Path
import numpy as np
import train_mixed_resolution_ppo as common
from train_paired_supplement_ppo import SeparateFullEvaluation
from localized_two_action_policy import policy_from
from evaluate_historical_dense import ROOT, write, sha

THRESHOLD_MM = 1.20
WIDTH_MM = .05
WEIGHT = .1


def corridor_cost(env, rows, dt):
    distances = np.array([env.corridor.point_distance(row[:2]) for row in rows])
    if not len(distances):
        return 0., 0.
    penalty = WEIGHT*dt*np.square(np.maximum(distances-THRESHOLD_MM,0.)/WIDTH_MM).sum()
    return float(penalty), float(distances.max())


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old,new)


def worker_source():
    source = inspect.getsource(common.physics_worker)
    source = replace_once(source,"'physical_reward','corner_penalty_s']",
                          "'physical_reward','corner_penalty_s','corridor_penalty_s','corridor_distance_peak_mm']")
    anchor = "penalty=cost.consume(info['rows'])[0] if case=='original' and len(info['rows']) else 0."
    source = replace_once(source,anchor,anchor+"\n                        margin_cost,margin_peak=corridor_cost(env,info['rows'],dt)")
    source = replace_once(source,'values=[x,x2,z,logp,value,physical-penalty,next_value,done,',
                          'values=[x,x2,z,logp,value,physical-penalty-margin_cost,next_value,done,')
    source = replace_once(source,'target,physical,penalty]', 'target,physical,penalty,margin_cost,margin_peak]')
    return source


def physics_worker(connection,dt,args,stage,deadline):
    source = worker_source()
    (stage/'source'/f'margin_worker_{dt}.py').write_text(source,encoding='utf-8')
    namespace = dict(common.__dict__)
    namespace.update(policy_from=policy_from,corridor_cost=corridor_cost)
    exec(compile(source,__file__,'exec'),namespace)
    namespace['physics_worker'](connection,dt,args,stage,deadline)


def parent_namespace():
    source = Path(common.__file__).read_text(encoding='utf-8')
    source = replace_once(source,"batch['reward']+batch['corner_penalty_s']",
                          "batch['reward']+batch['corner_penalty_s']+batch['corridor_penalty_s']")
    transform = SeparateFullEvaluation()
    tree = transform.visit(ast.parse(source))
    assert transform.count == 1
    ast.fix_missing_locations(tree)
    namespace = dict(common.__dict__)
    namespace['__name__'] = 'margin_ppo_training_core'
    exec(compile(tree,common.__file__,'exec'),namespace)
    namespace.update(policy_from=policy_from,physics_worker=physics_worker)
    return namespace


def main():
    if sys.argv[1:] == ['--verify-transform']:
        ast.parse(worker_source())
        parent_namespace()
        class Corridor:
            def point_distance(self,point):
                return abs(float(point[1]))
        class Environment:
            corridor = Corridor()
        cost,peak = corridor_cost(Environment(),np.array([[0.,1.20],[0.,1.25]]),.001)
        assert abs(cost-.0001)<1e-14 and peak==1.25
        assert corridor_cost(Environment(),np.empty((0,4)),.001)==(0.,0.)
        print('MARGIN_TRANSFORM_AND_REWARD_CHECK_PASSED')
        return
    name = sys.argv[sys.argv.index('--name')+1]
    protocol = ROOT/'experimental_supplement_20260923/corridor_margin_main_ppo24/protocol.json'
    registered = json.loads(protocol.read_text(encoding='utf-8'))
    assert (registered['threshold_mm'],registered['width_mm'],registered['weight']) == (THRESHOLD_MM,WIDTH_MM,WEIGHT)
    assert registered['source_sha256'][str(Path(__file__).resolve())] == sha(__file__)
    stage = ROOT/name
    assert not stage.exists()
    binding = dict(command_arguments=sys.argv[1:],protocol=str(protocol),protocol_sha256=sha(protocol),
        wrapper_sha256=sha(__file__),worker_source=worker_source(),
        reward_identity='reward + corner_penalty_s + corridor_penalty_s = physical_reward',
        hard_corridor_mm=1.25,margin_threshold_mm=THRESHOLD_MM,width_mm=WIDTH_MM,weight=WEIGHT,
        auxiliary_weight_each=.25,no_added_policy_input=True,no_execution_rule_change=True,
        complete_evaluation_separate_from_training_clock=True)
    write(protocol.parent/(name+'_launch.json'),binding)
    parent_namespace()['main']()
    write(stage/'corridor_margin_protocol_binding.json',binding)
    (stage/'source'/Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')


if __name__ == '__main__':
    main()
