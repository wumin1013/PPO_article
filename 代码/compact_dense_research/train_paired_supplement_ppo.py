"""Shared paired PPO with registered auxiliary weight and separate evaluation."""
import argparse
import ast
import json
import sys
from pathlib import Path
import train_mixed_resolution_ppo as common
import learning_update_coherent_anchor as learning
from localized_two_action_policy import policy_from
from evaluate_historical_dense import ROOT,write,sha


def physics_worker(connection,dt,args,stage,deadline):
    common.policy_from=policy_from
    common.physics_worker(connection,dt,args,stage,deadline)


class AuxiliaryWeight(ast.NodeTransformer):
    def __init__(self,weight):self.count=0;self.weight=weight
    def visit_Constant(self,node):
        if isinstance(node.value,float) and node.value==.25:
            self.count+=1
            return ast.copy_location(ast.Constant(value=self.weight),node)
        return node


class SeparateFullEvaluation(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_If(self,node):
        if ast.unparse(node.test)=='eligible and time.perf_counter() < deadline - 400.0':
            self.count+=1
            node.test=ast.Constant(value=False)
        return self.generic_visit(node)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--paired-protocol',type=Path,default=ROOT/'experimental_supplement_20260923/paired_action_ablation/protocol.json')
    parser.add_argument('--auxiliary-weight',type=float,choices=[0.,.25],default=0.)
    extra,remaining=parser.parse_known_args()
    protocol=extra.paired_protocol
    registered=json.loads(protocol.read_text(encoding='utf-8'))
    assert extra.auxiliary_weight==registered.get('ppo_auxiliary_weight_each',registered.get('auxiliary_target_loss_weight'))
    sys.argv=[sys.argv[0],*remaining]
    # Spawn children call this module's original import, not the patched parent
    # function, so the worker factory remains picklable and nonrecursive.
    transformed=AuxiliaryWeight(extra.auxiliary_weight)
    tree=transformed.visit(ast.parse(Path(learning.__file__).read_text(encoding='utf-8')))
    assert transformed.count==2
    ast.fix_missing_locations(tree)
    namespace=dict(learning.__dict__)
    exec(compile(tree,learning.__file__,'exec'),namespace)
    learning.update=namespace['update']
    transform=SeparateFullEvaluation()
    tree=transform.visit(ast.parse(Path(common.__file__).read_text(encoding='utf-8')))
    assert transform.count==1
    ast.fix_missing_locations(tree)
    namespace=dict(common.__dict__)
    namespace['__name__']='paired_common_training'
    exec(compile(tree,common.__file__,'exec'),namespace)
    namespace['policy_from']=policy_from
    namespace['physics_worker']=physics_worker
    name=sys.argv[sys.argv.index('--name')+1]
    stage=ROOT/name
    assert not stage.exists()
    write(protocol.parent/(name+'_launch.json'),dict(
        command_arguments=sys.argv[1:],wrapper_sha256=sha(__file__),
        protocol=str(protocol),protocol_sha256=sha(protocol),
        auxiliary_weight_each=extra.auxiliary_weight,long_route_checks_separate_from_training_clock=True,
        original_sampler_sha256=sha(common.__file__),original_update_sha256=sha(learning.__file__)))
    namespace['main']()
    write(stage/'paired_protocol_binding.json',dict(
        wrapper_sha256=sha(__file__),original_sampler_sha256=sha(common.__file__),
        original_update_sha256=sha(learning.__file__),
        protocol=str(protocol),protocol_sha256=sha(protocol),
        auxiliary_weight_each=extra.auxiliary_weight,long_route_checks_separate_from_training_clock=True,
        reason='Equal actor-update/decision budgets in both arms; complete evaluation is an unconditional separate stage, not omitted.',
        no_physical_relaxation=True))
    (stage/'source'/'train_paired_supplement_ppo.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')


if __name__=='__main__':main()
