"""Run the unchanged local diagnostic for an explicitly registered failed actor."""
import argparse
import json
import sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--evaluation',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--dt',type=float,choices=[.001,.0005],required=True)
options=parser.parse_args()
registration=json.loads((options.evaluation/'registration.json').read_text(encoding='utf-8'))
registered_checkpoint=Path(registration['checkpoint'])
key='1ms' if options.dt==.001 else 'half_ms'
traces=list(options.evaluation.glob(f'eval_*_dense_{key}.npz'))
assert len(traces)==1
registered_trace=traces[0]
registered_output=options.output.resolve()
source=Path(__file__).with_name('probe_supplement_dense_prefix.py')
code=source.read_text(encoding='utf-8')
replacements={
    "parent=ROOT/f'supplement_margin_main_{args.seed}_ppo24'": "parent=registered_checkpoint.parent",
    "checkpoint=parent/'checkpoint_0024.pt'": "checkpoint=registered_checkpoint",
    "trace=parent/f'evaluation/eval_0024_dense_{key}.npz'": "trace=registered_trace",
    "stage=ROOT/'experimental_supplement_20260923/dense_prefix_recovery_probe'/f'seed_{args.seed}'/key": "stage=registered_output",
    "[Path(__file__),checkpoint,trace,": "[Path(__file__),source,checkpoint,trace,",
}
for old,new in replacements.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
sys.argv=[str(Path(__file__)),'--seed',str(options.seed),'--dt',str(options.dt)]
exec(compile(code,str(source),'exec'))
