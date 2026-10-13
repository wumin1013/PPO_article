"""Launch the preserved project evaluator with a frozen checkpoint; no training."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root',type=Path,required=True,help='Existing compact_dense_research checkout with its bound dependencies')
    parser.add_argument('--method',choices=['J-NNC','Two-action'],required=True)
    parser.add_argument('--output',type=Path,required=True,help='New nonsynchronized output directory')
    parser.add_argument('--check-only',action='store_true',help='Check entry point and frozen dependencies; execute no physical steps')
    args=parser.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text('utf-8'))
    checkpoint=(HERE/manifest['models'][args.method]).resolve()
    for name,expected in manifest['evaluator_sources'].items():
        file=args.research_root/name
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Evaluator source mismatch: '+str(file))
    sys.path.insert(0,str(args.research_root.resolve()))
    from evaluate_historical_dense import verify_bound_inputs
    verify_bound_inputs()
    if args.check_only:
        print('Preserved evaluator sources and bound inputs verified; no physical steps executed.')
        return
    if args.output.exists():raise FileExistsError(args.output)
    if 'Nutstore' in str(args.output.resolve()):
        raise ValueError('Use a nonsynchronized output directory for execution records')
    command=[sys.executable,'-B',str(args.research_root/'evaluate_supplement_checkpoint.py'),
        '--checkpoint',str(checkpoint),'--output',str(args.output),'--seed','923202']
    subprocess.run(command,check=True)

if __name__=='__main__':main()
