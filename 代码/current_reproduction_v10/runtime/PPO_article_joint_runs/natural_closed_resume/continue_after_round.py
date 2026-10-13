"""Continue only an audited budget-limited prefix after its round has closed."""
from pathlib import Path
import argparse,json,subprocess,sys,time

ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--resolution',action='store_true');args=ap.parse_args()
stage=Path(args.stage);script=Path(__file__).resolve().parent
while not (stage/'receipt.json').exists():time.sleep(5)
result_file=stage/('result.json' if args.resolution else 'full_result.json')
if not result_file.exists():
    print('No full evaluation result; nothing continued.');sys.exit(0)
r=json.loads(result_file.read_text(encoding='utf-8'))
if 'TimeoutError' not in str(r.get('error')):
    print('No budget-only interruption; nothing continued.');sys.exit(0)
if not args.resolution and not (stage/'selected_policy.json').exists():
    reg=json.loads((stage/'registration.json').read_text(encoding='utf-8'))
    parent=json.loads((stage.parent/'selected_policy.json').read_text(encoding='utf-8'))
    parent.update(spec=reg['spec'],probe=reg.get('probe'),selected_from='existing_registered_intervention')
    (stage/'selected_policy.json').write_text(json.dumps(parent,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
target='resume_resolution_candidate.py' if args.resolution else 'resume_full_candidate.py'
subprocess.run([sys.executable,'-B',str(script/target),str(stage)],check=True)
