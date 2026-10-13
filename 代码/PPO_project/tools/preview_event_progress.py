"""Read logged training progress; never infer pending experiment results."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'论文项目/revision_20260909_event_preview/study'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--record',action='store_true');args=parser.parse_args()
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));rows=[]
    for variant in protocol['formal_variants']:
        for seed in protocol['formal_seeds']:
            run=OUT/'formal'/f'{variant}_seed{seed}';steps=0
            if (run/'training_summary.json').exists():steps=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))['steps']
            elif (run/'console.log').exists():
                matches=re.findall(r'steps=(\d+)/(\d+) episodes=',(run/'console.log').read_text(encoding='utf-8'))
                if matches:steps=int(matches[-1][0])
            rows.append(dict(variant=variant,seed=seed,logged_steps=steps,training_complete=steps==protocol['formal_steps'],
                validation_complete=(run/'evaluation_validation_nominal/summary.json').exists()))
    logged=sum(r['logged_steps'] for r in rows);budget=len(rows)*protocol['formal_steps']
    result=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),logged_steps=logged,registered_steps=budget,
        logged_progress_percent=100*logged/budget,trained_models=sum(r['training_complete'] for r in rows),
        validation_complete_models=sum(r['validation_complete'] for r in rows),model_count=len(rows),rows=rows,
        note='Log progress only, not a provenance audit or convergence statement; up to one episode of active steps may not yet be logged.')
    if args.record:
        with (OUT/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))


if __name__=='__main__':main()
