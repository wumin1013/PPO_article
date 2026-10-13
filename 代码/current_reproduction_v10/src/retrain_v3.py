"""Repeat all training-only stages in a new directory; preserve published runs."""
import argparse
import json
from pathlib import Path
from common import ROOT,sha,utc,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();target=args.output.resolve()
    if not target.is_relative_to(ROOT/'results') or target.exists():
        raise ValueError('Use a new output folder under results; existing evidence is never replaced')
    import unified_training_v3 as study
    study.OUT=target
    study.prepare()
    models=[]
    for seed in [101,211,307,401,503]:
        for arm in ['fixed','adaptive']:
            study.train(arm,seed)
            stage=target/'paired'/f'seed_{seed}'/arm
            receipt=json.loads((stage/'receipt.json').read_text('utf-8'))
            if not receipt['complete'] or not receipt['source_unchanged'] or not receipt['protected_sources_unchanged']:
                raise RuntimeError('Incomplete retraining; all produced prefixes/receipts remain '+str(stage))
            model=stage/'final.pt'
            models.append(dict(method=arm,seed=seed,file=model.relative_to(ROOT).as_posix(),sha256=sha(model)))
        if seed==101:
            for arm in ['fixed','adaptive']:
                validation=json.loads((target/'paired/seed_101'/arm/'validation_08192.json').read_text('utf-8'))
                if not all(r['success'] for r in validation['records']):raise RuntimeError('Independent seed-101 validation gate failed; retained evidence is not replaced')
    write_json(target/'models.json',dict(created_utc=utc(),models=models,independent_retraining=True))
    write_json(target/'retraining_complete.json',dict(created_utc=utc(),seeds=[101,211,307,401,503],
        arms=['fixed','adaptive'],main_source_sha256=sha(ROOT/'src/unified_training_v3.py'),
        final_test_used=False,selection='All final checkpoints retained; no retraining result replaces the original freeze.'))


if __name__=='__main__':main()
