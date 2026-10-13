"""CLI-only bridge for every already registered direct-explore variant.

The frozen shared CLI omitted selected_fixed_no_anchor from argparse choices.
This entry point uses the union of registered pilot/formal names and delegates
unchanged to the original study APIs. It never changes configuration, numerical
implementation, protocol or source hashes. Its own exact source is archived
separately beside each run, because it is outside the frozen runtime archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import preview_direct_explore as candidate


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','select','freeze'):commands.add_parser(name)
    train=commands.add_parser('train')
    train.add_argument('--variant',choices=list(dict.fromkeys((*candidate.PILOT_VARIANTS,*candidate.FORMAL_VARIANTS))),required=True)
    train.add_argument('--seed',type=int,required=True)
    train.add_argument('--stage',choices=['smoke','pilot','formal'],default='pilot')
    train.add_argument('--steps',type=int,default=candidate.study.PILOT_STEPS)
    for name in ('evaluate','eval'):
        evaluate=commands.add_parser(name)
        evaluate.add_argument('--run',type=Path,required=True)
        evaluate.add_argument('--split',choices=['validation','test','regression'],required=True)
        evaluate.add_argument('--regime',choices=['nominal','tight','low_jerk'],default='nominal')
    return parser.parse_args(argv)


def archive_launcher(study,args,argv):
    study.check_protocol()
    run=study.OUT/args.stage/f'{args.variant}_seed{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).read_bytes()
    metadata=dict(scope='CLI-only bridge; unchanged frozen runtime and training configuration',
        original_failure='argparse choices omitted already registered selected_fixed_no_anchor',
        launcher_sha256=hashlib.sha256(source).hexdigest(),
        protocol_sha256=study.backend.sha(study.OUT/'protocol.json'),
        runtime_source_sha256=study.source_state()['source_sha256'])
    files={'launcher_source.py':source,'launcher_provenance.json':(json.dumps(metadata,indent=2)+'\n').encode('utf-8')}
    for name,content in files.items():
        path=run/name
        if path.exists() and path.read_bytes()!=content:
            raise ValueError(f'Preserve previous CLI bridge source/provenance: {path}')
        if not path.exists():path.write_bytes(content)
    with (run/'launcher_invocations.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            arguments=list(argv),launcher_sha256=metadata['launcher_sha256']))+'\n')


def main(argv=None):
    actual=list(sys.argv[1:] if argv is None else argv)
    args=parse_args(actual)
    study=candidate.install();study.activate_backend()
    if args.command=='train':
        archive_launcher(study,args,actual)
        print(study.train(args.variant,args.seed,args.stage,args.steps),flush=True)
    elif args.command in ('evaluate','eval'):
        study.evaluate(args.run,args.split,args.regime)
    else:print(getattr(study,args.command)(),flush=True)


if __name__=='__main__':main()
