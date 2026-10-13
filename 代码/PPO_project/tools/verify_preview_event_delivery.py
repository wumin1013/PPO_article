"""Read-only final artifact, command-trajectory and manuscript consistency audit."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import yaml
import preview_event_study as study
from preview_event_nnc_transfer import require_freeze
from src.utils.revision_metrics import evaluate_trajectory


def differences(actual,expected,prefix=''):
    result=[]
    if isinstance(expected,dict):
        if not isinstance(actual,dict):return [prefix]
        for k,v in expected.items():result.extend(differences(actual.get(k),v,prefix+'.'+k))
    elif isinstance(expected,float):
        if actual is None or not math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-7):result.append(prefix)
    elif actual!=expected:result.append(prefix)
    return result


def manuscript_check():
    paper=study.ROOT/'论文项目';main=paper/'main.tex';bibliography=paper/'references.bib'
    visited={};missing=[]
    def visit(path):
        if path in visited:return
        if not path.exists():missing.append(str(path));return
        content=path.read_text(encoding='utf-8');visited[path]=content
        uncommented=re.sub(r'(?<!\\)%[^\n]*','',content)
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',active_file_branches(uncommented,paper)):
            if '\\' in name or '#' in name:continue
            child=paper/name
            if not child.suffix:child=child.with_suffix('.tex')
            visit(child)
    visit(main)
    keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bibliography.read_text(encoding='utf-8')))
    citations=set()
    for content in visited.values():
        content=active_file_branches(re.sub(r'(?<!\\)%[^\n]*','',content),paper)
        for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',content):citations.update(k.strip() for k in group.split(','))
    log=(paper/'main.log').read_text(encoding='utf-8',errors='replace') if (paper/'main.log').exists() else ''
    return dict(tex=main.relative_to(study.ROOT).as_posix(),tex_sha256=study.sha(main),pdf_sha256=study.sha(paper/'main.pdf'),
        source_files=len(visited),cited_keys=len(citations),missing_citations=sorted(citations-keys),missing_inputs=missing,
        undefined_reference_warnings=re.findall(r'^.*(?:undefined|multiply defined).*$' ,log,flags=re.M),
        overfull_boxes=re.findall(r'^Overfull.*$',log,flags=re.M),bibliography_sha256=study.sha(bibliography))


def active_file_branches(content,paper):
    """Resolve the manuscript's standard three-brace IfFileExists guards."""
    marker='\\IfFileExists';start=content.find(marker)
    while start>=0:
        cursor=start+len(marker);arguments=[]
        for _ in range(3):
            while cursor<len(content) and content[cursor].isspace():cursor+=1
            if cursor>=len(content) or content[cursor]!='{':raise ValueError('Unsupported IfFileExists syntax')
            began=cursor+1;depth=1;cursor+=1
            while cursor<len(content) and depth:
                if content[cursor] in '{}' and content[cursor-1]!='\\':depth+=1 if content[cursor]=='{' else -1
                cursor+=1
            if depth:raise ValueError('Unbalanced conditional')
            arguments.append(content[began:cursor-1])
        file,yes,no=arguments
        branch=yes if (paper/file).exists() else no
        content=content[:start]+branch+content[cursor:]
        start=content.find(marker)
    return content


def audit(split,regime):
    records=[];seeds=(11,) if regime!='nominal' else study.SEEDS
    if split=='regression':
        old=yaml.safe_load((study.ROOT/'论文项目/generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        references={p['name']:study.backend._build_path(p) for p in old['training']['path_curriculum']['paths']}
    else:references={p['id']:study.points(p['id']) for p in study.manifest()['paths'] if p['split']==split}
    for variant in study.FORMAL:
        for seed in seeds:
            run=study.OUT/'formal'/f'{variant}_seed{seed}';cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
            if regime=='tight':cfg['environment']['epsilon']=1.25
            if regime=='low_jerk':cfg['kinematic_constraints']['MAX_JERK']=31250.
            for path_id,reference in references.items():
                path=run/f'evaluation_{split}_{regime}'/f'{path_id}.json';row=study.read(path)
                study.validate_result(row,run,path_id,split,regime)
                trace=path.with_suffix('.npz');z=np.load(trace);columns=z['columns'].tolist();data=z['data'];get=lambda k:data[:,columns.index(k)]
                independent=evaluate_trajectory(reference,np.c_[get('x_mm'),get('y_mm')],get('timestamp_s'),
                    tolerance_mm=cfg['environment']['epsilon']/2,limits=cfg['kinematic_constraints'],
                    termination_reason=row['termination_reason'],claimed_success=row['claimed_success'],
                    initial_velocity=[0.,0.],initial_acceleration=[0.,0.],initial_heading=float(get('heading_rad')[0]),
                    initial_angular_velocity=0.,initial_angular_acceleration=0.,witness_spacing_mm=.01)
                mismatch=differences(row,independent)
                events=np.flatnonzero(get('preview_event'))
                expected=np.arange(1,len(data),cfg['revision']['preview_period_steps']) if cfg['revision']['adaptive_preview'] else np.array([],int)
                if not np.array_equal(events,expected):mismatch.append('preview_event_schedule')
                changed=np.flatnonzero(np.diff(get('lookahead_mm'))!=0)+1
                if cfg['revision']['adaptive_preview'] and not set(changed).issubset(set(events)):mismatch.append('held_preview_changed_off_event')
                if not cfg['revision']['adaptive_preview'] and not np.allclose(get('lookahead_mm'),cfg['revision']['fixed_distance_mm'],atol=1e-12,rtol=0):mismatch.append('fixed_preview_changed')
                records.append(dict(result=path.relative_to(study.ROOT).as_posix(),result_sha256=study.sha(path),trace_sha256=study.sha(trace),
                    checked_samples=len(data),strict_success=row['success'],mismatches=mismatch))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manuscript-only',action='store_true');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous audit; choose a new output file')
    records=[]
    if not args.manuscript_only:
        require_freeze(study)
        for split,regime in [('validation','nominal'),('test','nominal'),('regression','nominal'),('test','tight'),('test','low_jerk')]:
            records.extend(audit(split,regime))
    manuscript=manuscript_check()
    result=dict(command='python '+' '.join(sys.argv),script_sha256=study.sha(__file__),
        coverage='Formal PPO command trajectories and manuscript only; auxiliary planner/NNC/inference outputs have separate audit manifests.',
        numerical_source_sha256=study.source_state()['source_sha256'],trajectories=records,manuscript=manuscript,
        mismatch_count=sum(bool(r['mismatches']) for r in records))
    study.write(args.output,result)
    print(json.dumps(dict(trajectories=len(records),mismatches=result['mismatch_count'],manuscript=manuscript),ensure_ascii=False))
    if result['mismatch_count'] or manuscript['missing_citations'] or manuscript['missing_inputs'] or manuscript['undefined_reference_warnings']:raise SystemExit(1)


if __name__=='__main__':main()
