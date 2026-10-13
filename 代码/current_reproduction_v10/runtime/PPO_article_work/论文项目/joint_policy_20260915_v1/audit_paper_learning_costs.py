"""Inventory actual checkpoint composition and receipt costs without replaying research."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import time
import numpy as np
import torch
from study_io import BASE, ROOT, register, receipt, sha, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def enclosing_stage(path):
    for parent in path.parents:
        if parent == ROOT:
            return None
        if (parent/'registration.json').is_file():
            return parent
    return None


def closure(start):
    nodes, edges, outside = {}, [], []
    queue = [start]
    while queue:
        folder = queue.pop()
        if folder in nodes:
            continue
        assert len(nodes) < 1500, 'Declared dependency closure exceeded audit bound'
        regfile=folder/'registration.json'; reg=read(regfile); recfile=folder/'receipt.json'
        rec=read(recfile) if recfile.is_file() else None
        nodes[folder]=dict(registration=reg, receipt=rec,
            registration_sha256=sha(regfile), receipt_sha256=sha(recfile) if rec else None,
            receipt_registration_hash_matches=(rec.get('registration_sha256') == sha(regfile)) if rec else None)
        for source, expected in reg.get('source_sha256',{}).items():
            p=Path(source); p=p if p.is_absolute() else ROOT/p
            if not p.is_relative_to(ROOT):
                outside.append(source); continue
            parent=enclosing_stage(p)
            if parent is not None and parent != folder:
                edges.append(dict(source_stage=folder, input_stage=parent,
                                  input_file=p, expected_sha256=expected,
                                  model_input=p.suffix.lower() in ('.pt','.pth')))
                queue.append(parent)
    model_nodes=set(); queue=[start]
    while queue:
        folder=queue.pop()
        if folder in model_nodes: continue
        model_nodes.add(folder)
        queue.extend(e['input_stage'] for e in edges if e['source_stage']==folder and e['model_input'])
    return nodes, edges, model_nodes, outside


def numeric_fields(value, prefix=''):
    result={}
    for k,v in value.items():
        key=prefix+k
        if isinstance(v,(int,float)) and not isinstance(v,bool):result[key]=v
        elif isinstance(v,dict) and k=='costs':result.update(numeric_fields(v,key+'.'))
    return result


def model_inventory(md):
    blocks=[]
    def walk(x,path='root',depth=0):
        if not isinstance(x,dict):return
        if 'architecture' in x:
            shapes={k:list(v.shape) for k,v in x.items() if isinstance(v,(np.ndarray,torch.Tensor))}
            blocks.append(dict(path=path,depth=depth,architecture=x['architecture'],
                learning_method=x.get('learning_method'),array_shapes=shapes,
                runtime_decision_ms=x.get('action_interval_ms',x.get('decision_ms')),
                runtime_execution_ms=x.get('physical_tick_ms',x.get('execution_ms'))))
        for k,v in x.items():
            if isinstance(v,dict) and k not in ('state_dict', 'actor', 'critic'):
                walk(v,path+'/'+k,depth+1)
    walk(md)
    return blocks


def aggregate_costs(nodes, subset):
    fields=defaultdict(list); missing=[]; statuses=Counter(); resume_flags=[]
    for folder in subset:
        entry=nodes[folder]; rec=entry['receipt']
        if rec is None:missing.append(str(folder.relative_to(ROOT)));continue
        statuses[rec.get('status','unspecified')]+=1
        for k,v in numeric_fields(rec).items():fields[k].append(v)
        purpose=entry['registration'].get('purpose','').lower()
        if 'resume' in purpose or 'recovery' in rec.get('status','').lower() or 'continuation' in purpose:
            resume_flags.append(str(folder.relative_to(ROOT)))
    return dict(stages=len(subset), receipts=len(subset)-len(missing), missing_receipts=missing,
        receipt_status_counts=dict(statuses), possible_cumulative_or_recovery_receipts=resume_flags,
        reported_field_totals={k:dict(sum=sum(v),receipts_reporting=len(v)) for k,v in fields.items()},
        interpretation='Each original field is summed separately once per stage. These are receipt-book totals, not a deduplicated physical-execution total, a single training run, or fair matched budgets. Missing fields are unknown, not zero. Broad dependencies include comparisons and diagnostics.')


def main():
    review=BASE/'paper_evidence_review'; frozen=read(review/'frozen_evaluation_candidate.json')
    model=BASE/frozen['model']; assert sha(model)==frozen['sha256']
    root_stage=model.parent
    stage=register('paper_evidence_review/learning_costs',
        [__file__,review/'frozen_evaluation_candidate.json',review/'receipt.json',
         root_stage/'receipt.json',root_stage/'registration.json'],
        dict(new_training_updates=0,new_simulation_ticks=0,new_actor_calls=0,maximum_dependency_stages=1500,wall_s=120),
        'Audit registered provenance, model-input dependencies and actual checkpoint composition only. '
        'Keep PPO, offline fitting, parameter search, validation and physical diagnostics distinct. '
        'Do not treat planned ceilings as actual cost or the full dependency graph as a single training run. '
        'Do not open sealed coordinates, instantiate an actor, optimize, or repeat completed experiments.')
    started=time.perf_counter(); nodes,edges,model_nodes,outside=closure(root_stage)
    md=torch.load(model,map_location='cpu',weights_only=False)
    blocks=model_inventory(md)
    architecture_counts=dict(Counter(x['architecture'] for x in blocks))
    write_json(stage/'checkpoint_structure.json',dict(model=str(model.relative_to(BASE)),model_sha256=sha(model),
        bytes=model.stat().st_size,architecture_blocks=len(blocks),architecture_counts=architecture_counts,
        blocks=blocks,description='Serialized architecture blocks, including nested residuals and the complete base-policy container; not independent models or independent seeds.',
        actor_instantiated=False,actor_calls=0))
    exported=[]
    for folder,entry in sorted(nodes.items()):
        rec=entry['receipt'] or {}
        exported.append(dict(stage=str(folder.relative_to(ROOT)), model_input_closure=folder in model_nodes,
            registration_sha256=entry['registration_sha256'], receipt_sha256=entry['receipt_sha256'],
            receipt_registration_hash_matches=entry['receipt_registration_hash_matches'],
            status=rec.get('status','MISSING_RECEIPT'), actual_reported_fields=numeric_fields(rec),
            planned_budget=entry['registration'].get('budget'), purpose=entry['registration'].get('purpose'),
            completed_utc=rec.get('completed_utc')))
    broad=aggregate_costs(nodes,set(nodes)); model_only=aggregate_costs(nodes,model_nodes)
    write_json(stage/'ledger.json',dict(nodes=exported,
        edges=[dict(source_stage=str(e['source_stage'].relative_to(ROOT)),input_stage=str(e['input_stage'].relative_to(ROOT)),
            input_file=str(e['input_file'].relative_to(ROOT)),expected_sha256=e['expected_sha256'],model_input=e['model_input']) for e in edges],
        broad_registered_dependencies=broad, model_input_dependencies=model_only,
        out_of_project_references_not_followed=outside,
        mismatched_receipt_registrations=[str(p.relative_to(ROOT)) for p,v in nodes.items() if v['receipt_registration_hash_matches'] is False],
        limitations=['A registered model input can be an anchor or comparator; it is not automatically a retained-weight ancestor.',
            'No blind summation across cost aliases or cumulative resumes. Unknown counts remain unknown.',
            'Historical model-development choices and adaptive support selection do not define three completed independent end-to-end seeds.',
            'This ledger inventories existing evidence; it does not prescribe replaying the historical research chain.']))
    columns=['stage','model_input_closure','status','wall_s','costs.training_actor_calls','costs.training_physics_ticks',
             'costs.ppo_updates','ppo_updates','costs.fitting_updates','costs.supervised_updates',
             'costs.actor_calls','costs.physics_ticks','costs.evaluation_actor_calls','costs.evaluation_physics_ticks',
             'new_policies','new_candidates','actual_new_candidates','new_physics_ticks','receipt_sha256']
    with (stage/'cost_fields.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in exported:
            values={**r,**r['actual_reported_fields']};writer.writerow({k:values.get(k) for k in columns})
    receipt(stage,dict(status='completed_registered_learning_provenance_and_cost_inventory',
        registered_dependency_stages=len(nodes),model_input_dependency_stages=len(model_nodes),
        architecture_blocks=len(blocks),architecture_counts=architecture_counts,
        missing_receipts=len(broad['missing_receipts']),
        receipt_registration_mismatches=sum(v['receipt_registration_hash_matches'] is False for v in nodes.values()),
        matched_training_budget_established=False, completed_independent_full_seeds=0,
        new_training_updates=0,new_actor_calls=0,new_simulation_ticks=0,
        ledger_sha256=sha(stage/'ledger.json'),structure_sha256=sha(stage/'checkpoint_structure.json'),
        wall_s=time.perf_counter()-started,goal_complete=False))
    print(json.dumps(dict(stages=len(nodes),model_inputs=len(model_nodes),blocks=len(blocks),
        architecture_counts=architecture_counts,missing_receipts=broad['missing_receipts'],
        registration_mismatches=sum(v['receipt_registration_hash_matches'] is False for v in nodes.values()),
        model_input_reported_wall=model_only['reported_field_totals'].get('wall_s'),
        broad_reported_wall=broad['reported_field_totals'].get('wall_s')),ensure_ascii=False))


if __name__=='__main__': main()
