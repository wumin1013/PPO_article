"""Bind complete candidate provenance without training or advancing physics."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,pickle,shutil,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
from candidate_policy_factory import candidate_policy
from turn_aware_projector import environment_factory

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');args=ap.parse_args();folder=Path(args.candidate)
    selection=json.loads((folder/'selected_policy.json').read_text(encoding='utf-8'));assert sha(Path(selection['model']))==selection['model_sha256']
    stage=folder/'runtime_provenance';stage.mkdir(exist_ok=False);source=stage/'source';source.mkdir()
    reference_path=BASE/'paper_evidence_review/natural_closed_case/reference.csv';ref=np.loadtxt(reference_path,delimiter=',',skiprows=1)
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False);policy=candidate_policy(metadata,selection);env=environment_factory(ref)
    if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
    policy.bind(env);x=env.observation();z=policy.act(x,True)[0]
    layers=[];obj=policy
    while obj is not None:
        memory=getattr(obj,'memory',None)
        layers.append(dict(type=type(obj).__name__,module=type(obj).__module__,
            memory_shape=list(np.shape(memory)) if memory is not None else None,
            memory_scalars=int(np.size(memory)) if memory is not None else 0))
        obj=getattr(obj,'base',None)
    files={}
    for name,module in list(sys.modules.items()):
        raw=getattr(module,'__file__',None)
        if not raw:continue
        p=Path(raw).resolve()
        if p.suffix.lower()!='.py' or not p.is_file():continue
        if not any(p.is_relative_to(root) for root in [Path(r'D:\PPO_article_work'),OUT]):continue
        files.setdefault(p,[]).append(name)
    records=[]
    for i,(p,names) in enumerate(sorted(files.items(),key=lambda item:str(item[0]))):
        target=source/(str(i).zfill(3)+'_'+p.name);shutil.copy2(p,target)
        records.append(dict(original=str(p),sha256=sha(p),modules=names,snapshot=str(target)))
    report=dict(goal_complete=False,kind='bound_candidate_import_and_layer_provenance',physics_ticks=0,ppo_updates=0,
        policy_calls_for_binding=1,model=selection['model'],model_sha256=selection['model_sha256'],
        selection_sha256=sha(folder/'selected_policy.json'),reference_sha256=sha(reference_path),observation_dimension=int(x.size),
        layers=layers,memory_layer_count=sum(x['memory_shape'] is not None for x in layers),memory_scalar_count=sum(x['memory_scalars'] for x in layers),
        imported_source_files=records,decoded_initial_action=env.decode_action(z,env.fixed_d).tolist(),
        dimension_optimality_established=False,scope='Exact bound policy implementation and inherited checkpoint; this is not an independent retraining or dimension ablation.')
    write_json(stage/'manifest.json',report)
    print('RUNTIME_PROVENANCE',folder.name,len(records),'layers',len(layers),'memory',report['memory_layer_count'],report['memory_scalar_count'],flush=True)

if __name__=='__main__':main()
