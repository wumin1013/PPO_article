"""Recompute saved terminal commands, verify bindings, and preserve old sources."""
from pathlib import Path
import json
import zipfile

import numpy as np
import yaml

import preview_event_study as old
from terminal_metrics import evaluate_terminal_trajectory

OUT=old.ROOT/'论文项目/revision_20260909_terminal'


def main():
    target=OUT/'terminal_independent_recheck.json'
    if target.exists():raise FileExistsError('Preserve previous verification')
    old.check_protocol()
    protocol=old.read(OUT/'comparison_v2_protocol.json')
    with zipfile.ZipFile(OUT/'comparison_v2_source.zip') as z:
        import hashlib
        for name,value in protocol['sources'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==value,name
            assert old.sha(old.ROOT/name)==value,name
    checked=[]
    for split in ('validation','test'):
        records=old.read(OUT/'comparison_v2'/split/'summary.json')['rows']
        for row in records:
            reference=old.points(row['path_id'])
            if row['variant']=='original_two_step':
                source=old.ROOT/row['source_trace_path'];assert old.sha(source)==row['source_trace_sha256']
                command=np.loadtxt(source,delimiter=',',skiprows=1);t,p=command[:,0],command[:,1:]
                parent=old.read(source.parent/'result.json')
                initial={k:v for k,v in parent['initial_state'].items() if k.startswith('initial_')}
                cfg=yaml.safe_load((source.parent.parent/'common_config.yaml').read_text(encoding='utf-8'))
            else:
                folder=OUT/'comparison_v2'/split/f"{row['variant']}_seed{row['seed']}"/row['path_id']
                assert old.sha(folder/'command.npz')==row['command_sha256']
                assert old.sha(old.ROOT/row['source_trace_path'])==row['source_trace_sha256']
                with np.load(folder/'command.npz') as z:p,t=z['positions'],z['timestamps']
                initial=row['initial_state']
                cfg=yaml.safe_load((old.OUT/'formal'/f"{row['variant']}_seed{row['seed']}"/'config.yaml').read_text(encoding='utf-8'))
                with np.load(folder/'brake.npz') as z:
                    assert all(z[k][-1]==0 for k in ('v_mm_s','a_mm_s2','omega_rad_s','alpha_rad_s2'))
            fresh=evaluate_terminal_trajectory(reference,p,t,tolerance_mm=1.25,limits=cfg['kinematic_constraints'],
                termination_reason=row['termination_reason'],claimed_success=row['claimed_success'],
                witness_spacing_mm=.01,**initial)
            for key,value in fresh.items():
                if row[key]!=value:raise AssertionError((split,row['path_id'],row['variant'],key,row[key],value))
            checked.append(dict(split=split,path_id=row['path_id'],variant=row['variant'],seed=row.get('seed'),
                                sample_count=len(t),success=fresh['success'],all_fields_identical=True))
    original=old.read(OUT/'before_manifest.json')
    for name,value in original['artwork'].items():assert old.sha(old.ROOT/name)==value,name
    qa=old.read(OUT/'manuscript_final_qa.json')
    assert old.sha(old.ROOT/'论文项目/main.pdf')==qa['main_pdf_sha256']
    assert old.sha(old.ROOT/'论文项目/main.tex')==qa['main_tex_sha256']
    result=dict(records=checked,run_count=len(checked),all_metric_fields_identical=True,
        total_command_samples=sum(x['sample_count'] for x in checked),
        artwork_files_unchanged=len(original['artwork']),prior_event_source_sha256=old.source_state()['source_sha256'],
        manuscript_pages=qa['final_pages'],main_pdf_sha256=qa['main_pdf_sha256'],
        verification_source_sha256=old.sha(__file__),command='python PPO_project/tools/verify_terminal_delivery.py')
    old.write(target,result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
