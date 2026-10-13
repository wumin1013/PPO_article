"""Bind complete historical snapshots to separate learning and later fitting effects."""
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT, BASE, sha, write
from run_natural_resolution import physical_audit


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    delivery=BASE/'paper_evidence_review/verified_results/lowdim_original_revision'
    binding=read(delivery/'paper_result_binding.json')
    specs=[('PPO-derived policy before far-vector refinement','direct14_guided_batchreset_ppo_seed922302_round17','eval_0024_{}','checkpoint_0024.pt'),
           ('Shared far-vector backbone after fitting','direct14_farvector_local_feedback_fit_round21','eval_8000_{}','fit_08000.pt'),
           ('Final policy after local-feature refinement','direct14_dictionary02_resume_round73','eval_0000_{}_1ms','checkpoint_0000.pt')]
    records=[];sources={}
    for label,directory,pattern,checkpoint_name in specs:
        folder=ROOT/directory;checkpoint=folder/checkpoint_name;cases={}
        sources[str(checkpoint)]=sha(checkpoint)
        for case in ('original','dense'):
            result_path=folder/(pattern.format(case)+'.json');trace=folder/(pattern.format(case)+'.npz')
            r=read(result_path);assert r['success'] and r['checkpoint_sha256']==sha(checkpoint)
            sources[str(result_path)]=sha(result_path);sources[str(trace)]=sha(trace)
            with np.load(trace) as z:rows=z['rows'].copy();reference=z['reference'].copy()
            expected=np.loadtxt(binding['cases'][case]['reference'],delimiter=',',skiprows=1)
            assert np.array_equal(reference,expected)
            assert abs(len(rows)*.001-r['complete_time_s'])<1e-9
            physical=physical_audit(reference,rows,.001)
            assert physical['kinematic_pass'] and physical['rest_pass']
            analytic=r.get('analytic',{})
            assert analytic.get('analytic_corridor_pass'),(label,case)
            cases[case]=dict(complete_time_s=r['complete_time_s'],physical=physical,
                analytic_maximum_mm=analytic.get('analytic_maximum_distance_mm'),
                result=str(result_path),trace=str(trace))
        records.append(dict(stage=label,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),cases=cases))
    result=dict(records=records,source_sha256=sources,script_sha256=sha(__file__),
        scope='Three selected ancestral snapshots at1ms on the identical complete references. A cumulative provenance comparison, not independent seeds or a single-factor ablation. Earlier snapshot already includes offline initialization/fits.',
        new_training_updates=0,new_simulations=0)
    write(ROOT/'experimental_supplement_20260923/refinement_snapshot_audit.json',result)
    print('SNAPSHOTS_VERIFIED',[(r['stage'],[r['cases'][c]['complete_time_s'] for c in ('original','dense')]) for r in records])


if __name__=='__main__':main()
