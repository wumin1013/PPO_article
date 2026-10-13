"""Geometric input validation before any final-test execution or freezing."""
import json
import shutil
import numpy as np
from common import ROOT, sha, write_json, utc, load_path
from runtime_bootstrap import setup
setup()
from geometry_spec import Corridor


def main():
    path=ROOT/'data'/'manifest.json';m=json.loads(path.read_text('utf-8'))
    if (ROOT/'audit'/'test_freeze.json').exists():raise RuntimeError('Never amend a frozen test')
    if any((ROOT/'results'/'unseen').glob('*.json')):raise RuntimeError('Final tests have already been used')
    old=ROOT/'data'/'manifest_preflight_v2.json'
    if old.exists():raise FileExistsError(old)
    shutil.copy2(path,old)
    file=ROOT/'data'/'test_short_segments.csv'
    before=Corridor(load_path('test_short_segments')).ambiguities
    assert before and all(r['kind']=='reversal_180' for r in before)
    shutil.copy2(file,ROOT/'audit'/'invalid_short_segments_generator_fixture.csv')
    points=[[0.,0.],[12.,0.]]
    for k in range(10):
        y=3. if k%2==0 else 0.
        points.extend([[12.+3*k,y],[15.+3*k,y]])
    points.append([54.,0.]);points=np.array(points)
    np.savetxt(file,points,delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12g')
    for row in m['paths']:
        if row['id']=='test_short_segments':
            row.update(sha256=sha(file),vertices=len(points),length_mm=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),generator='preflight_paths.py')
    report=[]
    for row in m['paths']:
        c=Corridor(load_path(row['id']))
        report.append(dict(path=row['id'],ambiguities=c.ambiguities,adjacent_gate_conflicts=c.adjacent_gate_conflicts()))
    assert not any(r['ambiguities'] or r['adjacent_gate_conflicts'] for r in report)
    m['preflight_amendment']=dict(utc=utc(),prior_manifest_sha256=sha(old),source_sha256=sha(__file__),
        reason='The unexecuted staircase generator contained nine unintended exact reversals. Corrected the input specification after geometric validity check, not after policy evaluation.',
        training_and_validation_bytes_unchanged=True,test_trajectory_executions_before_change=0,
        prior_invalid_input='audit/invalid_short_segments_generator_fixture.csv')
    write_json(path,m);write_json(ROOT/'audit'/'path_preflight.json',dict(results=report,all_valid=True,previous_invalid_reversals=before))
    print('All whole-program geometries pass input preflight; no policy executed')


if __name__=='__main__':main()
