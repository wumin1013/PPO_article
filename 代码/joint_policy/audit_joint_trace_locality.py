"""Compare unsmoothed corner phases of an actual candidate and its protected parent."""
import argparse
import numpy as np
from localize_joint_shape_residual import characterize, intervals
from study_io import BASE, write_json, sha
from search_complete_return import read


def details(record):
    result=characterize(record['trace'])
    with np.load(record['trace']) as z:
        ref,rows=z['reference'].copy(),z['rows'].copy()
    u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
    sign=np.sign(u[0]*v[1]-u[1]*v[0]);normal=sign*np.array([-u[1],u[0]])
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    heading=np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@u))
    approach=(np.arange(len(rows))<result['apex_ms'])&(speed>5)&(rows[:,12]==0)
    mask=approach&(heading < -np.deg2rad(.05))
    result['significant_lateral_intervals_ms']=[list(map(int,x)) for x in intervals(mask)]
    result['minimum_approach_heading_deg']=float(np.rad2deg(heading[approach].min()))
    result['motion']=record['motion'];result['shape']=record['shape']
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_signed_heading_joint_recovery_20260920')
    parser.add_argument('--tag',default='geometry_secant_17')
    args=parser.parse_args();stage=BASE/args.stage
    history=read(stage/'state_joint/results.json')['candidates']
    candidate=next(c for c in history if c['tag']==args.tag)
    parent=history[0];records=[]
    for r,b in zip(candidate['records'],parent['records']):
        if r['path'] not in (14,104,105,204,205,304,305):continue
        actual,reference=details(r),details(b)
        records.append(dict(path=r['path'],candidate=actual,parent=reference))
        print('RAW_PHASE',r['path'],'lateral',actual['significant_lateral_intervals_ms'],
            'entry_heading',actual['minimum_approach_heading_deg'],'apex',actual['apex_ms'],
            'outgoing',actual['outgoing_min_mm'],actual['outgoing_min_ms'],
            'reverse',actual['preapex_reverse_deg'],actual['postapex_reverse_deg'],flush=True)
    write_json(stage/f'{args.tag}_raw_phase_audit.json',dict(tag=args.tag,records=records,new_rollouts=0,
        source_sha256=sha(stage/'state_joint/results.json'),interpretation='Saved unsmoothed1ms XY,V,A. '
        'Phase association is descriptive, not a new causal intervention or generalization result.'))


if __name__=='__main__':main()
