"""Recompute complete-run metrics and check frozen outputs; never trains or simulates."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def longest(mask):
    edges=np.diff(np.r_[False, mask, False].astype(int))
    return int(np.max(np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1), initial=0))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New result JSON outside the supplied package')
    parser.add_argument('--check-actions', action='store_true', help='Requires PyTorch; checks the Appendix B output formula')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if HERE == args.output.resolve().parent or HERE in args.output.resolve().parents:
        raise ValueError('Write results outside the immutable evidence package')
    manifest=json.loads((HERE/'manifest.json').read_text('utf-8'))
    for row in manifest['files']:
        if sha(HERE/row['path']) != row['sha256']: raise ValueError('Hash mismatch: '+row['path'])
    output={'scope':'Existing-record recalculation; no new execution or training', 'records':[]}
    if args.check_actions:
        import torch
        from frozen_policy import FrozenPolicy
        torch.set_num_threads(1)
        policies={key:FrozenPolicy(HERE/path) for key,path in manifest['models'].items()}
    for record in manifest['records']:
        data=np.load(HERE/record['trace'], allow_pickle=False)
        rows=data['rows']; dt=record['dt_s']
        reference=np.loadtxt(HERE/manifest['cases'][record['case']]['reference'],delimiter=',',skiprows=1)
        position=rows[:,:2]
        v=np.diff(np.vstack([reference[0],position]),axis=0)/dt
        a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
        j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
        speed=np.linalg.norm(v,axis=1); acceleration=np.linalg.norm(a,axis=1); jerk=np.linalg.norm(j,axis=1)
        time=len(rows)*dt
        assert abs(time-record['complete_time_s']) < 1e-8
        values={'method':record['method'],'case':record['case'],'dt_s':dt,'complete_time_s':time,
            'executed_distance_mm':float(speed.sum()*dt),'speed_deficit_mm':float((100.-speed).sum()*dt),
            'cruise_ge99_s':float((speed>=99.).sum()*dt),'longest_cruise_ge99_s':longest(speed>=99.)*dt,
            'mean_velocity_use_percent':float(speed.mean()),
            'acceleration_95pct_percent':float(np.percentile(acceleration,95)/2000*100),
            'jerk_95pct_percent':float(np.percentile(jerk,95)/62500*100),
            'vector_peaks':[float(x.max()) for x in [speed,acceleration,jerk]],
            'geometry_and_coverage':'See bound original result and gate audit; not re-certified by this script'}
        if args.check_actions and record['method'] in policies:
            p=policies[record['method']]
            obs=data['observations']; stored=data['latents']
            ids=np.unique(np.linspace(0,len(obs)-1,min(257,len(obs)),dtype=int))
            # One observation per call matches the original deterministic evaluation arithmetic.
            actual=np.array([p.mean(obs[i]).numpy() for i in ids])
            delta=float(np.max(np.abs(actual-stored[ids])))
            assert delta < 2e-6, (record['method'],record['case'],delta)
            values['checked_policy_outputs']=len(ids)
            values['max_policy_mean_error']=delta
            if 'actions' in data:
                decoded=np.array([p.decoded(obs[i]) for i in ids])
                values['max_decoded_action_error']=float(np.max(np.abs(decoded-data['actions'][ids])))
                assert values['max_decoded_action_error'] < 1e-4
        output['records'].append(values)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {len(output['records'])} bound complete traces; results: {args.output}")

if __name__=='__main__':main()
