"""Audit the complete registered grid and derive all final manuscript statistics."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.stats import beta,t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/results_v3'
METHODS=['adaptive','fixed','S2R','S2','SCMO']
LABELS={'adaptive':'J-NNC','fixed':'Fixed preview','S2R':'S2-R','S2':'S2-Q','SCMO':'SCMO'}
SEEDS=[101,211,307,401,503]

def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def audit_motion(record,jsonfile):
    npz=jsonfile.with_suffix('.npz')
    if not record.get('trace_sha256'):
        if record['success']:raise ValueError('Successful record has no trajectory '+str(jsonfile))
        return {'trajectory_present':False,'complete_time_s':None,'elapsed_prefix_s':None}
    if not npz.exists():
        npz=ROOT/record['trace']
    if sha(npz)!=record['trace_sha256']:raise ValueError('Trajectory hash mismatch '+str(npz))
    with np.load(npz,allow_pickle=False) as z:
        p=z['positions'];reference=z['reference']
    dt=record['dt_s']
    if len(p)<2:return {'trajectory_present':True,'complete_time_s':None,'elapsed_prefix_s':0.}
    v=np.diff(p,axis=0)/dt
    a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
    j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
    speed=np.linalg.norm(v,axis=1)
    peaks=np.array([np.linalg.norm(x,axis=1).max(initial=0) for x in [v,a,j]])
    duration=(len(p)-1)*dt
    length=speed.sum()*dt
    deficit=(100-speed).sum()*dt
    if abs(duration-(length+deficit)/100)>1e-8:raise ValueError('Time accounting identity failed')
    if 'vector_peaks' in record and not np.allclose(peaks,record['vector_peaks'],rtol=1e-8,atol=1e-7):
        raise ValueError('Stored kinematics disagree with positions '+str(jsonfile))
    if record['success']:
        assert np.all(peaks/np.array([100,2000,62500])<=1+2e-5)
        assert abs(duration-record['complete_time_s'])<1e-8
        assert record['geometry']['geometry_complete'] and record['geometry']['gates_passed']==record['geometry']['gate_count']
        assert record['terminal_rest_s']>=.02-1e-10
    return {'trajectory_present':True,'complete_time_s':duration if record['success'] else None,
            'elapsed_prefix_s':duration,'executed_length_mm':float(length),'speed_deficit_mm':float(deficit),
            'mean_speed_mm_s':float(speed.mean()),'v_peak':float(peaks[0]),'a_peak':float(peaks[1]),
            'j_peak':float(peaks[2]),'cruise_99_s':float(np.count_nonzero(speed>=99)*dt),
            'trace_file':npz.relative_to(ROOT).as_posix()}

def run():
    registry=read(ROOT/'prospective/final_v3/registration.json')
    indexed={};inputs={}
    for file in (ROOT/'results/final_v3b').glob('*.json'):
        r=read(file)
        if not all(key in r for key in ['method','seed','path','dt_s','success']):continue
        key=(r['method'],r['seed'],r['path'],r['dt_s'])
        if key in indexed:raise ValueError('Duplicated final cell '+str(key))
        indexed[key]=(r,file)
    expected=[(m,s,p['id'],dt) for p in registry['paths'] for m in METHODS for dt in [.001,.0005]
              for s in (SEEDS if m in ['adaptive','fixed'] and dt==.001 else [101] if m in ['adaptive','fixed'] else [None])]
    missing=[k for k in expected if k not in indexed]
    if missing:raise RuntimeError(f'Missing {len(missing)} of {len(expected)} registered final cells: {missing[:5]}')
    rows=[]
    for key in expected:
        r,file=indexed[key]
        inputs[file.relative_to(ROOT).as_posix()]=sha(file)
        motion=audit_motion(r,file)
        rows.append({'method':key[0],'seed':key[1],'path':key[2],'dt_s':key[3],'success':bool(r['success']),
                     'reason':r.get('reason'), 'error':r.get('error'), 'wall_s':r.get('wall_s'),
                     'max_polyline_deviation_mm':r.get('max_polyline_deviation_mm'),
                     'tolerance_mm':r.get('tolerance_mm'), 'terminal_rest_s':r.get('terminal_rest_s'),
                     'result_file':file.relative_to(ROOT).as_posix(),**motion})
    summary=[];paired=[];resolution=[]
    for path in registry['paths']:
        for method in METHODS:
            sample=[r for r in rows if r['method']==method and r['path']==path['id'] and r['dt_s']==.001]
            times=[r['complete_time_s'] for r in sample if r['success']]
            summary.append({'path':path['id'],'method':method,'attempts':len(sample),'completed':len(times),
                            'mean_s':float(np.mean(times)) if times else None,
                            'sd_s':float(np.std(times,ddof=1)) if len(times)>1 else None,
                            'success_95_interval':interval(len(times),len(sample)) if method in ['adaptive','fixed'] else None})
        diffs=[];seeds=[]
        for seed in SEEDS:
            pair=[next(r for r in rows if r['method']==m and r['path']==path['id'] and r['seed']==seed and r['dt_s']==.001) for m in ['adaptive','fixed']]
            if all(r['success'] for r in pair):diffs.append(pair[1]['complete_time_s']-pair[0]['complete_time_s']);seeds.append(seed)
        mean=float(np.mean(diffs)) if diffs else None
        half=float(t.ppf(.975,len(diffs)-1)*np.std(diffs,ddof=1)/np.sqrt(len(diffs))) if len(diffs)>1 else None
        paired.append({'path':path['id'],'complete_pairs':len(diffs),'seeds':seeds,'fixed_minus_adaptive_s':diffs,
                       'mean_difference_s':mean,'student_95_interval_s':[mean-half,mean+half] if half is not None else None})
        for method in METHODS:
            seed=101 if method in ['adaptive','fixed'] else None
            pair=[next(r for r in rows if r['method']==method and r['path']==path['id'] and r['seed']==seed and r['dt_s']==dt) for dt in [.001,.0005]]
            resolution.append({'path':path['id'],'method':method,'seed':seed,
                               'success_1ms':pair[0]['success'],'success_0p5ms':pair[1]['success'],
                               'fine_minus_coarse_s':pair[1]['complete_time_s']-pair[0]['complete_time_s'] if all(r['success'] for r in pair) else None})
    training=[]
    for seed in SEEDS:
        for method in ['adaptive','fixed']:
            folder=ROOT/'results/unified_v3/paired'/f'seed_{seed}'/method
            r=read(folder/'receipt.json');assert r['complete'] and r['source_unchanged'] and r['protected_sources_unchanged']
            assert r['cost']['training_interactions']==8192 and r['cost']['supervised_optimizer_steps']==2000
            training.append({'seed':seed,'method':method,**r['cost']})
            inputs[(folder/'receipt.json').relative_to(ROOT).as_posix()]=sha(folder/'receipt.json')
    OUT.mkdir(exist_ok=True)
    result={'valid_for_manuscript':True,'expected_cells':len(expected),'registered_programs':registry['paths'],
            'rows':rows,'summary':summary,'paired':paired,'resolution':resolution,'training_costs':training,
            'teacher_selection':{k:v for k,v in read(ROOT/'results/unified_v3/teacher_selection.json').items() if k!='candidates'},
            'statistical_unit':'Independent training seed, repeated within each complete program; times conditional on completion and counts always shown.',
            'inputs_sha256':inputs}
    write(OUT/'analysis.json',result)
    with (OUT/'source_data.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'valid_for_manuscript':True,'cells':len(rows),'summary':summary,'paired':paired},indent=2))
    return result

if __name__=='__main__':run()
