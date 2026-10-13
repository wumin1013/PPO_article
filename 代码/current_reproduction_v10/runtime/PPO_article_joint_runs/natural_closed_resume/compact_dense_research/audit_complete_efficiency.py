"""Read-only complete-time accounting from saved physical trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_historical_dense import ROOT,BASE,write,sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(trace,result_path,metric_path,dt):
    result=read(result_path);assert result['success']
    with np.load(trace) as data:
        rows=data['rows'];reference=data['reference'] if 'reference' in data else np.loadtxt(
            BASE/'paper_evidence_review/natural_closed_case/reference.csv',delimiter=',',skiprows=1)
    metric=read(metric_path);metric=metric.get('metrics',metric)
    assert metric['all34_windows_complete'] and len(metric['corners'])==34
    speed=np.linalg.norm(rows[:,2:4],axis=1);time=len(rows)*dt
    assert abs(time-result['complete_time_s'])<1e-8
    distance=float(speed.sum()*dt);deficit=float((100.-speed).sum()*dt)
    corner_mask=np.zeros(len(rows),bool)
    for corner in metric['corners']:
        lo,hi=corner['first_index'],corner['last_index']+1
        assert not corner_mask[lo:hi].any()
        corner_mask[lo:hi]=True
    partitions=[]
    for name,mask in [('all34_fixed_windows',corner_mask),('outside_fixed_windows',~corner_mask)]:
        partitions.append(dict(name=name,time_s=float(mask.sum()*dt),
            distance_mm=float(speed[mask].sum()*dt),deficit_mm=float((100-speed[mask]).sum()*dt)))
    high=speed>=99.;edges=np.diff(np.r_[False,high,False].astype(int))
    longest=float((np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)).max()*dt) if high.any() else 0.
    corner_deficit=float(sum(c['speed_deficit_mm'] for c in metric['corners']))
    assert abs(corner_deficit-partitions[0]['deficit_mm'])<1e-5
    position_length=float(np.linalg.norm(np.diff(np.vstack([reference[0],rows[:,:2]]),axis=0),axis=1).sum())
    return dict(time_s=time,executed_distance_mm=distance,total_speed_deficit_mm=deficit,
        corner_deficit_mm=corner_deficit,outside_corner_deficit_mm=partitions[1]['deficit_mm'],
        cruise_ge99_s=float(high.sum()*dt),longest_cruise_ge99_s=longest,
        position_integral_discrepancy_mm=position_length-distance,
        time_identity_error_s=(distance+deficit)/100-time,partitions=partitions,
        source_sha256={str(p):sha(p) for p in [trace,result_path,metric_path]})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();output=a.stage/f'complete_efficiency_{a.index:04d}.json'
    assert not output.exists(),'Preserve the previous derived audit'
    records=[]
    for key,dt in [('1ms',.001),('half_ms',.0005)]:
        prefix=a.stage/f'eval_{a.index:04d}_original_{key}'
        current=summarize(prefix.with_suffix('.npz'),prefix.with_suffix('.json'),prefix.with_name(prefix.name+'_metrics.json'),dt)
        historical=ROOT.parent/'all34_dominance_research/joint_transit_retained_seed922222_full'
        if key=='half_ms':historical=historical/'half_ms'
        specs=[('historical_main',historical/'full_trace.npz',historical/'full_result.json',historical/('full_metrics.json' if key=='1ms' else 'corner_metrics.json'))]
        for index in [0,24]:
            prefix=ROOT/'direct14_capacity_two_action_selection_round60'/f'eval_{index:04d}_original_{key}'
            specs.append((f'capacity_two_action_{index:04d}',prefix.with_suffix('.npz'),prefix.with_suffix('.json'),prefix.with_name(prefix.name+'_metrics.json')))
        comparisons=[]
        for label,trace,result,metric in specs:
            other=summarize(trace,result,metric,dt)
            gain=other['time_s']-current['time_s']
            distance_gain=(other['executed_distance_mm']-current['executed_distance_mm'])/100
            deficit_gain=(other['total_speed_deficit_mm']-current['total_speed_deficit_mm'])/100
            assert abs(gain-distance_gain-deficit_gain)<1e-9
            comparisons.append(dict(comparator=label,baseline=other,time_gain_s=gain,
                path_length_contribution_s=distance_gain,speed_deficit_contribution_s=deficit_gain,
                corner_deficit_contribution_s=(other['corner_deficit_mm']-current['corner_deficit_mm'])/100,
                cruise99_gain_s=current['cruise_ge99_s']-other['cruise_ge99_s']))
        records.append(dict(dt_s=dt,main=current,comparisons=comparisons))
    write(output,dict(records=records,equation='T = integral(|v|)dt/100 + integral(100-|v|)dt/100',
        scope='Complete successful trajectories including braking and rest; exact accounting, not causal attribution.',
        new_simulation_steps=0,new_ppo_updates=0,audit_script_sha256=sha(__file__)))
    for r in records:
        print(json.dumps(dict(dt_s=r['dt_s'],main_time_s=r['main']['time_s'],cruise99_s=r['main']['cruise_ge99_s'],
            comparisons=[{k:v for k,v in c.items() if k!='baseline'} for c in r['comparisons']])))


if __name__=='__main__':main()
