"""Offline comparison on the interiors of unchanged reference straight sections."""
import json
import numpy as np
from evaluate_historical_dense import ROOT,BASE,sha,write
from natural_case_metrics import ordered_projection

STAGE=ROOT/'direct14_multicase_ppo_seed922302_round1'
CASE=BASE/'paper_evidence_review/natural_closed_case'


def main():
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    paths={'initial':ROOT/'direct14_saved_motion_fit_seed922302_original_full/full_trace.npz'}
    for path in STAGE.glob('eval_*_original.npz'):
        if json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['success']:paths[path.stem]=path
    records=[]
    for label,path in paths.items():
        with np.load(path) as data:
            reference=data['reference'];rows=data['rows'];held=data['executed_actions_1ms']
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
        progress=ordered_projection(reference,rows[:,:2]);speed=np.linalg.norm(rows[:,2:4],axis=1)
        straight=np.zeros(len(rows),dtype=bool)
        for p in geometry['primitives']:
            if p['kind']=='line':
                lo=arc[p['start_reference_index']]+30.;hi=arc[p['end_reference_index']]-30.
                straight|=(progress>=lo)&(progress<=hi)
        for half,mask in [('all',straight),('first',straight&(progress<arc[-1]/2)),('second',straight&(progress>=arc[-1]/2))]:
            ss=speed[mask];feed=held[mask,1]
            records.append(dict(label=label,half=half,samples=int(mask.sum()),duration_s=float(mask.sum()*.001),
                actual_speed_mean=float(ss.mean()),actual_speed_percentile_5_50_95=np.percentile(ss,[5,50,95]).tolist(),
                requested_feed_mean=float(feed.mean()),requested_feed_below99_fraction=float((feed<99).mean()),
                cruise99_s=float((ss>=99).sum()*.001),speed_deficit_mm=float((100.-ss).sum()*.001)))
    result=dict(scope='Offline diagnostic; no controller change, new training or simulation.',
        definition='Straight reference primitives, excluding30mm from each end; identical definition for all saved complete traces.',
        sources={str(p):sha(p) for p in paths.values()},records=records)
    write(STAGE/'straight_cruise_diagnostic.json',result)
    print(json.dumps(records,ensure_ascii=False))


if __name__=='__main__':main()
