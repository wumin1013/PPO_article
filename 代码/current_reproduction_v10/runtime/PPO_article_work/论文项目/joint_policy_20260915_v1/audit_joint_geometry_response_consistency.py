"""Check whether the proposal model excludes actually feasible sampled axes."""
import time
import numpy as np
from learn_coupled_feasibility_credit import build_geometric_response
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def main():
    source=BASE/'canonical_gate_band_compact_joint_growth_completion_20260919'
    data=read(source/'results.json')
    index={r['tag']:r for r in data['candidates']}
    current=index['reused_retained86']
    paired=[(index[f'iter0_axis{h}_-1'],index[f'iter0_axis{h}_+1']) for h in range(48)]
    stage=register('canonical_gate_band_joint_geometry_response_consistency_20260919',
        [__file__,BASE/'learn_coupled_feasibility_credit.py',source/'results.json',source/'receipt.json'],
        dict(reused_candidates=96,new_physics_ticks=0,new_actor_calls=0,wall_s=60),
        'The proposal optimizer predicted improvements yet missed the best actually '
        'measured feasible axis. Check calibration consistency before collecting '
        'more data: does the signed sampled-corridor linear response reject any '
        'known complete, primary-preserving axis used to build it? Reuse all96 '
        'raw responses; no physical rollout, optimization update or adoption. '
        'A false exclusion diagnoses this local approximation only, not PPO '
        'bias or a universal mechanism. Preserve all actual geometry checks.')
    started=time.perf_counter()
    margin,matrix,queries=build_geometric_response(source,current,paired)
    rows=[]
    for axis,pair in enumerate(paired):
        for sign,c in zip((-1,1),pair):
            complete=all(r['success'] for r in c['records'])
            primary=c.get('primary_preserved',False)
            predicted=float((margin-sign*matrix[:,axis]).min())
            rows.append(dict(tag=c['tag'],complete=complete,primary=primary,
                minimum_predicted_point_margin_mm=predicted,
                actual_minimum_segment_margin_mm=min(r['corridor']['margin_lower_mm'] for r in c['records']),
                false_exclusion=bool(complete and predicted < -1e-7)))
    bad=[r for r in rows if r['false_exclusion']]
    result=dict(records=rows,false_excluded_complete_axes=len(bad),
        false_excluded_primary_axes=sum(r['primary'] for r in bad),sampled_geometry=queries,
        worst=sorted(bad,key=lambda r:r['minimum_predicted_point_margin_mm'])[:5])
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_joint_geometry_response_consistency',
        false_excluded_complete_axes=len(bad),false_excluded_primary_axes=result['false_excluded_primary_axes'],
        wall_s=time.perf_counter()-started,new_physics_ticks=0,adopted=False,goal_complete=False))
    print('JOINT_GEOMETRY_CONSISTENCY',result['false_excluded_complete_axes'],result['false_excluded_primary_axes'],result['worst'])


if __name__=='__main__':main()
