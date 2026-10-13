"""Matched saved constant/pulse responses and actual 1ms command instrumentation."""
import numpy as np
from study_io import BASE,write_json
from search_complete_return import read


def key(c):return tuple(np.round(c['coefficients'],12))


def main():
    old=BASE/'canonical_gate_band_early_fullroute_joint_learning_20260920'
    stage=BASE/'canonical_gate_band_early_learned_pulse_joint_learning_20260920'
    previous=read(old/'state_joint/results.json')['candidates']
    current=read(stage/'state_joint/results.json')['candidates']
    old_by_key={key(c):c for c in previous}
    rescored={c['tag']:c for c in read(old/'full_endpoint_coverage_audit.json')['candidates']}
    baseline=current[0];old_base=previous[0]
    assert abs(baseline['score']-rescored[old_base['tag']]['full_endpoint_total_score'])<1e-9
    response=[];instrument=[]
    for c in current[1:]:
        k=key(c)
        if k not in old_by_key:continue
        o=old_by_key[k];arms=[]
        for path in (206,304):
            b=next(r for r in baseline['records'] if r['path']==path)
            values={}
            for name,vector in (('constant',o),('learned_pulse',c)):
                r=next(r for r in vector['records'] if r['path']==path)
                values[name]=dict(success=r['success'],delta={metric:r['shape'][metric]-b['shape'][metric]
                    for metric in ('full_physical_reverse_deg','out_edge_outward_mm','reverse_lateral_total_mm',
                                   'reverse_lateral_significant_duration_s','full_normalized_energy',
                                   'full_spatial_peak_curvature_mm_inv','apex_speed_mm_s')})
            arms.append(dict(path=path,**values))
        response.append(dict(coefficients=c['coefficients'],constant_tag=o['tag'],pulse_tag=c['tag'],
            constant_score=rescored[o['tag']]['full_endpoint_total_score'],pulse_score=c['score'],
            constant_original_and_endpoint_pass=rescored[o['tag']]['original_guard_pass'] and rescored[o['tag']]['full_endpoint_pass'],
            pulse_pass=min(c['measured_inequalities'])>=-1e-8,records=arms))
    for c in current[1:7]:
        for r in c['records']:
            if r.get('reused_exact_zero_support'):continue
            with np.load(r['trace']) as z:
                if 'executed_actions_1ms' not in z:continue
                rows=z['rows'].copy();executed=z['executed_actions_1ms'].copy()
            assert len(rows)==len(executed)
            assert np.isfinite(executed[rows[:,12]==0]).all()
            changes=[]
            for start in range(0,len(rows)-19,20):
                block=executed[start:start+20]
                if np.isfinite(block).all():changes.append(np.ptp(block,axis=0))
            maximum=np.max(changes,axis=0) if changes else np.zeros(3)
            instrument.append(dict(tag=c['tag'],path=r['path'],maximum_within_decision_command_change=maximum.tolist(),
                all_ordinary_ticks_recorded=True,physical_pass=r['audit']['kinematic_pass'],rest_pass=r['audit']['rest_pass']))
    write_json(stage/'matched_temporal_response_audit.json',dict(matched_nonzero_vectors=len(response),responses=response,
        command_instrumentation=instrument,baseline_full_endpoint_objective_identical=True,
        constant_physical_reruns=0,pulse_representation_is_learned_parameter_optimization_not_PPO=True))
    print('MATCHED_TEMPORAL_RESPONSES',len(response),'instrumented',len(instrument),
          'nonconstant',sum(max(r['maximum_within_decision_command_change'])>0 for r in instrument),flush=True)
    for r in response:print('MATCHED_SCORES',r['constant_tag'],r['constant_score'],r['pulse_score'],r['pulse_pass'],flush=True)


if __name__=='__main__':main()
