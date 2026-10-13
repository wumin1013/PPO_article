"""Retain a verified intermediate model without declaring ideal-shape success."""
import time
import numpy as np
from audit_current_command_coordinates import command_signals
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    source=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918'
    visual=BASE/'canonical_gate_band_shape_cruise_completion_xy_review_20260918'
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    data=read(source/'results.json');candidate=data['full_candidate'];old=read(parent/'adoption.json')
    model=source/candidate['model'];assert candidate['checks']['eligible']
    assert read(visual/'visual_check.json')['inspected']
    stage=register('canonical_gate_band_shape_cruise_retention_20260918',
        [__file__,model,source/'results.json',source/'receipt.json',visual/'results.json',
         visual/'receipt.json',visual/'visual_check.json',parent/'adoption.json'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=30,saved_paths=64),
        'Retain the0.5 learned shared-parameter candidate only as an intermediate '
        'comparison model after64 complete paths, all declared per-corner shape '
        'nonregression, original dynamics/rest, unchanged aggregate time and '
        'nondecreasing actual cruise. Independently summarize both first-turn '
        'thresholds and actual route-type cruise from saved trajectories. The '
        'visual review still shows concentrated curvature and no ideal tangency '
        'acceptance. Preserve399a and all earlier strong models. This is not '
        'independent-seed, long-route, fair final-comparator or sealed validation.')
    start=time.perf_counter();onsets=[];commands=[];cruise=dict(straight=0.,isolated_corner=0.,multisegment=0.)
    for result,trace in zip(candidate['results'],candidate['traces']):
        assert result['success']
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        category='straight' if len(ref)==2 else 'isolated_corner' if len(ref)==3 else 'multisegment'
        cruise[category]+=float((speed>=99.).sum()*.001)
        if len(ref)!=3:continue
        s=command_signals(ref,rows,actions)
        apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
        use=(np.arange(len(rows))<apex)&(rows[:,12]==0)&(speed>5.)
        for threshold in (.05,.5):
            ids=np.flatnonzero(use&(abs(s['physical_heading_deg'])>threshold))
            assert len(ids)
            onsets.append(dict(path=result['original_path'],threshold_deg=threshold,
                first_inside=bool(s['physical_heading_deg'][ids[0]]>0),first_tick=int(ids[0])))
        commands.append(dict(path=result['original_path'],
            negative_physical_heading_ms=int((use&(s['physical_heading_deg']<-.05)).sum()),
            negative_combined_target_ms=int((use&(s['combined_deg']<-.05)).sum())))
    assert len(onsets)==32 and all(r['first_inside'] for r in onsets)
    peaks=np.max([r['audit']['peaks'] for r in candidate['results']],axis=0)
    assert np.all(peaks<=np.array([100.,2000.,62500.]))
    summary=candidate['summary']
    adopted=dict(model=str(model.relative_to(BASE)),sha256=sha(model),status='retained_intermediate_only',
        parent_preserved=old['model'],parent_sha256=old['sha256'],full_training_successes=64,
        full_time_s=summary['full_time_s'],cruise99_s=summary['complete_cruise_s']['99'],
        cruise99_by_route_type=cruise,entry_mm=summary['entry_sum_mm'],exit_mm=summary['exit_sum_mm'],
        reverse_deg=summary['full_reverse_sum_deg'],
        exit_reduction_fraction=1.-summary['exit_sum_mm']/old['exit_mm'],
        reverse_reduction_fraction=1.-summary['full_reverse_sum_deg']/old['reverse_deg'],
        negative_heading_ms=sum(r['negative_physical_heading_ms'] for r in commands),
        raw_combined_negative_target_ms=sum(r['negative_combined_target_ms'] for r in commands),
        all16_first_significant_turns_inside=True,strict_vector_peaks=peaks.tolist(),
        training_only=True,independent_generalization_verified=False,ideal_shape_accepted=False,goal_complete=False)
    write_json(stage/'results.json',dict(adoption=adopted,onsets=onsets,commands=commands,goal_complete=False))
    write_json(source/'adoption.json',adopted)
    receipt(stage,dict(status='retained_verified_intermediate_shape_candidate',model_sha256=sha(model),
        new_physics_ticks=0,new_actor_decisions=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('RETAINED_INTERMEDIATE',adopted,flush=True)


if __name__=='__main__':main()
