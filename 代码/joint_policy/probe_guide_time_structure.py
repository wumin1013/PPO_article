"""One-decision causal contrast of a held offset versus held combined bearing."""
import math
import time
import numpy as np
import torch
from gate_band_state import GateBandRuntime
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from audit_xy_curvature_resolution import curvature_metrics
from audit_current_command_coordinates import command_signals
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


class HeldBearingProbe(GateBandRuntime):
    held_difference = None

    def guidance_target(self):
        if self.held_difference is not None:
            return self.position+self.held_difference
        return super().guidance_target()


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_monotone_block_learning_20260917'
    audit=BASE/'canonical_gate_band_current_command_coordinates_20260917'
    data=read(parent/'results.json');selected=data['full_candidate']
    model=parent/data['chosen']['model'];metadata=torch.load(model,map_location='cpu',weights_only=False)
    assert sha(model)==read(audit/'results.json')['parent_sha256']
    paths=[0,7,8,15]
    stage=register('canonical_gate_band_guide_time_structure_probe_20260917',
        [__file__,model,parent/'results.json',parent/'receipt.json',audit/'results.json',
         audit/'receipt.json',BASE/'gate_band_state.py',BASE/'joint_runtime.py',
         BASE/'probe_coordinated_suffix.py',BASE/'audit_current_command_coordinates.py'],
        dict(training_paths=paths,anchors=['first_negative_target','largest_intradecision_change'],
             intervention_decisions=1,new_suffixes=8,actor_calls=2500,new_physics_ticks=65000,
             saved_prefix_geometry_ticks=12000,wall_s=90,new_learned_parameters=0),
        'Current saved commands show outward LOS overwhelming inward residual '
        'and large subdecision bearing sweeps. At each of four declared corner '
        'paths, intervene for exactly one20ms decision at the first negative '
        'combined target or the largest intra-decision target change. Hold the '
        'initial LOS difference VECTOR in moving position coordinates, thereby '
        'holding its bearing, with identical initial learned steering/feed/preview. '
        'Do not alter actor observations, frozen V/A/J execution or coverage. '
        'Then restore original continuously updated guide and finish. This tests '
        'causality of intra-decision guide timing, not a deployable controller, '
        'learned target path, or historical learning origin. Retain failures and '
        'full raw XY/heading/curvature/time/cruise. No automatic global guide change.')
    start=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture']);records=[]
    descriptions={r['path']:r for r in read(audit/'results.json')['records']}
    for path in paths:
        j=next(j for j,r in enumerate(selected['results']) if r['original_path']==path)
        with np.load(BASE/selected['traces'][j]) as z:
            ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        d=descriptions[path]
        anchors=dict(first_negative_target=d['first_negative_target']['decision'],
            largest_intradecision_change=d['largest_intradecision_changes'][0]['decision'])
        original_shape=complete_corner_metrics(ref,rows)
        for name,anchor in anchors.items():
            frame=anchor*20
            normal=reconstruct(ref,rows,actions,frame,costs,env_factory=factory)
            env=HeldBearingProbe(ref,max_time=normal.max_ticks*.001);env.restore(normal.state())
            latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(latent),actions[anchor])
            env.held=actions[anchor].copy()
            target=env.guidance_target();env.held_difference=target-env.position
            assert np.max(abs(env.guidance_target()-target))<1e-12
            chunks=[];held=[]
            while not env.done:
                assert time.perf_counter()-start<90 and costs['actor_calls']<2500
                assert costs['new_physics_ticks']<62880
                if chunks:
                    latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
                _,_,_,info=env.step(latent)
                chunks.append(info['rows']);held.append(info['held_action'])
                costs['new_physics_ticks']+=len(info['rows'])
                env.held_difference=None
            full=np.vstack([rows[:frame],*chunks]);all_actions=np.vstack([actions[:anchor],held])
            audit_values=difference_audit(ref[0],full)
            success=bool(env.success and audit_values['kinematic_pass'] and audit_values['rest_pass'])
            s=command_signals(ref,full,all_actions)
            initial_error=float(abs(s['combined_deg'][frame]-command_signals(ref,rows,actions)['combined_deg'][frame]))
            assert initial_error<1e-9
            span=float(np.ptp(s['combined_deg'][frame:frame+20]));assert span<1e-9
            shape=complete_corner_metrics(ref,full)
            trace=f'path{path}_{name}.npz'
            np.savez_compressed(stage/trace,reference=ref,rows=full,actions=all_actions,anchor=anchor)
            record=dict(path=path,anchor_kind=name,anchor=anchor,success=success,reason=env.reason,
                trace=trace,audit=audit_values,initial_target_error_deg=initial_error,
                intervention_target_range_deg=span,baseline_shape=original_shape,
                shape=shape,motion=complete_motion_metrics(ref,full),curvature=curvature_metrics(ref,full),
                complete_time_s=len(full)*.001 if success else None)
            records.append(record)
            write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,
                costs=costs,diagnostic_only=True,goal_complete=False))
            print('GUIDE_TIME_STRUCTURE',path,name,success,
                {k:(original_shape[k],shape[k]) for k in ('in_edge_outward_mm',
                    'out_edge_outward_mm','full_physical_reverse_deg','heading_overshoot_deg')},flush=True)
    receipt(stage,dict(status='completed_one_decision_guide_time_structure_probe',
        suffixes=len(records),complete_suffixes=sum(r['success'] for r in records),
        costs=costs,wall_s=time.perf_counter()-start,diagnostic_only=True))


if __name__=='__main__':main()
