"""Complete a matched diagnostic of outward LOS versus actor residual terms."""
import math
import time
import numpy as np
import torch
from joint_runtime import wrap
from joint_runtime_v5 import JointRuntimeV5
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct, encode
from audit_preswing_mechanism_chain import objective
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


class DiagnosticGuidanceRuntime(JointRuntimeV5):
    """Finite analysis-only intervention; original observation samples stay intact."""
    def sample_reference(self, arcs):
        target,direction=super().sample_reference(arcs)
        if getattr(self,'suppress_outward_los',False) and np.ndim(arcs)==0:
            delta=target-self.position
            angle=math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>1e-12 else self.heading
            if self.turn_sign*wrap(angle-self.incoming_heading)<0:
                target=self.position+np.linalg.norm(delta)*self.incoming_unit
                self.suppressed_ticks+=1
        return target,direction


def main():
    torch.set_num_threads(1)
    source=BASE/'preswing_mechanism_chain_20260916'
    parent=BASE/'exit_local_credit_parameter_learning_20260916';pool=parent/'training_1'
    old=read(source/'results.json')
    stage=register('preswing_los_residual_factorial_20260916',
        [__file__,source/'receipt.json',source/'results.json',parent/'candidate_1.pt',
         BASE/'audit_preswing_mechanism_chain.py',BASE/'joint_runtime.py',
         BASE/'joint_runtime_v5.py',BASE/'probe_coordinated_suffix.py'],
        dict(new_suffixes=16,reused_suffixes=8,reused_baselines=8,actor_calls=4000,
             new_physics_ticks=95000,wall_s=150,intervention_decisions=6,new_training_decisions=0),
        'Matched2x2 finite intervention at the same8 states and120ms windows as the previous '
        'preswing audit. Factors are removal of negative actor steering residual and removal '
        'of negative incoming-relative LOS bearing. Reuse00 and10 physics. New01/11 alter '
        'only scalar guide target queries during the finite window, keeping its distance '
        'and eliminating its negative bearing. Vector reference samples used by the actor '
        'observation are untouched. Original vector V/A/J projector,20/1ms clocks and '
        'terminal stop remain unchanged. Restore original guidance after120ms and finish '
        'every branch. This diagnoses a physical generating mechanism, not a deployable '
        'controller, proof of optimality, or proof of the historical learning cause.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    policy.load_state_dict(torch.load(parent/'candidate_1.pt',map_location='cpu',weights_only=False)['policy'])
    records=[]
    for base in old['baselines']:
        i=base['path'];anchor=base['decomposition']['anchor'];frame=anchor*20
        with np.load(pool/f'{i}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        normal=reconstruct(ref,rows,actions,frame,costs)
        env=DiagnosticGuidanceRuntime(ref,max_time=normal.max_ticks*.001)
        env.restore(normal.state());env.turn_sign=base['decomposition']['sign']
        env.incoming_unit=(ref[1]-ref[0])/np.linalg.norm(ref[1]-ref[0])
        env.incoming_heading=math.atan2(env.incoming_unit[1],env.incoming_unit[0])
        env.suppress_outward_los=False;env.suppressed_ticks=0
        assert np.array_equal(env.observation(),normal.observation())
        initial=env.state()
        for remove_residual in (False,True):
            env.restore(initial);env.suppressed_ticks=0;chunks=[];held=[];count=0
            while not env.done:
                assert time.perf_counter()-started<150 and costs['actor_calls']<4000
                assert costs['new_physics_ticks']<92880
                # Observation queries are arrays and therefore bypass scalar-only intervention.
                obs=env.observation();latent=policy.act(obs,True)[0];costs['actor_calls']+=1
                if count<6 and remove_residual:
                    action=env.decode_action(latent)
                    if env.turn_sign*action[0]<0:action[0]=0.;latent=encode(action)
                env.suppress_outward_los=count<6
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);count+=1
                costs['new_physics_ticks']+=len(info['rows'])
            env.suppress_outward_los=False
            full=np.vstack([rows[:frame],*chunks]);all_actions=np.vstack([actions[:anchor],held]) if anchor else np.asarray(held)
            audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            name=f'p{i}_los1_residual{int(remove_residual)}.npz'
            np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions,anchor=anchor)
            value=objective(ref,full,all_actions,env)
            row=dict(path=i,remove_los=True,remove_residual=remove_residual,success=success,
                full_time_s=len(full)*.001 if success else None,reason=env.reason,
                suppressed_guide_ticks=env.suppressed_ticks,trace=name,audit=audit,**value)
            row['return_changes']={k:value[k]-base[k] for k in ('base_return','incoming_window_return','two_edge_window_return','two_edge_full_return')}
            records.append(row)
            print('LOS_CAUSE',i,remove_residual,success,'entry',round(base['shape']['in_edge_outward_mm'],5),
                round(value['shape']['in_edge_outward_mm'],5),'suppressed',env.suppressed_ticks,flush=True)
        write_json(stage/'results.json',dict(records=records,reused_source=source.relative_to(BASE).as_posix(),costs=costs,
            runtime_intervention_adopted=False,historical_learning_cause_established=False))
    receipt(stage,dict(status='completed_los_residual_matched_causal_intervention',costs=costs,
        new_suffixes=len(records),complete_suffixes=sum(r['success'] for r in records),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
