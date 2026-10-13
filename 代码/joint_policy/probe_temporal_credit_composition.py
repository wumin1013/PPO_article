"""Check whether separately profitable local action blocks compose safely."""
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_entry_action_interactions import score
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916';model=parent/'candidate_1.pt'
    source=BASE/'entry_action_interaction_probe_20260916'
    selected=read(BASE/'matched_entry_interaction_feedback_20260916/targets.json')['window']['chosen']
    original=read(source/'results.json');paths=[0,1,7,15]
    stage=register('temporal_local_credit_composition_20260916',
        [__file__,model,source/'receipt.json',source/'results.json',
         BASE/'matched_entry_interaction_feedback_20260916/targets.json',BASE/'probe_entry_action_interactions.py'],
        dict(paths=paths,new_suffixes=4,new_decisions=1500,new_physics_ticks=35000,wall_s=60),
        'Test temporal composition of two independently return-improving60ms interventions on the '
        'same original training path. Reuse original baseline and both separately completed single '
        'blocks. The only new run per path applies both blocks at their original macro-decision '
        'indices then resumes parent feedback to full stop. Explicit temporal interaction, not a '
        'policy evaluation or deployable timing rule. Compare complete success, full actual shape '
        'and return with the additive prediction; a failure would refute naive composition of '
        'independently profitable labels. No change to the ongoing fitted-actor experiment.')
    p=LocalGeometryFeedbackPolicy(mask_endpoint=True);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0);started=time.monotonic();results=[]
    for path in paths:
        targets=sorted([r for r in selected if r['path']==path],key=lambda r:r['anchor']);assert len(targets)==2
        singles=[next(r for r in original['records'] if r['trace']==t['trace']) for t in targets]
        base=next(r for r in original['baselines'] if r['path']==path)
        with np.load(parent/'training_1'/f'{path}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        first=targets[0]['anchor'];env=reconstruct(ref,rows,actions,first*20,costs)
        chunks=[];held=[];count=0
        while not env.done:
            assert time.monotonic()-started<60 and costs['actor_calls']<1500 and costs['new_physics_ticks']<32880
            latent=p.act(env.observation(),True)[0];costs['actor_calls']+=1
            for single in singles:
                if single['anchor']<=first+count<single['anchor']+3:latent=latent+np.asarray(single['delta'])
            _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);count+=1
            costs['new_physics_ticks']+=len(info['rows'])
        full=np.vstack([rows[:first*20],*chunks]);allactions=np.vstack([actions[:first],held])
        audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
        value=score(ref,full,allactions,env)
        np.savez_compressed(stage/f'{path}.npz',reference=ref,rows=full,actions=allactions)
        results.append(dict(path=path,anchors=[r['anchor'] for r in singles],success=success,reason=env.reason,
            **value,delta_return=value['return_window']-base['return_window'],
            interaction_return=value['return_window']-sum(r['return_window'] for r in singles)+base['return_window'],
            baseline=base,singles=[dict(trace=r['trace'],return_window=r['return_window'],shape=r['shape']) for r in singles],
            full_time_s=len(full)*.001 if success else None,audit=audit,geometry=env.tracker.snapshot(1.25)))
        print('TEMPORAL',path,success,'gain',round(results[-1]['delta_return'],4),'interaction',round(results[-1]['interaction_return'],4),
              'exit',round(value['shape']['out_edge_outward_mm'],3),'reverse',round(value['shape']['full_physical_reverse_deg'],2),flush=True)
    write_json(stage/'results.json',dict(records=results,costs=costs))
    receipt(stage,dict(status='completed_temporal_action_composition',costs=costs,successful=sum(r['success'] for r in results),wall_s=time.monotonic()-started))


if __name__=='__main__':main()
