"""Account for exact measured and predicted gains of the completion-event control."""
import copy
import numpy as np
from completion_event_response import event_values,fit
from audit_coupled_trust_model import labels
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    source=BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920'
    stage=BASE/'canonical_gate_band_completion_event_joint_learning_20260920'
    old=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    history=read(stage/'state_joint/results.json')['candidates'];by={c['tag']:c for c in history}
    initial=by['refreshed_trust_2'];specs=read(stage/'completion_specs.json')
    for c in old:event_values(c,specs)
    with np.load(source/'current_joint_response.npz') as z:fresh=z['fresh']
    model,emodel,buffer=fit(old,initial,fresh)
    normalizer=read(source/'response_error_channels.json')['normalizer']
    names=labels(initial['records']);out=[]
    for block in read(stage/'learning_updates.json'):
        if 'measured' not in block:continue
        c=by[block['measured']];d=np.array(c['subspace_coordinates'])-initial['subspace_coordinates']
        a=np.array(initial['augmented_learning_inequalities']);b=np.array(c['augmented_learning_inequalities'])
        predicted=a+d@model[:,:-1]
        if block['arm']=='event':
            for spec in specs:predicted[names.index(f"path{spec['path']}.complete_time_s")]=0.
        actual_changes=.1*(np.maximum(-b,0)-np.maximum(-a,0))
        predicted_changes=.1*(np.maximum(-predicted,0)-np.maximum(-a,0))
        error=actual_changes-predicted_changes
        channel_rows=[dict(channel=names[i],error=float(error[i]),actual_change=float(actual_changes[i]),
                           predicted_change=float(predicted_changes[i])) for i in np.argsort(abs(error))[-15:][::-1]]
        groups={}
        for n,change,err in zip(names,actual_changes,error):
            p=n.split('.')[0];g=groups.setdefault(p,dict(actual_change=0.,error=0.))
            g['actual_change']+=float(change);g['error']+=float(err)
        row=dict(tag=c['tag'],summary=aggregate(c['records']),top_prediction_errors=channel_rows,
                 by_path=groups,cost_prediction_error=float((c['training_score']-initial['training_score']-d@model[:,-1])/normalizer))
        out.append(row);print('EVENT_OUTCOME',c['tag'],'cost_error',row['cost_prediction_error'],
            'channels',channel_rows[:8],'paths',sorted(groups.items(),key=lambda x:abs(x[1]['error']),reverse=True)[:5],flush=True)
    assert sha(stage/'shadow_training_policy.pt')==sha(source/'shadow_training_policy.pt')
    write_json(stage/'outcome_audit.json',dict(records=out,new_physical_runs=0,
        saved_shadow_identical_to_previously_audited_actor=True,source_actor_audit_sha256=sha(source/'shadow_raw_identity_audit.json')))


if __name__=='__main__':main()
