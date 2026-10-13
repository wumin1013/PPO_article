"""Check whether worst-guard restoration is supported by the measured joint model."""
import numpy as np
from completion_event_response import fit
from worst_guard_response import solve
from audit_coupled_trust_model import labels
from study_io import BASE,write_json
from search_complete_return import read


STAGE=BASE/'canonical_gate_band_complementary_joint_direction_learning_20260920'


def setup():
    history=read(STAGE/'state_joint/results.json')['candidates']
    anchor=next(c for c in history if c['tag']==read(STAGE/'receipt.json')['training_best'])
    with np.load(STAGE/'current_joint_response.npz') as z:fresh,transform,base=z['fresh'],z['transform'],z['base']
    model,emodel,buffer=fit(history,anchor,fresh)
    independent=[b for b in read(STAGE/'learning_updates.json') if b.get('event_margins_mm') is not None]
    over=np.max([np.array(b['predicted_event_margins_mm'])-b['event_margins_mm'] for b in independent],axis=0)
    buffer=np.maximum(buffer,np.maximum(over,0.)+.001)
    normalizer=read(BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920/response_error_channels.json')['normalizer']
    return history,anchor,fresh,transform,base,model,emodel,buffer,normalizer,read(STAGE/'completion_specs.json')


def main():
    history,anchor,fresh,transform,base,model,emodel,buffer,normalizer,specs=setup()
    names=labels(anchor['records']);g=anchor['augmented_learning_inequalities']
    worst=sorted(zip(names,g),key=lambda p:p[1])[:12]
    rows=[]
    for radius in (.25,.5,1.):
        result=solve(anchor,transform,model,emodel,buffer,radius,normalizer,base,specs)
        row=dict(radius=radius,feasible=result is not None)
        if result is not None:
            delta,predicted,events,info=result
            row.update(delta=delta.tolist(),predicted_worst_merit_drop=predicted,
                       predicted_event_margins_mm=events.tolist(),**info)
        rows.append(row);print('WORST_GUARD_CREDIT',row,flush=True)
    write_json(STAGE/'worst_guard_credit_audit.json',dict(proposals=rows,worst_learning_guards=worst,
        event_buffer_mm=buffer.tolist(),new_physical_runs=0,
        buffer_scope='Measured positive prediction errors plus .001mm; heuristic only, not a guaranteed bound.'))


if __name__=='__main__':main()
