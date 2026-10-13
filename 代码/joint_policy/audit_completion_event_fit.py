"""Check observed support and feasibility before new completion-aware learning."""
import copy
import numpy as np
from completion_event_response import event_values,fit,propose
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920'
    history=copy.deepcopy(read(stage/'state_joint/results.json')['candidates'])
    specs=[dict(path=p,tick=980,total_ticks=1100) for p in (304,305)]
    for c in history:event_values(c,specs)
    anchor=next(c for c in history if c['tag']=='refreshed_trust_2')
    with np.load(stage/'current_joint_response.npz') as z:fresh,transform,base=z['fresh'],z['transform'],z['base']
    model,emodel,buffer=fit(history,anchor,fresh)
    normalizer=read(stage/'response_error_channels.json')['normalizer']
    proposals=[]
    for radius in (.125,.25,.5,1.):
        solved=propose(anchor,transform,model,emodel,buffer,radius,normalizer,base,specs)
        row=dict(radius=radius,feasible=solved is not None)
        if solved is not None:
            delta,predicted,events=solved
            row.update(delta=delta.tolist(),predicted_drop=predicted,event_margins_mm=events.tolist())
        proposals.append(row);print('COMPLETION_EVENT_FIT',row,flush=True)
    write_json(stage/'completion_event_fit_audit.json',dict(specs=specs,buffer_mm=buffer.tolist(),
        all_saved_candidates_predicate_equivalence=True,proposals=proposals,
        new_physical_runs=0,event_margin_response=emodel.tolist()))


if __name__=='__main__':main()
