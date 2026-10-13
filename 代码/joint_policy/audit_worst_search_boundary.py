"""Separate an imposed response-center box from physical actor bounds."""
import copy
import numpy as np
from completion_event_response import fit
from worst_guard_response import solve
from study_io import BASE,write_json
from search_complete_return import read


STAGE=BASE/'canonical_gate_band_worst_guard_joint_learning_20260920'


def setup():
    history=read(STAGE/'state_joint/results.json')['candidates']
    current=next(c for c in history if c['tag']==read(STAGE/'receipt.json')['training_best'])
    measured=BASE/'canonical_gate_band_complementary_joint_direction_learning_20260920'
    with np.load(measured/'current_joint_response.npz') as z:fresh,transform,base=z['fresh'],z['transform'],z['base']
    model,emodel,buffer=fit([c for c in history if c['event_support_valid']],current,fresh)
    updates=read(STAGE/'learning_updates.json')+read(measured/'learning_updates.json')
    over=np.max([np.array(b['predicted_event_margins_mm'])-b['event_margins_mm']
        for b in updates if b.get('event_margins_mm') is not None],axis=0)
    buffer=np.maximum(buffer,np.maximum(over,0.)+.001)
    normalizer=read(BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920/response_error_channels.json')['normalizer']
    return history,current,fresh,transform,base,model,emodel,buffer,normalizer,read(STAGE/'completion_specs.json')


def main():
    history,current,fresh,transform,base,model,emodel,buffer,normalizer,specs=setup()
    center=np.array(current['subspace_coordinates']);u=np.array(current['coefficients'])
    centered=copy.deepcopy(current);centered['subspace_coordinates']=[0.]*len(center)
    proposals=[]
    for radius in (.25,.5,1.):
        for name,anchor,origin in [('old_center_box',current,base),('recentered_box',centered,u)]:
            solution=solve(anchor,transform,model,emodel,buffer,radius,normalizer,origin,specs)
            r=dict(box=name,radius=radius,feasible=solution is not None)
            if solution is not None:
                delta,predicted,events,info=solution
                r.update(delta=delta.tolist(),predicted_drop=predicted,event_margins_mm=events.tolist(),**info)
            proposals.append(r);print('WORST_BOUNDARY',name,radius,None if solution is None else predicted,flush=True)
    out=dict(measured_center_coordinates=center.tolist(),active_measurement_box_coordinates=int(sum(abs(center)>=2.-1e-8)),
        active_actor_coefficient_bounds=int(sum(abs(u)>=3.-1e-8)),proposals=proposals,
        new_physical_runs=0,interpretation='Offline local model only. A better recentered prediction requires freshly '
        'measured current directional returns before further actor updates; does not prove full-space feasibility.')
    write_json(STAGE/'search_boundary_audit.json',out)
    print('WORST_BOUNDARY_ACTIVE',out['active_measurement_box_coordinates'],out['active_actor_coefficient_bounds'],flush=True)


if __name__=='__main__':main()
