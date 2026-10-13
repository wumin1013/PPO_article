"""Replay whether exploratory SVD retains a feasible full-space correction."""
import numpy as np
from audit_worst_search_boundary import setup
from joint_response_models import fit_local_response
from worst_guard_response import solve
from study_io import BASE,write_json,sha
from search_complete_return import read


STAGE=BASE/'canonical_gate_band_recentered_minimax_joint_learning_20260920'


def main():
    _,_,prior,transform,base,_,event_prior,_,normalizer,specs=setup()
    history=read(STAGE/'state_joint/results.json')['candidates'];by={c['tag']:c for c in history}
    output=[]
    for row in read(STAGE/'rounds.json'):
        ri=row['round'];anchor=by[row['anchor']];a=np.array(anchor['subspace_coordinates'])
        end=next(i for i,c in enumerate(history) if c['tag']==f'round{ri}_response0_plus')
        available=[c for c in history[:end] if c['physical_pass'] and c['event_support_valid']]
        model,info=fit_local_response([dict(c,coefficients=c['subspace_coordinates']) for c in available],
            dict(anchor,coefficients=anchor['subspace_coordinates']),np.ones(18),prior)
        near=[c for c in available if c['tag']!=anchor['tag'] and np.max(abs(np.array(c['subspace_coordinates'])-a))<=4.]
        x=np.array([np.array(c['subspace_coordinates'])-a for c in near])
        y=np.array([np.array(c['completion_event_margins_mm'])-anchor['completion_event_margins_mm'] for c in near])
        w=1./np.maximum(np.linalg.norm(x,axis=1),.25)
        emodel=np.linalg.solve((x*w[:,None]).T@x+.01*np.eye(18),(x*w[:,None]).T@y+.01*event_prior)
        with np.load(STAGE/f'round{ri}_current_response.npz') as z:
            basis,local_t,buffer,fresh,efresh=z['basis'],z['transform'],z['buffer'],z['fresh'],z['event']
        mapping=np.linalg.lstsq(transform,local_t,rcond=None)[0]
        local=dict(anchor,subspace_coordinates=[0.]*basis.shape[1]);u=np.array(anchor['coefficients'])
        full=solve(dict(anchor,subspace_coordinates=[0.]*18),transform,model,emodel,buffer,.5,normalizer,u,specs)
        projected=solve(local,local_t,mapping.T@model,mapping.T@emodel,buffer,1.,normalizer,u,specs)
        current=solve(local,local_t,fresh,efresh,buffer,1.,normalizer,u,specs)
        result=dict(round=ri,anchor=anchor['tag'],full18_predicted_drop=None if full is None else full[1],
            projected_prior3_predicted_drop=None if projected is None else projected[1],
            current_measured3_predicted_drop=None if current is None else current[1])
        if full is not None:
            d=full[0];residual=d-basis@(basis.T@d)
            result.update(full_direction=d.tolist(),direction_l2=float(np.linalg.norm(d)),
                relative_projection_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(d),1e-30)))
        output.append(result);print('SUBSPACE_RETENTION',result,flush=True)
    write_json(STAGE/'subspace_retention_audit.json',dict(records=output,new_physical_runs=0,
        results_sha256=sha(STAGE/'state_joint/results.json'),
        scope='Exact exploration model replay; a missing predicted feasible direction is not measured full-space feasibility.'))


if __name__=='__main__':main()
