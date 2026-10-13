"""Offline sufficient witness that an already available brake duration stays feasible."""
import numpy as np
from terminal_duration_credit import normalized_profile_maps, fixed_duration_rows
from probe_closed_loop_tangent_credit import STATE_SCALE


class TerminalCompletionWitness:
    """Allow shorter durations; never freeze the first-feasible branch or runtime.

    Feasibility and endpoint proximity concern this explicit duration only.
    A shorter duration chosen by the original planner can have another endpoint,
    so its actual event must still be checked by the exact predictor and physics.
    """
    def __init__(self,state6,response,count,endpoint):
        self.state=np.asarray(state6)[2:6]/STATE_SCALE[2:6]
        self.response=np.asarray(response)[2:6]/STATE_SCALE[2:6,None]
        self.maps=normalized_profile_maps(count)
        nominal=np.einsum('ctij,j->cti',self.maps,self.state)
        self.bound=.5*(1.+np.linalg.norm(nominal,axis=2).max(axis=1))
        assert np.all(self.bound<=1.+1e-10)
        endmap=np.stack([fixed_duration_rows(unit,count,count)[-1,:2] for unit in np.eye(6)],axis=-1)
        self.end0=endmap@state6-np.asarray(endpoint)
        self.endg=endmap@response
        self.endbound=.5*(1.+np.linalg.norm(self.end0))
        assert self.endbound<=1.+1e-10
        lam=(count-2)*.001/3.
        self.q=np.block([[10000.*np.eye(2),lam*100.*2000./2.*np.eye(2)],
                         [lam*100.*2000./2.*np.eye(2),np.zeros((2,2))]])
        self.q0=float(self.state@self.q@self.state)
        self.qg=2*self.q@self.state
        self.eigen=min(0.,float(np.linalg.eigvalsh(self.q).min()))
        assert self.q0>=-1e-9

    def evaluate(self,direction):
        delta=self.response@direction;state=self.state+delta;out=[]
        for c in range(2):
            part=slice(2*c,2*c+2);norm=np.linalg.norm(state[part])
            g=(state[part]/max(norm,1e-30))@self.response[part]
            out.append((float(norm-1.),g,1.,f'initial_{c}'))
        values=np.einsum('ctij,j->cti',self.maps,state)
        norms=np.linalg.norm(values,axis=2)
        for c in range(3):
            tick=int(norms[c].argmax());norm=norms[c,tick]
            g=values[c,tick]@self.maps[c,tick]@self.response/max(norm,1e-30)
            out.append((float(norm-self.bound[c]),g,max(1.-self.bound[c],1e-6),f'profile_{c}'))
        endpoint=self.end0+self.endg@direction;norm=np.linalg.norm(endpoint)
        out.append((float(norm-self.endbound),endpoint@self.endg/max(norm,1e-30),
                    max(1.-self.endbound,1e-6),'endpoint'))
        value=-.5*self.q0-self.qg@delta-self.eigen*(delta@delta)
        g=(-self.qg-2*self.eigen*delta)@self.response
        out.append((float(value),g,max(.5*self.q0,1e-6),'nonreversal'))
        return out
