"""Offline credit within a frozen brake branch, with explicit branch guards."""
import numpy as np
from joint_runtime import terminal
from probe_closed_loop_tangent_credit import STATE_SCALE


def fixed_duration_rows(state6, count, length):
    """Evaluate the existing cubic family, for derivatives only."""
    v, a, _ = terminal.discrete_planar_stop_profile(state6[2:4], state6[4:6], .001, count)
    assert length >= count
    v = np.pad(v[1:], ((0, length-count), (0, 0)))
    a = np.pad(a[1:], ((0, length-count), (0, 0)))
    return np.column_stack([state6[:2]+np.cumsum(v*.001, axis=0), v, a])


def fixed_duration_jacobian(count, length):
    jac = np.zeros((length, 6, 10))
    for axis in range(6):
        unit = np.zeros(6); unit[axis] = STATE_SCALE[axis]
        jac[:, :, axis] = fixed_duration_rows(unit, count, length)
    return jac


def normalized_profile_maps(count):
    """Linear maps from normalized initial v/a to normalized v/a/j samples."""
    limits=np.array([100.,2000.,62500.*terminal.PlanarBrakingPolicy().limit_fraction])
    maps=np.empty((3,count,2,4))
    for axis in range(4):
        state=np.zeros(4);state[axis]=STATE_SCALE[axis+2]
        profiles=terminal.discrete_planar_stop_profile(state[:2],state[2:],.001,count)
        for channel,profile in enumerate(profiles):maps[channel,:,:,axis]=profile[1:]/limits[channel]
    return maps


class TerminalDurationGuard:
    """Convex sufficient conditions for retaining the nominal selected count.

    Keep the selected duration feasible, and retain a norm-violation witness
    for EVERY shorter count. The nonreversal condition uses a global quadratic
    lower bound. These guards concern predicted brake initial states only;
    ordinary-state prediction errors still require separate validation.
    """
    def __init__(self,state6,response,count):
        self.count=count;self.state=np.asarray(state6)[2:6]/STATE_SCALE[2:6]
        self.response=np.asarray(response)[2:6]
        self.maps=normalized_profile_maps(count)
        nominal=np.einsum('ctij,j->cti',self.maps,self.state)
        self.peak=np.linalg.norm(nominal,axis=2).max(axis=1)
        assert np.all(self.peak<=1.+1e-10)
        self.bound=(1.+self.peak)/2.
        witnesses=[];margins=[]
        for shorter in range(2,count):
            maps=normalized_profile_maps(shorter)
            values=np.einsum('ctij,j->cti',maps,self.state)
            norms=np.linalg.norm(values,axis=2);c,t=np.unravel_index(norms.argmax(),norms.shape)
            value=float(norms[c,t])
            if value<=1.+1e-12:
                raise ValueError(f'Count {shorter} needs a nonreversal-only infeasibility witness')
            witnesses.append(values[c,t]@maps[c,t]/value)
            margins.append((value-1.)/2.)
        self.witness=np.asarray(witnesses).reshape(-1,4)
        self.margin=np.asarray(margins)
        # v[k] dot v[0] has its most restrictive nonzero bracket at k=n-2.
        # The common positive (n-k)(n-1-k) factor is omitted.
        n=count;k=n-2
        cv=1/(n*(n-1))+k*(1/(n*(n-1))-1/(n*(n+1)))
        ca=k*.001/(n*(n+1))
        self.qmatrix=np.block([[cv*10000*np.eye(2),ca*100*2000/2*np.eye(2)],
                              [ca*100*2000/2*np.eye(2),np.zeros((2,2))]])
        self.q0=float(self.state@self.qmatrix@self.state)
        self.qgradient=2*self.qmatrix@self.state
        self.lower_eigenvalue=min(0.,float(np.linalg.eigvalsh(self.qmatrix).min()))
        assert self.q0>=-1e-12

    def evaluate(self,direction):
        delta=self.response@direction;state=self.state+delta;out=[]
        for channel in range(2):
            part=slice(2*channel,2*channel+2);norm=np.linalg.norm(state[part])
            derivative=np.zeros(4);derivative[part]=state[part]/max(norm,1e-30)
            out.append((float(norm-1.),derivative@self.response,1.,f'initial_state_channel{channel}'))
        values=np.einsum('ctij,j->cti',self.maps,state)
        norms=np.linalg.norm(values,axis=2)
        for channel in range(3):
            t=int(norms[channel].argmax());norm=norms[channel,t]
            derivative=values[channel,t]@self.maps[channel,t]/max(norm,1e-30)
            out.append((float(norm-self.bound[channel]),derivative@self.response,
                        max(1.-self.bound[channel],1e-8),f'selected_duration_channel{channel}'))
        if len(self.margin):
            value=-self.witness@delta-self.margin;k=int(value.argmax())
            out.append((float(value[k]),-self.witness[k]@self.response,
                        max(float(self.margin[k]),1e-8),'shorter_duration_infeasible'))
        if self.count>2:
            value=-.5*self.q0-self.qgradient@delta-self.lower_eigenvalue*(delta@delta)
            derivative=-self.qgradient-2*self.lower_eigenvalue*delta
            out.append((float(value),derivative@self.response,max(.5*self.q0,1e-8),'nonreversing_brake'))
        return out
