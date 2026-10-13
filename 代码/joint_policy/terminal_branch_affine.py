"""Affine tail for one explicitly selected original cubic brake branch."""
import numpy as np
from terminal_duration_credit import fixed_duration_rows


def branch_affine(reference, ordinary_rows, ordinary_response, initiation_tick, count):
    """Return a conditional affine full path; caller MUST recheck branch selection.

    The state response is in physical units. No derivative of the integer
    count or initiation event is asserted. The 23 recorded hold ticks are
    exactly those produced by the original planner.
    """
    tick=int(initiation_tick);length=int(count)+23
    assert tick<=len(ordinary_rows) and tick%20==0
    if tick:
        state=np.asarray(ordinary_rows[tick-1,:6])
        derivative=np.asarray(ordinary_response[tick-1])
    else:
        state=np.r_[reference[0],np.zeros(4)]
        derivative=np.zeros((6,ordinary_response.shape[-1]))
    tail=fixed_duration_rows(state,count,length)
    maps=np.stack([fixed_duration_rows(unit,count,length) for unit in np.eye(6)],axis=-1)
    tail_response=np.einsum('tcs,sd->tcd',maps,derivative)
    return (np.vstack([ordinary_rows[:tick,:6],tail]),
            np.concatenate([ordinary_response[:tick],tail_response],axis=0))
