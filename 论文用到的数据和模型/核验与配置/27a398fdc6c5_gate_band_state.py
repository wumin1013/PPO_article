"""Continuous state support across lateral gate endpoints; coverage is unchanged."""
import numpy as np
from canonical_gate_state import CanonicalGateRuntime,canonical_observation


class GateBandRuntime(CanonicalGateRuntime):
    def observation(self):
        # Ordered adjacent gates select the channel; their longitudinal band
        # gives continuous state support without a lateral Boolean cutoff.
        return canonical_observation(self,gate_channel_support=False)


def environment_factory(reference,*,fixed_d=None,max_time=None):
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    return GateBandRuntime(reference,fixed_d=fixed_d,
        max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
