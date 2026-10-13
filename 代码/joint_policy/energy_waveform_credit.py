"""Signed square-root bending-energy densities at original 1 ms samples."""
import numpy as np
from prepare_current_waveform_credit import wave_signals


def energy_signals(reference,rows,length=None):
    rows=np.asarray(rows)
    if length is not None:
        if len(rows)<length:
            assert np.linalg.norm(rows[-1,2:4])<.001
            rows=np.vstack([rows,np.repeat(rows[-1:],length-len(rows),axis=0)])
        rows=rows[:length]
    values,valid=wave_signals(reference,rows)
    u,v=np.diff(reference,axis=0)
    angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    lengths=np.linalg.norm(np.diff(np.vstack([reference[0],rows[:,:2]]),axis=0),axis=1)
    measure=np.r_[.5*(lengths[:-1]+lengths[1:]),0.]
    masses=np.column_stack([measure,speed*.001])*1.25/angle**2
    supported=valid[:,[3,4]]
    signals=np.zeros((len(rows),2))
    for j,c in enumerate((3,4)):
        signals[supported[:,j],j]=values[supported[:,j],c]*np.sqrt(masses[supported[:,j],j])
    return signals,supported
