"""Training-teacher activation from current14inputs,without route IDs or memory."""
import math


def strength(observation):
    bend=math.hypot(1.-float(observation[5]),float(observation[13]))
    signal=max(bend,abs(float(observation[6])))
    unit=max(0.,min(1.,(signal-.005)/.145))
    return unit*unit*(3.-2.*unit)


def modify(action,observation,feed_delta,preview_gain):
    result=action.copy();gate=strength(observation)
    result[1]+=gate*feed_delta
    result[2]=((6.4+5.6*float(result[2]))*(1.+gate*(preview_gain-1.))-6.4)/5.6
    return result
