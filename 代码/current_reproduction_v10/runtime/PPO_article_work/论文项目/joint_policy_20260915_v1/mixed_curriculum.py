"""Frozen development curriculum: single/consecutive corners, arcs and loops."""
import math
import numpy as np
from geometry_spec import Corridor
from inspect_geometry_semantics import nonadjacent_overlaps


def polyline(headings,lengths):
    headings = np.asarray(headings,float)
    vectors = np.asarray(lengths)[:,None]*np.c_[np.cos(headings),np.sin(headings)]
    return np.vstack([np.zeros(2),np.cumsum(vectors,axis=0)])


def line_arc(radius,sweep,sign=1.):
    angles = np.linspace(-np.pi/2,-np.pi/2+math.radians(sweep),int(math.ceil(radius*math.radians(sweep)/8.))+1)
    arc = np.array([25.,radius])+radius*np.c_[np.cos(angles),np.sin(angles)]
    tangent = np.array([math.cos(math.radians(sweep)),math.sin(math.radians(sweep))])
    ref = np.vstack([np.zeros(2),arc,arc[-1]+30*tangent])
    ref[:,1] *= sign
    return ref


def circle(radius,sign=1.):
    n = max(12,int(math.ceil(2*math.pi*radius/10.)))
    angles = np.linspace(-math.pi/2,-math.pi/2+sign*2*math.pi,n+1)
    ref = radius*np.c_[np.cos(angles),np.sin(angles)]
    ref -= ref[0]
    ref[-1] = ref[0]
    return ref


def _check(ref):
    closed = bool(np.linalg.norm(ref[0]-ref[-1])<1e-9)
    corridor = Corridor(ref,closed=closed)
    return not (corridor.ambiguities or corridor.adjacent_gate_conflicts() or nonadjacent_overlaps(corridor))


def make_paths():
    rng = np.random.default_rng(915082)
    train = []
    for angle in [20,35,60,80,100,120,140,155]:
        for sign in [-1,1]:
            train.append(polyline([0,math.radians(sign*angle)],rng.uniform(20,70,2)))
    attempts = 0
    while len(train)<32:
        attempts += 1
        if attempts>256: raise RuntimeError('Frozen geometry-generation budget exhausted')
        first,second = rng.uniform(35,155,2)
        sign = rng.choice([-1,1])
        ref = polyline(np.deg2rad([0,sign*first,0,-sign*second,0]),
            [rng.uniform(20,60),rng.uniform(8,24),rng.uniform(8,24),rng.uniform(8,24),rng.uniform(20,60)])
        if _check(ref): train.append(ref)
    for radius,sweep,sign in [(20,50,1),(25,85,-1),(30,110,1),(40,140,-1),
                              (45,65,1),(50,105,-1),(60,125,1),(65,150,-1)]:
        train.append(line_arc(radius,sweep,sign))
    for radius,sign in [(22,1),(25,-1),(30,1),(35,-1),(40,1),(45,-1),(50,1),(55,-1)]:
        train.append(circle(radius,sign))
    valid = [polyline([0,math.radians(angle)],[32,24]) for angle in [50,95,125,150]]
    for first,second in [(45,135),(75,150),(115,55),(145,95)]:
        valid.append(polyline(np.deg2rad([0,first,0,-second,0]),[36,18,20,16,28]))
    valid += [line_arc(35,80,-1),line_arc(55,115,1),circle(32,-1),circle(48,1)]
    for ref in train+valid:
        if not _check(ref): raise ValueError('Unresolved predefined curriculum geometry')
    return train,valid


def environment_factory(reference,*,fixed_d=None,max_time=None):
    from joint_runtime_v2 import JointRuntimeV2
    length = np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    return JointRuntimeV2(reference,fixed_d=fixed_d,max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
