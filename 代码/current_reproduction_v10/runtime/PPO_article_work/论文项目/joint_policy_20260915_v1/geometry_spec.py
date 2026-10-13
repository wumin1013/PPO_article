"""Candidate Euclidean corridor and directed gate semantics.

This module does not replace the historical checkpoint-coverage audit. A gate
sequence is an explicit task definition, not a universal anti-shortcut proof.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

NUMERIC_MM = 1e-9


def point_segment_distance(point, start, end):
    d = end - start
    den = float(d @ d)
    t = float(np.clip((point-start) @ d / den, 0., 1.)) if den else 0.
    return float(np.linalg.norm(point-(start+t*d)))


def _linear_interval(value, rate, lower, upper):
    if abs(rate) <= 1e-15:
        return (0., 1.) if lower-NUMERIC_MM <= value <= upper+NUMERIC_MM else None
    t0, t1 = (lower-value)/rate, (upper-value)/rate
    left, right = max(0., min(t0,t1)), min(1., max(t0,t1))
    return (left,right) if left <= right else None


def _circle_interval(start, delta, center, radius):
    den = float(delta @ delta)
    relative = center-start
    if den <= 1e-30:
        return (0.,1.) if np.linalg.norm(relative) <= radius+NUMERIC_MM else None
    tmid = float(relative @ delta / den)
    perpendicular = relative - tmid*delta
    disc = radius*radius - float(perpendicular @ perpendicular)
    if disc < -2.*radius*NUMERIC_MM:
        return None
    half = math.sqrt(max(0.,disc)/den)
    left,right = max(0.,tmid-half), min(1.,tmid+half)
    return (left,right) if left <= right else None


def capsule_intervals(start, end, a, b, radius):
    """Intersection parameters of a command segment with a finite capsule."""
    delta = end-start
    axis = b-a
    length = float(np.linalg.norm(axis))
    if length <= 1e-15:
        found = _circle_interval(start,delta,a,radius)
        return [] if found is None else [found]
    tangent = axis/length
    normal = np.array([-tangent[1],tangent[0]])
    rel = start-a
    along = _linear_interval(float(rel@tangent),float(delta@tangent),0.,length)
    across = _linear_interval(float(rel@normal),float(delta@normal),-radius,radius)
    intervals=[]
    if along is not None and across is not None:
        lo,hi=max(along[0],across[0]),min(along[1],across[1])
        if lo<=hi:
            intervals.append((lo,hi))
    for center in (a,b):
        interval=_circle_interval(start,delta,center,radius)
        if interval is not None:
            intervals.append(interval)
    return intervals


@dataclass(frozen=True)
class Gate:
    vertex_index: int
    point: np.ndarray
    normal: np.ndarray


class Corridor:
    def __init__(self, reference, epsilon=1.25, *, closed=False):
        self.reference=np.asarray(reference,dtype=float).copy()
        self.epsilon=float(epsilon)
        self.closed=bool(closed)
        if self.reference.ndim!=2 or self.reference.shape[1]!=2 or len(self.reference)<2:
            raise ValueError('Need at least two planar reference points')
        if not np.isfinite(self.reference).all() or not math.isfinite(self.epsilon) or not 0.<self.epsilon:
            raise ValueError('Reference and tolerance must be finite and positive')
        self.starts,self.ends=self.reference[:-1],self.reference[1:]
        self.vectors=self.ends-self.starts
        self.lengths=np.linalg.norm(self.vectors,axis=1)
        if np.any(self.lengths<=1e-10):
            raise ValueError('Repeated zero-length reference segment needs explicit source handling')
        if self.closed and np.linalg.norm(self.reference[0]-self.reference[-1])>NUMERIC_MM:
            raise ValueError('Closed reference must explicitly end at its start')
        self.tangents=self.vectors/self.lengths[:,None]
        self.arcs=np.r_[0.,np.cumsum(self.lengths)]
        self.box_min=np.minimum(self.starts,self.ends)-self.epsilon-NUMERIC_MM
        self.box_max=np.maximum(self.starts,self.ends)+self.epsilon+NUMERIC_MM
        self.gates=[]
        self.ambiguities=[]
        for i in range(1,len(self.reference)-1):
            summed=self.tangents[i-1]+self.tangents[i]
            norm=float(np.linalg.norm(summed))
            if norm<=1e-8:
                self.ambiguities.append({'kind':'reversal_180','vertex_index':i})
                continue
            self.gates.append(Gate(i,self.reference[i].copy(),summed/norm))

    def candidate_segments(self, start, end):
        lo,hi=np.minimum(start,end),np.maximum(start,end)
        mask=np.all(self.box_max>=lo,axis=1)&np.all(self.box_min<=hi,axis=1)
        return np.flatnonzero(mask)

    def point_distance(self, point):
        point=np.asarray(point,float)
        t=np.clip(np.sum((point-self.starts)*self.vectors,axis=1)/(self.lengths**2),0.,1.)
        return float(np.min(np.linalg.norm(point-(self.starts+t[:,None]*self.vectors),axis=1)))

    def segment_inside(self, start, end):
        start,end=np.asarray(start,float),np.asarray(end,float)
        if not np.isfinite(np.r_[start,end]).all():
            return False
        if np.linalg.norm(end-start)<=1e-15:
            return self.point_distance(start)<=self.epsilon+NUMERIC_MM
        intervals=[]
        for i in self.candidate_segments(start,end):
            intervals.extend(capsule_intervals(start,end,self.starts[i],self.ends[i],self.epsilon))
        frontier=0.
        tolerance_t=NUMERIC_MM/max(float(np.linalg.norm(end-start)),1.)
        for left,right in sorted(intervals):
            if left>frontier+tolerance_t:
                return False
            frontier=max(frontier,right)
            if frontier>=1.-tolerance_t:
                return True
        return False

    def gate_contains(self, gate, point):
        i=gate.vertex_index
        return (abs(float((point-gate.point)@gate.normal))<=NUMERIC_MM*4
                and point_segment_distance(point,self.reference[i-1],self.reference[i])<=self.epsilon+NUMERIC_MM
                and point_segment_distance(point,self.reference[i],self.reference[i+1])<=self.epsilon+NUMERIC_MM)

    def adjacent_gate_conflicts(self):
        conflicts=[]
        for left,right in zip(self.gates,self.gates[1:]):
            matrix=np.stack([left.normal,right.normal])
            if abs(float(np.linalg.det(matrix)))<=1e-10:
                if abs(float((right.point-left.point)@left.normal))<=NUMERIC_MM:
                    conflicts.append({'kind':'coincident_planes','vertices':[left.vertex_index,right.vertex_index]})
                continue
            cross=np.linalg.solve(matrix,np.array([left.normal@left.point,right.normal@right.point]))
            if self.gate_contains(left,cross) and self.gate_contains(right,cross):
                conflicts.append({'kind':'intersecting_gates','vertices':[left.vertex_index,right.vertex_index],
                                  'point':cross.tolist()})
        return conflicts


class GateTracker:
    """Advance on witnessed negative-to-positive crossings, including contact."""
    def __init__(self,corridor,initial_position):
        if corridor.ambiguities:
            raise ValueError('Unresolved path semantics: '+str(corridor.ambiguities))
        self.corridor=corridor
        self.position=np.asarray(initial_position,float).copy()
        self.start_error=float(np.linalg.norm(self.position-corridor.reference[0]))
        self.frontier=0
        self.pending_negative=False
        self.inside=corridor.point_distance(self.position)<=corridor.epsilon+NUMERIC_MM
        self.events=[]
        self.segment_count=0

    def update(self,position):
        end=np.asarray(position,float)
        start=self.position.copy()
        self.inside=bool(self.inside and self.corridor.segment_inside(start,end))
        earliest=-1.
        advanced=0
        while self.frontier<len(self.corridor.gates):
            gate=self.corridor.gates[self.frontier]
            h0=float((start-gate.point)@gate.normal)
            h1=float((end-gate.point)@gate.normal)
            crossing=None
            if h0<-NUMERIC_MM and h1>NUMERIC_MM:
                crossing=-h0/(h1-h0)
            elif abs(h0)<=NUMERIC_MM and h1>NUMERIC_MM and self.pending_negative:
                crossing=0.
            if crossing is None:
                if h0<-NUMERIC_MM and abs(h1)<=NUMERIC_MM:
                    self.pending_negative=self.corridor.gate_contains(gate,end)
                elif h1<-NUMERIC_MM or h1>NUMERIC_MM:
                    self.pending_negative=False
                break
            point=start+crossing*(end-start)
            if crossing<=earliest+1e-12 or not self.corridor.gate_contains(gate,point):
                break
            self.events.append({'gate':self.frontier,'vertex_index':gate.vertex_index,
                                'command_segment':self.segment_count,'fraction':float(crossing),
                                'point':point.tolist()})
            earliest=crossing
            self.frontier+=1
            advanced+=1
            self.pending_negative=False
        self.segment_count+=1
        self.position=end.copy()
        return advanced

    def snapshot(self,endpoint_tolerance=.05):
        error=float(np.linalg.norm(self.position-self.corridor.reference[-1]))
        return {'corridor_inside':self.inside, 'gates_passed':self.frontier,
                'gate_count':len(self.corridor.gates),
                'ordered_complete':self.frontier==len(self.corridor.gates),
                'endpoint_error_mm':error,
                'start_error_mm':self.start_error,
                'geometry_complete':self.inside and self.frontier==len(self.corridor.gates) and error<=endpoint_tolerance and self.start_error<=endpoint_tolerance,
                'kinematic_or_stop_certificate':False}


def audit_commands(reference,commands,*,epsilon=1.25,closed=False,endpoint_tolerance=.05):
    commands=np.asarray(commands,float)
    corridor=Corridor(reference,epsilon,closed=closed)
    tracker=GateTracker(corridor,commands[0])
    for point in commands[1:]:
        tracker.update(point)
    return tracker.snapshot(endpoint_tolerance),tracker.events
