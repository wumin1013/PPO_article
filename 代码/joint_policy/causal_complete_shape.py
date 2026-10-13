"""Causal increments of the existing independent-edge complete shape objective."""
import math
import numpy as np


class CausalCompleteShape:
    """Consume raw 1ms rows; previously charged costs never depend on future rows.

    The energy grid is fixed at the initial XY and advances in actual arc length.
    A turn is charged only after its third spatial point has become observable.
    The complete scalar equals the independent_zero objective already audited.
    This utility does not change the controller or extend isolated-corner metrics
    to multi-corner geometry.
    """
    def __init__(self, reference, spacing=.1):
        ref=np.asarray(reference,float)
        self.active=len(ref)==3
        self.spacing=float(spacing)
        self.previous=ref[0].copy();self.arc=0.;self.grid_index=1
        self.spatial_previous=ref[0].copy();self.spatial_arc=0.
        self.points=[ref[0].copy()]
        self.energy=0.;self.reverse=0.;self.in_outward=0.;self.out_outward=0.;self.cost=0.
        self.ticks=0
        if self.active:
            u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
            self.angle=math.atan2(u[0]*v[1]-u[1]*v[0],u@v)
            self.active=abs(self.angle)>1e-12
            self.sign=np.sign(self.angle)
            self.vertex=ref[1].copy()
            self.ni=self.sign*np.array([-u[1],u[0]])
            self.no=self.sign*np.array([-v[1],v[0]])

    def consume(self, rows):
        old=self.cost
        for row in rows:
            self.ticks+=1
            if not self.active:continue
            position=np.asarray(row[:2]);displacement=position-self.previous
            distance=float(np.linalg.norm(displacement));new_arc=self.arc+distance
            if distance>1e-10:
                while self.grid_index*self.spacing<new_arc:
                    location=self.grid_index*self.spacing
                    point=self.spatial_previous+(location-self.spatial_arc)/(new_arc-self.spatial_arc)*(position-self.spatial_previous)
                    self.points.append(point)
                    if len(self.points)>=3:
                        a=self.points[-2]-self.points[-3];b=self.points[-1]-self.points[-2]
                        turn=math.atan2(a[0]*b[1]-a[1]*b[0],a@b)
                        self.energy+=turn*turn/self.spacing
                        self.points=self.points[-2:]
                    self.grid_index+=1
                self.spatial_previous=position.copy();self.spatial_arc=new_arc
            # Original grid keeps accumulated arc even across discarded near-zero segments.
            self.arc=new_arc;self.previous=position.copy()
            xy=position-self.vertex
            self.in_outward=max(self.in_outward,float(-xy@self.ni))
            self.out_outward=max(self.out_outward,float(-xy@self.no))
            speed_squared=float(row[2:4]@row[2:4])
            if speed_squared>25.:
                omega=self.sign*(row[2]*row[5]-row[3]*row[4])/speed_squared
                self.reverse+=max(-omega,0.)*.001
            q=(1.25*self.energy/self.angle**2+
               5*(self.in_outward+self.out_outward)/1.25+2*self.reverse/abs(self.angle))
            self.cost=float(-np.expm1(-q))
        assert self.cost>=old-1e-14 and 0<=self.cost<=1
        return self.cost-old


def causal_shape_pieces(reference, rows, action_count):
    accumulator=CausalCompleteShape(reference)
    pieces=[]
    for index in range(action_count):
        end=(index+1)*20 if index<action_count-1 else len(rows)
        pieces.append(accumulator.consume(rows[index*20:end]))
    return np.asarray(pieces),accumulator
