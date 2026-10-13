"""Convex fixed-segment enclosures of every affine 1 ms command segment."""
import numpy as np
from audit_function_constraint_interference import distances_to_segments,command_segment_margin


class CapsuleCorridorModel:
    def __init__(self,reference,nominal,gradient,nominal_distance_upper=None):
        reference=np.asarray(reference)
        self.nominal=np.vstack([reference[0],nominal])
        self.gradient=gradient
        if nominal_distance_upper is None:
            nominal_distance_upper=command_segment_margin(reference,nominal)['max_distance_upper_mm']
        def interpolate(t):
            left=np.floor(t).astype(int);right=np.minimum(left+1,len(nominal))
            alpha=t-left
            return (1.-alpha[:,None])*self.nominal[left]+alpha[:,None]*self.nominal[right]
        a=np.arange(len(nominal),dtype=float);b=a+1.;accepted=[]
        # Subdivide only geometric intersections. Execution stays at 1 ms.
        # Each accepted affine subinterval has both endpoints in one capsule.
        for depth in range(26):
            da=distances_to_segments(reference,interpolate(a))
            db=distances_to_segments(reference,interpolate(b))
            maxima=np.maximum(da,db);assignment=maxima.argmin(axis=1)
            upper=maxima[np.arange(len(a)),assignment]
            good=upper<=nominal_distance_upper+1e-10
            accepted.append(np.column_stack([a[good],b[good],assignment[good]]))
            if good.all():break
            a,b=a[~good],b[~good];middle=.5*(a+b)
            a,b=np.r_[a,middle],np.r_[middle,b]
        else:raise ValueError('Nominal capsule subdivision did not converge')
        intervals=np.concatenate(accepted)
        pairs=np.unique(np.concatenate([intervals[:,[0,2]],intervals[:,[1,2]]]),axis=0)
        self.sample_coordinate=pairs[:,0]
        self.point_index=np.floor(self.sample_coordinate).astype(int)
        self.right_index=np.minimum(self.point_index+1,len(nominal))
        self.alpha=self.sample_coordinate-self.point_index
        self.segment_index=pairs[:,1].astype(int)
        self.starts=reference[:-1][self.segment_index]
        self.vectors=np.diff(reference,axis=0)[self.segment_index]
        self.squared_length=np.sum(self.vectors**2,axis=1)
        self.intervals=intervals
        self.refinement_depth=depth

    def evaluate(self,direction):
        xy=self.nominal.copy()
        xy[1:]+=np.einsum('tcd,d->tc',self.gradient,direction)
        points=(1.-self.alpha[:,None])*xy[self.point_index]+self.alpha[:,None]*xy[self.right_index]
        offsets=points-self.starts
        projection=np.clip(np.sum(offsets*self.vectors,axis=1)/self.squared_length,0.,1.)
        residual=offsets-projection[:,None]*self.vectors
        distance=np.linalg.norm(residual,axis=1)
        active=int(distance.argmax());point=int(self.point_index[active])
        row=np.zeros(self.gradient.shape[-1]);alpha=self.alpha[active]
        normal=residual[active]/max(distance[active],1e-30)
        if point:row+=(1.-alpha)*(normal@self.gradient[point-1])
        if alpha:row+=alpha*(normal@self.gradient[self.right_index[active]-1])
        return float(distance[active]),row,dict(sample_coordinate=float(self.sample_coordinate[active]),
            reference_segment=int(self.segment_index[active]))
