"""Exact finite-polyline distances with a conservative midpoint-tree broad phase.

For a query p, its closest reference-vertex distance D is an upper bound on its
closest segment distance. Every segment that can improve D has its midpoint
within D + half_length <= D + max_half_length. The final distances are the same
finite-segment projections as the exhaustive evaluator, without simplifying
the reference or changing the acceptance tolerance.
"""
import numpy as np
from scipy.spatial import cKDTree


class SegmentDistanceIndex:
    def __init__(self, reference):
        reference = np.asarray(reference, float)
        self.starts = reference[:-1]
        self.vectors = np.diff(reference, axis=0)
        self.lengths2 = np.sum(self.vectors*self.vectors, axis=1)
        if np.any(self.lengths2 <= 0):
            raise ValueError('Nonzero finite reference segments required')
        self.max_half_length = .5*float(np.sqrt(self.lengths2).max())
        self.vertex_tree = cKDTree(reference)
        self.midpoint_tree = cKDTree(self.starts+.5*self.vectors)
        self.coordinate_scale = max(1., float(np.max(abs(reference))))

    def distances(self, points, batch=20000):
        points = np.asarray(points, float)
        answer = np.empty(len(points))
        for start in range(0, len(points), batch):
            query = points[start:start+batch]
            upper, _ = self.vertex_tree.query(query)
            roundoff = 64*np.finfo(float).eps*max(self.coordinate_scale, float(np.max(abs(query))), 1.)
            radius = upper+self.max_half_length+roundoff
            candidates = self.midpoint_tree.query_ball_point(query, radius)
            counts = np.fromiter((len(c) for c in candidates), int, count=len(query))
            if np.any(counts == 0):
                raise AssertionError('Conservative nearest-segment broad phase cannot be empty')
            owners = np.repeat(np.arange(len(query)), counts)
            indices = np.concatenate(candidates).astype(int, copy=False)
            delta = query[owners]-self.starts[indices]
            fraction = np.clip(np.sum(delta*self.vectors[indices], axis=1)/self.lengths2[indices], 0., 1.)
            values = np.linalg.norm(delta-fraction[:, None]*self.vectors[indices], axis=1)
            local = np.full(len(query), np.inf)
            np.minimum.at(local, owners, values)
            answer[start:start+len(query)] = local
        return answer
