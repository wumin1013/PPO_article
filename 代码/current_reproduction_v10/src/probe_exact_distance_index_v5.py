"""Read-only equivalence/cost probe; does not change the common metrics module."""
from pathlib import Path
import json
import time
import numpy as np
from exact_distance_index_v5 import SegmentDistanceIndex
import copp_path_adapter_v4 as core

ROOT = Path(__file__).resolve().parents[1]


def exhaustive(reference, points):
    start = reference[:-1]
    vectors = np.diff(reference, axis=0)
    lengths2 = np.linalg.norm(vectors, axis=1)**2
    output = []
    for query in np.array_split(points, max(1, len(points)//500)):
        delta = query[:, None, :]-start[None, :, :]
        fraction = np.clip(np.sum(delta*vectors, axis=2)/lengths2, 0., 1.)
        output.extend(np.linalg.norm(delta-fraction[:, :, None]*vectors, axis=2).min(axis=1))
    return np.asarray(output)


def main():
    folder = ROOT/'results/copp_v5b_complete_development'
    trace = folder/'original_iteration1_dt1000us.npz'
    data = np.load(trace)
    reference, positions = data['reference'], data['positions']
    start = time.perf_counter()
    index = SegmentDistanceIndex(reference)
    distance = index.distances(positions)
    indexed_wall = time.perf_counter()-start
    record = json.loads((folder/'original.json').read_text('utf-8'))
    published = record['iterations'][0]['runs'][0]['metrics']
    chosen = np.linspace(0, len(positions)-1, 10001, dtype=int)
    rng = np.random.default_rng(88031)
    adversarial = rng.uniform(reference.min(axis=0)-100., reference.max(axis=0)+100., size=(500, 2))
    queries = np.vstack([positions[chosen], reference, adversarial])
    started = time.perf_counter()
    expected = exhaustive(reference, queries)
    exhaustive_wall = time.perf_counter()-started
    observed = index.distances(queries)
    error = float(np.max(abs(expected-observed)))
    maximum, rms = float(distance.max()), float(np.sqrt(np.mean(distance**2)))
    report = dict(scope='Read-only distance-statistics equivalence probe; common metrics.py and running experiments unchanged.',
        source_trace=str(trace), source_trace_sha256=core.sha(trace), samples=len(positions),
        independently_compared_queries=len(queries), maximum_per_query_error_mm=error,
        exhaustive_subset_wall_s=exhaustive_wall, indexed_full_trace_wall_s=indexed_wall,
        stored_full_trace_maximum_mm=published['max_polyline_deviation_mm'], indexed_full_trace_maximum_mm=maximum,
        stored_full_trace_rms_mm=published['rms_polyline_deviation_mm'], indexed_full_trace_rms_mm=rms,
        source_sha256={name: core.sha(ROOT/'src'/name) for name in ['exact_distance_index_v5.py', 'probe_exact_distance_index_v5.py']})
    assert error < 1e-10
    assert abs(maximum-published['max_polyline_deviation_mm']) < 1e-10
    assert abs(rms-published['rms_polyline_deviation_mm']) < 1e-10
    core.write(ROOT/'audit/exact_distance_index_probe_v5.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
