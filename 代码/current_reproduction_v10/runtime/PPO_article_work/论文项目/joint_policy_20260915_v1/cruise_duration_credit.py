"""Exact speed margins for minimum cruise counts and sustained cruise durations."""
from functools import lru_cache
import numpy as np
from duration_joint_metrics import guard_names
from heading_duration_credit import learning_guards


@lru_cache(maxsize=2048)
def raw_cruise_slacks(trace, total_count, longest_count):
    with np.load(trace) as z:speed=np.linalg.norm(z['rows'][:,2:4],axis=1)
    moving=speed>=99.
    edge=np.diff(np.r_[False,moving,False].astype(int))
    lengths=np.flatnonzero(edge==-1)-np.flatnonzero(edge==1)
    actual_total=int(moving.sum());actual_longest=int(lengths.max()) if len(lengths) else 0
    if total_count<=0:total_margin=1.
    elif total_count>len(speed):total_margin=-99.
    else:total_margin=float(np.partition(speed,-total_count)[-total_count]-99.)
    if longest_count<=0:longest_margin=1.
    elif longest_count>len(speed):longest_margin=-99.
    else:
        windows=np.lib.stride_tricks.sliding_window_view(speed,longest_count)
        longest_margin=float(np.max(np.min(windows,axis=1))-99.)
    assert (actual_total>=total_count)==(total_margin>=0.)
    assert (actual_longest>=longest_count)==(longest_margin>=0.)
    return total_margin,longest_margin,actual_total,actual_longest


def event_learning_guards(records,baseline,original):
    values=learning_guards(records,baseline,original)
    names=guard_names();n=len(records)
    total_offset=names.index('motion.cruise_ge99_s')
    longest_offset=names.index('motion.longest_cruise_ge99_s')
    for i,(r,b) in enumerate(zip(records,baseline)):
        if not r['success']:continue
        total=int(round(b['motion']['cruise_ge99_s']/.001))
        longest=int(round(b['motion']['longest_cruise_ge99_s']/.001))
        tm,lm,actual_total,actual_longest=raw_cruise_slacks(str(r['trace']),total,longest)
        assert actual_total==int(round(r['motion']['cruise_ge99_s']/.001))
        assert actual_longest==int(round(r['motion']['longest_cruise_ge99_s']/.001))
        loose=raw_cruise_slacks(str(r['trace']),total-20,0)[0]
        replacements={11*i+7:loose,11*n+len(names)*i+total_offset:tm,
                      11*n+len(names)*i+longest_offset:lm}
        for row,margin in replacements.items():
            assert (values[row]>=-1e-8)==(margin>=0.),('cruise_guard_equivalence',r['path'],row)
            values[row]=margin
    return values
