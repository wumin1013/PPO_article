"""Full-record kinematic utilization statistics, including startup and rest."""
import numpy as np
from revise_nnc_utilization_figures import OLD, OUT, read, write, sha, METHODS, LIMITS

def main():
    b=read(OLD/'paper_result_binding.json'); result=[]
    for case in ['original','dense']:
        ref=np.loadtxt(b['cases'][case]['reference'],delimiter=',',skiprows=1)
        for name in METHODS:
            entry=b['methods'][name][case][0]
            assert sha(entry['trace'])==entry['trace_sha256']
            z=np.load(entry['trace']); xy=z['rows'][:,:2]; h=.001
            v=np.diff(np.vstack([ref[0],xy]),axis=0)/h
            a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/h
            j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/h
            u=np.column_stack([np.linalg.norm(x,axis=1)/cap for x,cap in zip([v,a,j],LIMITS)])
            row=dict(case=case,method=name,ticks=len(xy),mean_V_percent=float(u[:,0].mean()*100),
                     p95_A_percent=float(np.percentile(u[:,1],95)*100),p95_J_percent=float(np.percentile(u[:,2],95)*100),
                     peaks_percent=(u.max(axis=0)*100).tolist(),source=entry['trace'],source_sha256=entry['trace_sha256'])
            assert abs(row['mean_V_percent']-entry['executed_distance_mm']/entry['complete_time_s'])<1e-7
            result.append(row)
    write(OUT/'utilization_statistics.json',dict(interval_s=.001,scope='Complete task including startup, all ordered traversal, braking and rest; equal physical-time sample weights; XY-derived planar norms; numpy percentile linear interpolation.',records=result))
    print([{k:v for k,v in r.items() if k not in ['source','source_sha256','peaks_percent']} for r in result])

if __name__=='__main__':main()
