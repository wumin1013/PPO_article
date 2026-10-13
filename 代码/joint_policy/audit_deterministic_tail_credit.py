"""Measure how much a single sampled memory-tail query influences complete credit."""
import numpy as np
from study_io import BASE,write_json
from search_complete_return import read


def main():
    source=BASE/'canonical_gate_band_deterministic_local_credit_20260920'
    rows=[]
    for e in read(source/'deterministic_episodes.json')['episodes']:
        by={(b['decision'],b['head'],b['radius'],b['sign']):b for b in e['branches']}
        dense=np.zeros((18,3));tail=np.zeros_like(dense)
        queries=[]
        for s in e['samples']:
            g=np.zeros_like(dense)
            for h in range(3):
                minus,plus=[by[s['decision'],h,.1,sign] for sign in (-1,1)]
                g[:,h]=np.array(s['design'])[:,h]*(plus['score']-minus['score'])/(2*plus['epsilon'])
            if s['query_weight']>1:
                tail+=s['query_weight']*g
                queries.append(dict(decision=s['decision'],weight=s['query_weight'],
                    unweighted_gradient_norm=float(np.linalg.norm(g))))
            else:dense+=g
        row=dict(path=e['path'],dense_gradient_norm=float(np.linalg.norm(dense)),
            weighted_tail_gradient_norm=float(np.linalg.norm(tail)),
            tail_fraction_of_norm_sum=float(np.linalg.norm(tail)/max(np.linalg.norm(tail)+np.linalg.norm(dense),1e-12)),
            cosine=float(np.sum(tail*dense)/max(np.linalg.norm(tail)*np.linalg.norm(dense),1e-12)),tail_queries=queries)
        rows.append(row);print('TAIL_CREDIT',row,flush=True)
    write_json(source/'tail_credit_influence_audit.json',dict(rows=rows,new_physical_runs=0,
        single_sample_influence_not_a_variance_estimate=True))


if __name__=='__main__':main()
