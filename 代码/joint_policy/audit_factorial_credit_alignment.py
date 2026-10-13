"""Compare saved stochastic and deterministic credit in the identical 54-weight actor."""
import numpy as np
from search_complete_return import read
from study_io import BASE,write_json,sha


def cosine(a,b):
    return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def main():
    stage=BASE/'canonical_gate_band_factorial_joint_ppo_20260920'
    deterministic=BASE/'canonical_gate_band_deterministic_full_support_credit_20260920'
    episodes=read(stage/'factorial_episodes.json')['episodes']
    de=read(deterministic/'deterministic_episodes.json')['episodes']
    parent_by={e['path']:e for e in de}
    contributions={};gradients={}
    for e in de:
        by={(b['decision'],b['head'],b['radius'],b['sign']):b for b in e['branches']}
        total=np.zeros((18,3))
        for s in e['samples']:
            j=s['decision'];design=np.array(s['design']);g=np.zeros_like(design)
            for h in range(3):
                lo,hi=[by[j,h,.1,sign] for sign in (-1,1)]
                g[:,h]=design[:,h]*(hi['score']-lo['score'])/(2*hi['epsilon'])
            contributions[e['path'],j]=g.ravel();total+=g
        gradients[e['path']]=total.ravel()
    draws=[];stochastic={};sampled={}
    for e in episodes:
        sg=np.zeros(54);dg=np.zeros(54);drifts=[]
        ps={s['decision']:s for s in parent_by[e['path']]['samples']}
        for t in e['tables']:
            s=t['sample'];design=np.array(s['design']);noise=np.array([c['noise'] for c in t['cells']])
            q=np.array([c['total_return'] for c in t['cells']]);advantage=q-q[np.arange(8)^7]
            g=-(design/np.array(s['sigma'])*np.mean(noise*advantage[:,None],axis=0)).ravel()
            sg+=s['query_weight']*g
            dg+=s['query_weight']*contributions[e['path'],s['decision']]
            p=ps[s['decision']]
            drifts.append(float(np.linalg.norm(np.array(p['design'])-design)/max(np.linalg.norm(p['design']),1e-30)))
        stochastic.setdefault(e['path'],[]).append(sg);sampled.setdefault(e['path'],[]).append(dg)
        draws.append(dict(path=e['path'],seed=e['seed'],stochastic_cost_gradient=sg.tolist(),
            matched_query_deterministic_gradient=dg.tolist(),cosine_matched_query=cosine(sg,dg),
            cosine_complete_path=cosine(sg,gradients[e['path']]),max_design_relative_drift=max(drifts)))
    g_all=sum(gradients.values());g_train=sum(gradients[p] for p in stochastic)
    g_stochastic=sum(np.mean(v,axis=0) for v in stochastic.values())
    g_sampled=sum(np.mean(v,axis=0) for v in sampled.values())
    history=read(stage/'state_joint/results.json')['candidates'];checks=[]
    for c in history[1:]:
        u=np.array(c['coefficients'])
        checks.append(dict(tag=c['tag'],stochastic_predicted_cost_change=float(g_stochastic@u),
            deterministic_matched_query_predicted_change=float(g_sampled@u),
            deterministic_full_training_paths_change=float(g_train@u),
            deterministic_all7_predicted_change=float(g_all@u),
            deterministic_all7_actual_change=c['score']-history[0]['score'],
            per_path=[dict(path=r['path'],predicted_change=float(gradients[r['path']]@u)) for r in c['records']]))
    stability=[dict(path=p,replicate_gradient_cosine=cosine(v[0],v[1]),
        stochastic_vs_complete_cosine=cosine(np.mean(v,axis=0),gradients[p])) for p,v in stochastic.items()]
    out=dict(source_receipt_sha256=sha(stage/'receipt.json'),
        deterministic_receipt_sha256=sha(deterministic/'receipt.json'),draws=draws,replicate_stability=stability,
        aggregate_cosine_stochastic_vs_matched_deterministic=cosine(g_stochastic,g_sampled),
        aggregate_cosine_stochastic_vs_complete_training_deterministic=cosine(g_stochastic,g_train),
        aggregate_cosine_stochastic_vs_all7_deterministic=cosine(g_stochastic,g_all),updates=checks,
        new_physical_runs=0,scope='Saved finite-sample and finite-difference estimates; '
        'state/future-noise and gradient-sampling changes are not isolated causal factors.')
    write_json(stage/'deterministic_alignment_audit.json',out)
    print('CREDIT_ALIGNMENT', {k:v for k,v in out.items() if k.startswith('aggregate_')},flush=True)
    print('REPLICATE_STABILITY',stability,flush=True)
    for c in checks:print('UPDATE_ALIGNMENT',{k:v for k,v in c.items() if k!='per_path'},flush=True)


if __name__=='__main__':main()
