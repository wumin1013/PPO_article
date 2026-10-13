"""Matched credit-sampling comparison, including actual complete-policy outcomes."""
import numpy as np
from study_io import BASE,write_json,sha
from search_complete_return import read


def inspect(stage,weighted):
    episodes = read(stage/'on_policy_episodes.json')['episodes']
    scale = read(stage/'ppo_update_audit.json')['global_advantage_scale']
    gradients,rows = {},[]
    for e in episodes:
        samples = e['samples']
        design = np.array([s['design'] for s in samples])
        sigma = np.array([s['sigma'] for s in samples])
        noise = np.array([s['noise'] for s in samples])
        cf = np.array([s['counterfactual_advantage'] for s in samples])
        common = np.array([s['baseline_advantage'] for s in samples])[:,None]
        weights = np.array([s['query_weight'] for s in samples]) if weighted else np.full(len(samples),44./3.)
        normalized = np.divide(design,sigma[:,None,:],out=np.zeros_like(design),where=sigma[:,None,:]>0.)
        g = np.sum(normalized*(noise*cf*weights[:,None]/scale)[:,None,:],axis=0).ravel()
        g_common = np.sum(normalized*(noise*common*weights[:,None]/scale)[:,None,:],axis=0).ravel()
        gradients[e['path'],e['seed']] = g
        delta = np.array([r['total_return']-e['observed_return'] for r in e['counterfactuals']])
        rows.append(dict(path=e['path'],seed=e['seed'],queries=[s['decision'] for s in samples],
            observed_return=e['observed_return'],head_credit_norm=float(np.linalg.norm(g)),
            common_credit_norm=float(np.linalg.norm(g_common)),
            counterfactual_return_change_rms=float(np.sqrt(np.mean(delta**2))),
            counterfactual_return_change_max=float(abs(delta).max()),
            new_suffixes=e.get('new_counterfactual_suffixes',len(e['counterfactuals'])),
            reused_suffixes=e.get('reused_counterfactual_suffixes',0)))
    cosines = []
    for path in sorted({e['path'] for e in episodes}):
        g1,g2 = [v for (p,s),v in sorted(gradients.items()) if p==path]
        cosines.append(dict(path=path,replicate_gradient_cosine=float(g1@g2/max(np.linalg.norm(g1)*np.linalg.norm(g2),1e-30))))
    history = read(stage/'state_joint/results.json')['candidates']
    outcomes = [dict(tag=c['tag'],loss=c['score'],loss_change=c['score']-history[0]['score'],
        successes=sum(r['success'] for r in c['records']),paths=len(c['records']),
        guard_min=min(c['augmented_measured_inequalities']),
        guard_violations=sum(v < -1e-8 for v in c['augmented_measured_inequalities'])) for c in history]
    return dict(receipt_sha256=sha(stage/'receipt.json'),episodes=rows,gradient_consistency=cosines,
        outcomes=outcomes,global_advantage_scale=scale)


def main():
    old = BASE/'canonical_gate_band_head_counterfactual_ppo_collection_20260920'
    new = BASE/'canonical_gate_band_stratified_head_credit_ppo_update_20260920'
    a,b = inspect(old,False),inspect(new,True)
    assert a['global_advantage_scale']==b['global_advantage_scale']
    returns = {(r['path'],r['seed']):r['observed_return'] for r in a['episodes']}
    assert all(returns[r['path'],r['seed']]==r['observed_return'] for r in b['episodes'])
    result = dict(uniform=a,stratified=b,same_stochastic_episodes=True,same_queries_per_episode=3,
        changed_sampling_inclusion_weights=True,new_physical_runs_by_review=0,
        independent_seeds_or_generalization_claim=False,adopted=False,goal_complete=False)
    write_json(new/'matched_sampling_credit_review.json',result)
    print('MATCHED_SAMPLING_OUTCOMES',{'uniform':a['outcomes'],'stratified':b['outcomes']},flush=True)
    print('GRADIENT_CONSISTENCY',{'uniform':a['gradient_consistency'],'stratified':b['gradient_consistency']},flush=True)


if __name__=='__main__':
    main()
