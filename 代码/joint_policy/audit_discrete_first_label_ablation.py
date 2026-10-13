"""Verify default reward compatibility and same-path failure ordering offline."""
import numpy as np
from bounded_shape_return import terminal_shape_cost
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'contextual_learned_function_ppo_20260916'
    stage=register('discrete_first_label_reward_audit_20260916',[__file__,BASE/'bounded_shape_return.py',source/'receipt.json',
        BASE/'turn_onset_label_audit_20260916/receipt.json'],dict(physics_ticks=0,new_decisions=0),
        'Verify removing only the discrete first-turn indicator leaves all other spatial/time/failure '
        'terms unchanged and cost bounded[0,1]. The old default must reproduce every saved episode cost '
        'exactly. Check all saved same-path groups containing both success and failure; analytical '
        'success-over-failure ordering follows unchanged failure penalty and cost range.')
    rows=read(source/'episode_results.json')['episodes'];records=[]
    for r in rows:
        with np.load(source/'episodes'/r['trace']) as z:
            old,_=terminal_shape_cost(z['reference'],z['rows'],z['actions'])
            new,_=terminal_shape_cost(z['reference'],z['rows'],z['actions'],discrete_first_label=False)
            assert np.array_equal(old,z['shape_cost'])
            assert 0<=new.sum()<=old.sum()+1e-12 and new.sum()<=1
        records.append(dict(episode=r['episode'],path=r['path'],success=r['success'],old_return=r['shaped_return'],
                            new_return=r['base_return']-float(new.sum()),old_cost=float(old.sum()),new_cost=float(new.sum())))
    margins=[]
    for group in read(source/'paired_groups.json')['groups']:
        selected=[records[i] for i in group['episodes']];good=[r['new_return'] for r in selected if r['success']];bad=[r['new_return'] for r in selected if not r['success']]
        if good and bad:
            margin=min(good)-max(bad);assert margin>0;margins.append(margin)
    write_json(stage/'results.json',dict(episodes=len(rows),old_default_bitwise_identical=True,
        changed_episodes=sum(abs(r['old_cost']-r['new_cost'])>1e-12 for r in records),same_path_success_failure_margins=margins,
        cost_range=[min(r['new_cost'] for r in records),max(r['new_cost'] for r in records)],records=records,
        proof='Success return>=L/100-Tmax-1; failure return<L/100-Tmax-1 from unchanged Tmax+1+remaining/20 charge and positive elapsed time. Cost remains[0,1].'))
    receipt(stage,dict(status='reward_ablation_contract_verified',physics_ticks=0,new_decisions=0,episodes=len(rows)))
    print('CONTRACT',len(rows),'changed',sum(abs(r['old_cost']-r['new_cost'])>1e-12 for r in records),'margins',margins)


if __name__=='__main__':main()
