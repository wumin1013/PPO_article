"""Audit first-turn direction of saved diagnostic candidates, without simulation."""
from pathlib import Path
import json,pickle
import numpy as np

ROOT=Path(r'D:\PPO_article_joint_runs\natural_closed_resume\all34_dominance_research')

def main():
    stage=ROOT/'conditional_anticipation_search'
    data=json.loads((stage/'progress.json').read_text(encoding='utf-8'))
    with (ROOT/'selected_causal_contexts/contexts.pkl').open('rb') as f:contexts=pickle.load(f)
    records=[]
    for trial in data['trials']:
        if not trial['valid']:continue
        checks=[]
        for ci in (0,1):
            c=contexts[ci];k=c['corner']['reference_vertex_index'];ref=c['reference']
            incoming=ref[k]-ref[k-1];incoming/=np.linalg.norm(incoming)
            outgoing=ref[k+1]-ref[k];outgoing/=np.linalg.norm(outgoing)
            sign=np.sign(c['corner']['signed_heading_change_deg']);normal=sign*np.array([-incoming[1],incoming[0]])
            rows=np.load(stage/(trial['label']+'_'+c['corner']['id']+'.npz'))['rows']
            speed=np.linalg.norm(rows[:,2:4],axis=1);h=np.unwrap(np.arctan2(rows[:,2:4]@normal,rows[:,2:4]@incoming))
            ii=np.flatnonzero((speed>5)&(abs(h)>np.deg2rad(.05)))
            first_inside=bool(len(ii) and h[ii[0]]>0)
            checks.append(dict(corner=c['corner']['id'],first_turn_inside=first_inside,
                first_heading_deg=float(np.rad2deg(h[ii[0]])) if len(ii) else None,
                maximum_reverse_heading_deg=float(np.rad2deg(np.maximum(-h,0).max())),
                exit_heading_error_deg=float(np.rad2deg(h[-1])-abs(c['corner']['signed_heading_change_deg']))))
        records.append(dict(label=trial['label'],parameters=trial['parameters'],cost=trial['cost'],
                            first_turn_inside_both=all(r['first_turn_inside'] for r in checks),checks=checks,
                            speed_deficits=[r['speed_deficit_mm'] for r in trial['records']]))
    eligible=[r for r in records if r['first_turn_inside_both']]
    result=dict(kind='post-search_mechanism_audit_against_existing_user_requirement',simulations=0,
                source_trial_count=len(data['trials']),feasible_count=len(records),inside_count=len(eligible),
                all_records=records,best_inside=min(eligible,key=lambda r:r['cost']) if eligible else None,
                goal_complete=False,full_lap_not_tested=True)
    folder=stage/'mechanism_audits';folder.mkdir(exist_ok=True)
    (folder/f'trials_{len(data["trials"]):03d}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('MECHANISM_AUDIT',result['source_trial_count'],result['feasible_count'],result['inside_count'],flush=True)
    print('BEST_INSIDE',result['best_inside'],flush=True)

if __name__=='__main__':main()
