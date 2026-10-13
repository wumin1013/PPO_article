"""Localize current complete-route geometry changes without rerunning physics."""
import argparse
from localize_joint_shape_residual import characterize
from audit_coupled_trust_model import labels
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--tags',nargs='*');args=parser.parse_args();stage=BASE/args.stage
    history=read(stage/'state_joint/results.json')['candidates'];by={c['tag']:c for c in history}
    tags=args.tags or [read(stage/'receipt.json')['training_best']]
    rows=[]
    for tag in tags:
        c=by[tag];names=labels(c['records']);guards=c['augmented_measured_inequalities']
        worst=sorted(zip(names,guards),key=lambda p:p[1])[:12]
        row=dict(tag=tag,summary=aggregate(c['records']),failed_guards=sum(x < -1e-8 for x in guards),
                 worst_guards=[dict(channel=n,margin=v) for n,v in worst],paths=[])
        for r in c['records']:
            if r['path'] not in (14,105,204,304,305):continue
            raw=characterize(r['trace']);row['paths'].append(dict(path=r['path'],**raw))
            print('CURRENT_GEOMETRY',tag,r['path'],'apex',raw['apex_ms'],
                'pre/post_reverse',raw['preapex_reverse_deg'],raw['postapex_reverse_deg'],
                'exit_mm',-raw['outgoing_min_mm'],'negative_intervals',raw['negative_turn_intervals'],flush=True)
        rows.append(row);print('CURRENT_GUARDS',tag,row['failed_guards'],worst[:5],flush=True)
    write_json(stage/'current_geometry_audit.json',dict(records=rows,new_physics_ticks=0,new_actor_calls=0,
        source_results_sha256=sha(stage/'state_joint/results.json'),
        interpretation='Temporal localization in actual XY/V/A; it does not identify a unique causal action head.'))


if __name__=='__main__':main()
