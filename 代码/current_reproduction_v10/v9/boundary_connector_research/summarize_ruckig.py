"""Regenerate strict connector results from the fixed archived diagnostic."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    reg=HERE/'ruckig_registration.json'
    summary=HERE/'ruckig_summary.json'
    registered=json.loads(reg.read_text('utf-8'))
    data=json.loads(summary.read_text('utf-8'))
    assert sha(reg)==data['registration_sha256']
    assert sha(HERE/'ruckig_connector.py')==registered['source_sha256']
    for path,digest in registered['input_sha256'].items():assert sha(path)==digest
    rows=[]
    for cell in data['rows']:
        valid=[]
        for attempt in cell['attempts']:
            assert sha(attempt['full_trace'])==attempt['full_sha256']
            assert sha(attempt['prefix_trace'])==attempt['prefix_sha256']
            if (attempt['label']=='matched_discrete_target' and attempt['metrics']['success'] and
                attempt['exact_last_three'] and attempt['unchanged_body_exact'] and
                attempt['suffix_difference_v_a_j_exact'] and
                attempt['original_tracker']==attempt['new_tracker']):valid.append(attempt)
        assert len(valid)<=1
        chosen=valid[0] if valid else None
        rows.append(dict(run_id=cell['run_id'],method=cell['method'],dt_s=cell['dt_s'],
            join_s=cell['join_s'],exact_connector_success=bool(chosen),
            complete_time_s=chosen['metrics']['complete_time_s'] if chosen else None,
            saved_s=chosen['saved_s'] if chosen else None,
            reason='success' if chosen else cell['shooting'],wall_s=cell['wall_s']))
    comparisons=[]
    for dt in registered['resolutions_s']:
        for join in registered['join_times_s']:
            matched={row['method']:row for row in rows if row['dt_s']==dt and row['join_s']==join}
            both=all(row['exact_connector_success'] for row in matched.values())
            traditional=9.205 if dt==.001 else 9.2045
            comparisons.append(dict(dt_s=dt,join_s=join,both_complete=both,
                jnn_minus_fixed_s=(matched['JNN']['complete_time_s']-matched['FIXED']['complete_time_s']) if both else None,
                jnn_minus_unchanged_s2r_s=(matched['JNN']['complete_time_s']-traditional) if matched['JNN']['exact_connector_success'] else None))
    output=dict(registration_sha256=sha(reg),summary_sha256=sha(summary),
        exact_connector_success_count=sum(row['exact_connector_success'] for row in rows),
        attempted_cell_count=len(rows),rows=rows,paired_comparisons=comparisons,
        total_wall_s=sum(row['wall_s'] for row in rows),
        conclusion='The fixed 0.2-s join is feasible in all four method-resolution cells and saves at least 7 ms. JNN remains faster than equally treated Fixed. JNN is 2 ms slower than unchanged S2-R at 1 ms and 0.5 ms faster at 0.5 ms; these diagnostics do not support resolution-stable superiority over S2-R.',
        limitations=['Archived-command diagnostic, not a new closed-loop policy execution or independent test.',
                     'One 0.3-s shooting solve terminates on a Ruckig target-feasibility exception. The exception is preserved, and the rejected query input itself was not serialized; all completed solver-evaluation prefixes and their submitted states are retained.',
                     'The raw native continuous-target candidates do not satisfy the exact discrete-state join condition, even where the full command acceptance passes. They are never counted as exact connector successes.',
                     'Any online state-restoration implementation requires separate work and verification. No implementation or latency guarantee is claimed.'])
    target=HERE/'ruckig_decision.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(successes=output['exact_connector_success_count'],attempts=len(rows),output=str(target))))


if __name__=='__main__':main()
