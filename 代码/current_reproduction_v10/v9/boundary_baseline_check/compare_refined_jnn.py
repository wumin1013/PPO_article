"""Join explicit, completed JNN summaries with common-startup baseline results."""
from pathlib import Path
import argparse
import hashlib
import json

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learned-summary',type=Path,action='append',required=True)
    args=parser.parse_args()
    baselines=read(HERE/'summary.json')
    learned={};receipts={}
    for file in args.learned_summary:
        receipts[str(file.resolve())]=sha(file)
        for row in read(file)['rows']:
            if row['method']!='JNN':continue
            key=(row['case'],row['dt_s'])
            if key in learned:raise ValueError('Repeated JNN evidence for '+str(key))
            metrics=row.get('metrics')
            if metrics is None and row.get('record'):
                record=Path(row['record'])
                if row.get('record_sha256'):assert sha(record)==row['record_sha256']
                loaded=read(record);metrics=loaded.get('metrics',loaded)
            if metrics is None:metrics=row
            learned[key]=dict(success=metrics['success'],complete_time_s=metrics['complete_time_s'],
                             source_summary=str(file.resolve()),trace=row.get('trace'))
    rows=[]
    for row in baselines['rows']:
        competitor=learned.get((row['case'],row['dt_s']))
        both=bool(competitor and competitor['success'] and row.get('success'))
        delta=competitor['complete_time_s']-row['complete_time_s'] if both else None
        rows.append(dict(run_id=row['run_id'],case=row['case'],method=row['method'],dt_s=row['dt_s'],
            baseline_status=row['status'],baseline_success=row.get('success'),
            original_time_s=row.get('original_complete_time_s'),startup_saved_s=row.get('saved_s'),
            refined_baseline_time_s=row['complete_time_s'],
            refined_jnn_success=competitor['success'] if competitor else None,
            refined_jnn_time_s=competitor['complete_time_s'] if competitor else None,
            jnn_minus_baseline_s=delta,jnn_faster=delta<0 if delta is not None else None,
            comparison_status='both_complete' if both else 'baseline_failed' if not row.get('success') else 'jnn_unavailable_or_failed'))
    output=dict(baseline_summary_sha256=sha(HERE/'summary.json'),learned_summary_sha256=receipts,
        rows=rows,total_cells=len(rows),both_complete_count=sum(r['comparison_status']=='both_complete' for r in rows),
        jnn_faster_count=sum(r['jnn_faster'] is True for r in rows),
        scope='Pairing on the same complete path and numerical resolution. Failed baselines have null time, never treated as infinite-speedup comparisons. No averaging over successful subsets or treating resolutions as training replicates.')
    target=HERE/'jnn_comparison.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(target),both_complete=output['both_complete_count'],jnn_faster=output['jnn_faster_count'])))


if __name__=='__main__':main()
