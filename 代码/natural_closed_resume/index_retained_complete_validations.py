"""Bind all actual complete pairs of the three already retained search candidates."""
from pathlib import Path
import argparse,json,sys
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('search');ap.add_argument('fixed');ap.add_argument('candidates',nargs=3);args=ap.parse_args()
    search=Path(args.search);fixed=Path(args.fixed)
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    retained=read(search/'retained.json');records=[]
    for candidate in map(Path,args.candidates):
        selection=read(candidate/'selected_policy.json');r=next(r for r in retained if r['spec']==selection['bend_spec'])
        paths=[candidate/'selected_policy.json',candidate/'full_result.json',candidate/'half_ms/full_result.json',candidate/('paired_acceptance_'+fixed.name)/'assessment.json']
        nominal=read(paths[1]);half=read(paths[2]);assessment=read(paths[3]);assert nominal['success'] and half['success']
        records.append(dict(conditional_search_seed=r['seed'],selected_label=r['selected_label'],candidate=str(candidate),
            nominal_complete_time_s=nominal['complete_time_s'],half_ms_complete_time_s=half['complete_time_s'],
            all34_passed_at_both_resolutions=all(x['passed'] for x in assessment['corner_comparisons'].values()),
            strict_quantitative_acceptance_passed=assessment['quantitative_core_passed'],source_sha256={str(p):sha(p) for p in paths}))
    assert len(set(r['conditional_search_seed'] for r in records))==3
    records.sort(key=lambda r:r['conditional_search_seed']);dest=search/'retained_complete_validation_index.json';assert not dest.exists()
    write_json(dest,dict(all_complete_pairs_available=True,records=records,independent_full_pipeline_ppo_repeats=0,
        inherited_model_and_training_contexts_shared=True,source_sha256={str(search/'retained.json'):sha(search/'retained.json')},
        purpose='Actual complete-path validation of related retained parameter candidates; no invented independent-model variance.',method_frozen=False,goal_complete=False))
    print('RETAINED_COMPLETE_INDEX',dest,flush=True)

if __name__=='__main__':main()
