"""Read the last 60k training decisions preceding each selected joint actor."""
import gzip
import json
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json, record_progress, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    names = ('joint_independent_refinement_v7','joint_gsde_refinement_v7')
    sources = [Path(__file__), BASE/'v7_selected_lock3_v1/results.json',
        BASE/'v7_long_time_regions_v1/results.json', BASE/'saved_preview_tail_mass_v1/receipt.json']
    specifications = []
    for name in names:
        folder = BASE/name
        end = read(folder/'selection.json')['selected']['decisions']
        external = Path(read(folder/'storage.json')['rolling_state_directory'])
        filename = 'decisions_recovered_20260915_v3.jsonl.gz' if end == 359999 else 'decisions.jsonl.gz'
        specifications.append((name, end, external/filename))
        sources += [folder/'selection.json',folder/'receipt.json',folder/'storage.json']
    bindings = [dict(stage=name,path=str(path),sha256=sha(path),first_decision=end-59999,last_decision=end)
        for name,end,path in specifications]
    stage = register('v7_selected_exploration_coverage_v1',sources,
        dict(journals=2,analyzed_decisions=120000,model_calls=0,physics_ticks=0,training_decisions=0),
        'Inspect actual exploration before the selected v7 joint policies; prior tail probabilities did not measure these training windows.')
    write_json(stage/'external_sources.json',bindings)
    groups = []
    for (name,end,path),binding in zip(specifications,bindings):
        rows = []
        decompressed = 0
        with gzip.open(path,'rt',encoding='utf-8') as file:
            for line in file:
                row = json.loads(line)
                decompressed += 1
                if row['decision'] > end-60000:
                    rows.append(row)
                if row['decision'] == end:
                    break
        assert [r['decision'] for r in rows] == list(range(end-59999,end+1))
        held = np.array([r['held'] for r in rows])
        d,feed = held[:,2],held[:,1]
        masks = dict(d_ge_4p6=d>=4.6,d_ge_6=d>=6.,d_ge_8=d>=8.,
            d_ge_8_and_feed_25_to_45=(d>=8.)&(feed>=25.)&(feed<=45.))
        runs = {}
        for key,mask in masks.items():
            longest = current = 0
            for i,yes in enumerate(mask):
                if i and rows[i]['episode'] != rows[i-1]['episode']:
                    current = 0
                current = current+1 if yes else 0
                longest=max(longest,current)
            runs[key]=dict(count=int(mask.sum()),fraction=float(mask.mean()),max_consecutive_decisions=longest)
        terminal = [r for r in rows if r['terminal']]
        groups.append(dict(stage=name,first_decision=end-59999,last_decision=end,
            decompressed_rows=decompressed,analyzed_rows=len(rows),
            preview_min_max_mm=[float(d.min()),float(d.max())],
            preview_quantiles_mm=np.quantile(d,[.01,.1,.5,.9,.99]).tolist(),
            feed_quantiles_mm_s=np.quantile(feed,[.01,.1,.5,.9,.99]).tolist(),
            coverage=runs,ended_episodes=len(terminal),failed_ended_episodes=sum(r['reason']!='success' for r in terminal)))
        assert sha(path) == binding['sha256']
    result=dict(groups=groups,
        scope='Actual sampled held actions in a separate 60k window before each selected model. Different training windows and changing policies; not conditional corner coverage or probabilities, independent seeds, or a causal comparison. D8 and feed25--45 describe a previously feasible probe, not an optimal-action prescription.',
        model_calls=0,physics_ticks=0,training_decisions=0,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_v7_selected_actual_exploration',analyzed_decisions=120000,
        decompressed_rows=sum(g['decompressed_rows'] for g in groups),model_calls=0,physics_ticks=0,training_decisions=0))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json',stage/'external_sources.json'],
        '已只读所选v7联合模型之前各60k真实探索窗口，共120k记录，检查大D及速度组合覆盖，不重做旧尾概率或物理探测。结合锁D的小收益与长mixed差距判断后继，不能凭全局动作频率声称角点条件概率或因果。长路径新结果尚需接入报告；最终目标仍未完成。')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
