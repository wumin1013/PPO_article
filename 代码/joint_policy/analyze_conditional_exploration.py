"""Close both registered AR exploration arms from saved evidence only."""
import gzip
import json
from pathlib import Path
import numpy as np
from study_io import BASE,register,receipt,write_json,sha,record_progress
from summarize_conditioned_training import diagnostics
from analyze_preview_exploration import coverage


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    names=[f'ar_conditional_{mode}_20260916' for mode in ('joint','fixed')]
    for name in names:
        if not (BASE/name/'receipt.json').exists():raise RuntimeError('Wait for both original completion receipts; do not restart.')
    sources=[Path(__file__),BASE/'conditional_exploration_design.md',BASE/'summarize_conditioned_training.py',
             BASE/'analyze_preview_exploration.py',BASE/'v7_checkpoint_60000_analysis_v1/results.json',
             BASE/'v7_complete_curve_audit_v1/receipt.json']
    for name in names:
        sources += [BASE/name/f for f in ('receipt.json','registration.json','training_progress.json','selection.json','storage.json','validations.json','initial_reuse.json','final_validation/summary.json')]
        sources += [BASE/name/'final_validation'/f'{i}.npz' for i in range(12)]
    stage=register('conditional_exploration_results_20260916',sources,
        dict(saved_trace_analyses=24,unique_closed_journal_rows=120000,decompressed_journal_rows=240000,model_calls=0,physics_ticks=0,training_decisions=0),
        'Evaluate the registered 0/60k selection, complete times and actual corner/sequence diagnostics; no replay, no extra selection or training.')
    original=read(BASE/'v7_checkpoint_60000_analysis_v1/results.json')['groups']
    groups=[];external=[]
    for name,mode in zip(names,('joint','fixed')):
        folder=BASE/name;rec=read(folder/'receipt.json');reg=read(folder/'registration.json')
        assert sha(folder/'registration.json')==rec['registration_sha256']
        assert sha(folder/'final.pt')==rec['final_sha256']
        assert rec['costs']['training_actor_calls']==60000 and rec['wall_s']<=reg['budget']['wall_s']
        assert reg['budget']['sampler']=='ar_conditional8'
        log=Path(read(folder/'storage.json')['decision_journal'])
        pointer=read(log.parent/'resume_pointer.json');assert pointer['training_complete'] and pointer['decision']==60000
        binding=dict(path=str(log),sha256=sha(log));external.append(binding)
        journal_coverage=coverage(log)
        innovations=[];max_error=0.
        with gzip.open(log,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);z=np.array(r['latent']);mean=np.array(r['conditional_mean']);std=np.array(r['conditional_std'])
                e=(z-mean)/std;active=3 if mode=='joint' else 2
                lp=np.sum(-.5*e[:active]**2-np.log(std[:active])-.5*np.log(2*np.pi))
                max_error=max(max_error,abs(lp-r['log_probability']));innovations.append(e[:active])
        assert len(innovations)==60000 and max_error<2e-4
        cases=[];values=read(folder/'validations.json')['validations']
        assert [v['decisions'] for v in values]==[0,60000]
        assert read(folder/'final_validation/summary.json')['results']==values[-1]['results']
        choices=[]
        for v in values:
            assert sorted(r['path'] for r in v['results'])==list(range(12))
            count=sum(r['success'] for r in v['results'])
            choices.append(dict(decisions=v['decisions'],tag=v['tag'],successes=count,
                complete_time_sum_s=sum(r['completion_time_s'] for r in v['results']) if count==12 else None))
        expected=max(choices,key=lambda r:(r['successes'],
            -r['complete_time_sum_s'] if r['complete_time_sum_s'] is not None else float('-inf')))
        for r in values[-1]['results']:
            trace=folder/'final_validation'/f"{r['path']}.npz"
            with np.load(trace) as d:
                ref=d['reference'];rows=d['rows'];diag=diagnostics(ref,rows,d['actions'])
                speed=np.linalg.norm(rows[:,2:4],axis=1)
                for corner in diag['corners']:
                    vi=corner['vertex_index'];xy=rows[:,:2]-ref[vi];local=np.linalg.norm(xy,axis=1)<10
                    corner['median_speed_within_10mm']=float(np.median(speed[local])) if local.any() else None
                    if r['path']<4:
                        mid=int(np.argmin(np.linalg.norm(xy,axis=1)))
                        inbound=(ref[vi]-ref[vi-1]);inbound/=np.linalg.norm(inbound)
                        outbound=(ref[vi+1]-ref[vi]);outbound/=np.linalg.norm(outbound)
                        normal=np.array([-inbound[1],inbound[0]])
                        prior=np.flatnonzero((xy@inbound>=-10)&(np.arange(len(xy))<=mid))
                        after=np.flatnonzero((xy@outbound>=10)&(np.arange(len(xy))>=mid))
                        corner['inbound_10mm_signed_deviation']=float(xy[prior[0]]@normal) if len(prior) else None
                        corner['inbound_10mm_speed']=float(speed[prior[0]]) if len(prior) else None
                        corner['outbound_10mm_speed']=float(speed[after[0]]) if len(after) else None
            cases.append(dict(result=r,diagnostics=diag,trace_sha256=sha(trace)))
        selected=read(folder/'selection.json')['selected'];assert selected==rec['selected']==expected
        current=values[-1]['results'];successes=sum(r['success'] for r in current)
        total=sum(r['completion_time_s'] for r in current) if successes==12 else None
        original_name='joint_independent_refinement_v7' if mode=='joint' else 'fixed3_independent_refinement_v7'
        base=next(g for g in original if g['stage']==original_name)
        endpoint_support=successes==12 and total<base['full_time_s']
        progress=read(folder/'training_progress.json')
        assert len(progress['updates'])==rec['costs']['ppo_updates']
        for k in ('actor_minibatches','critic_minibatches'):
            assert sum(u['statistics'][k] for u in progress['updates'])==rec['costs'][k]
        eps=progress['episodes'];e=np.array(innovations)
        groups.append(dict(stage=name,mode=mode,successes=successes,full_time_s=total,selected=selected,
            reference_full_time_s=base['full_time_s'],endpoint_support=endpoint_support,
            selected_improves_initial=selected['decisions']>0 and selected['successes']==12 and selected['complete_time_sum_s']<choices[0]['complete_time_sum_s'],
            cases=cases,exploration=journal_coverage,conditional_logp_max_error=max_error,
            standardized_innovation_mean=e.mean(0).tolist(),standardized_innovation_std=e.std(0).tolist(),
            training_successes=sum(r['success'] for r in eps),ended_training_episodes=len(eps),
            actor_early_stops=sum(u['statistics']['actor_early_stop'] for u in progress['updates']),
            costs=rec['costs'],wall_s=rec['wall_s']))
        assert sha(log)==binding['sha256']
    supported=all(g['endpoint_support'] for g in groups) and groups[0]['selected_improves_initial']
    result=dict(groups=groups,candidate_supported=supported,automatic_extension=False,external_sources=external,
        scope='Paired common-initialization development only. History-conditional sampler changes the policy process and its likelihood together. No independent full seeds, final long paths or held-out superiority proof.',goal_complete=False)
    write_json(stage/'results.json',result)
    lines=['# 时间相关探索：60k结果','','| 模式 | 原独立60k/s | 条件探索60k/s | 完整成功 | 所选0/60k时间/s |','|---|---:|---:|---:|---:|']
    for g in groups:
        time_label='不可排名' if g['full_time_s'] is None else f"{g['full_time_s']:.3f}"
        lines.append(f"| {g['mode']} | {g['reference_full_time_s']:.3f} | {time_label} | {g['successes']}/12 | {g['selected']['complete_time_sum_s']:.3f} |")
    lines+=['',f"预注册采用支持：{supported}。预算结束，不自动继续。",
        '','所有旧初评与原独立探索终点评价复用。失败前缀不参与完整时间排名。新24条保存轨迹只读分析；图形仍需目视检查。',
        '','训练回合成功/失败和动作持续性属于持续改变策略的描述统计，不是独立部署成功率。强v7联合19.856、固定18.464及S2八条9.493 s仍保留。']
    (stage/'结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt(stage,dict(status='completed_conditional_exploration_analysis',saved_trace_analyses=24,
        analyzed_journal_rows=120000,decompressed_journal_rows=240000,model_calls=0,physics_ticks=0,training_decisions=0,candidate_supported=supported))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        f'时间相关条件探索联合/固定两组均完成60k，完成24条新开发评价及120k保存日志核验，预注册采用支持={supported}。按真实结果与所选初值比较，不自动扩训；形状视觉审核和研究报告更新随后。13组旧训练和强v7/S2均保留，研究目标仍未完成。')
    print('\n'.join(lines))


if __name__=='__main__':main()
