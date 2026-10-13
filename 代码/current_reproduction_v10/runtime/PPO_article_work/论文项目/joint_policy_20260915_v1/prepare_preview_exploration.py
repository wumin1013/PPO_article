"""Create a separately bounded source from the preserved v7 training entry."""
from pathlib import Path
import ast
import difflib
from study_io import BASE, register, receipt, write_json, sha


def main():
    original=BASE/'train_optimizer_refinement_v7.py'
    target=BASE/'train_preview_exploration.py'
    if target.exists():
        raise FileExistsError(target)
    source=original.read_text(encoding='utf-8')
    source=source.replace('360000','60000').replace('360001','60001').replace('18000.','7200.').replace('wall_s=18000','wall_s=7200')
    old="policy.log_std.copy_(torch.log(torch.tensor([.03,.10,.20])))"
    assert source.count(old)==1
    source=source.replace(old,"policy.log_std.copy_(torch.log(torch.tensor([.03,.10,.40])))")
    old="for key,value in reused['policy'].items():\n                assert torch.equal(value,policy.state_dict()[key]),key"
    new="for key,value in reused['policy'].items():\n                if key=='log_std':\n                    assert torch.equal(value[:2],policy.log_std[:2]),key\n                    continue\n                assert torch.equal(value,policy.state_dict()[key]),key"
    assert source.count(old)==1
    source=source.replace(old,new)
    source=source.replace("'Four-arm lower-step PPO refinement after saved-buffer diagnosis; paired exploratory study, not independent seeds or final superiority evidence.'",
        "'One pre-registered preview-std-only comparison against reused narrow-v7 and unaffected fixed3 0/60k controls; not final superiority evidence.'")
    source=source.replace("'learning_design_v7.md','smooth_policy_candidate.py',",
        "'learning_design_v7.md','preview_exploration_design_v1.md','smooth_policy_candidate.py',")
    source=source.replace("validation_runs=84 if args.mode=='joint' and not args.initial_reuse else 72,",
        "validation_runs=12, initial_preview_std=.40, gae_lambda=.98,")
    source=source.replace("physics_main_ticks=7200000,", "physics_main_ticks=1200000,")
    source=source.replace("terminal_tail_ticks_upper_bound=756000000,", "terminal_tail_ticks_upper_bound=126000000,")
    marker="args=parser.parse_args()\n"
    assert source.count(marker)==1
    source=source.replace(marker,marker+"    if (args.stage,args.mode,args.sampler,args.initial_reuse)!=('wide_preview_independent_v1','joint','independent','joint_independent_refinement_v7'):\n        raise ValueError('Use only the registered single candidate and initial-evaluation reuse')\n")
    source=source.replace("'v7较小actor步长配对细化一组完成60000新决策；按四组全部检查点判定，旧v6、60k探索及七值固定搜索全部保留。无稳定优势预断，仍需充分训练/多种子、长路径、匹配消融、封存测试及20–25页全PDF验收。'",
        "'单组D初始探索标准差.40已按60000决策完成；仅与原v7独立窄探索和有效设置不变的固定3在0/60k配对，原初评复用。核全部真实运动和覆盖，不能以回报或更大D输出代替成功；强v7、简化负结果和S2保留。总目标仍未完成。'")
    ast.parse(source)
    assert '[.03,.10,.40]' in source and '[.03,.10,.20]' not in source
    assert 'range(1,60001)' in source and 'decision==60000' in source
    assert "validation_runs=12" in source and '360000' not in source
    stage=register('preview_exploration_preparation_v1',[Path(__file__),original,
        BASE/'preview_exploration_design_v1.md',BASE/'v7_selected_exploration_coverage_v1/receipt.json',
        BASE/'v7_long_time_regions_v1/receipt.json'],dict(training_decisions=0,model_calls=0,physics_ticks=0),
        'Prepare one independent source with a single exploration-distribution change and smaller matched budget; no training during preparation.')
    target.write_text(source,encoding='utf-8')
    diff=''.join(difflib.unified_diff(original.read_text(encoding='utf-8').splitlines(True),source.splitlines(True),
        fromfile=original.name,tofile=target.name))
    (stage/'source_diff.patch').write_text(diff,encoding='utf-8')
    write_json(stage/'source_binding.json',dict(path=str(target),sha256=sha(target),original_sha256=sha(original),
        actor_initialization_change='D log_std only: log(.20) to log(.40)',original_entry_unchanged=True))
    receipt(stage,dict(status='prepared_not_trained',model_calls=0,physics_ticks=0,training_decisions=0))
    print(diff)


if __name__=='__main__':
    main()
