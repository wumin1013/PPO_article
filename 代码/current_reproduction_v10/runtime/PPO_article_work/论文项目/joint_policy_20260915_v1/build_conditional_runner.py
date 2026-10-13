"""Generate a reviewable runner from the frozen v7 loop without editing it."""
from pathlib import Path
import hashlib

BASE=Path(__file__).resolve().parent


def once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new,1)


def main():
    original=BASE/'train_optimizer_refinement_v7.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='0fd61ac7a260e2ab6e90232279bffbd4673a0340ab90161c9a3ae82a0fde4e0e'
    s=original.read_text(encoding='utf-8')
    replacements=[
        ('from learning_update_v4 import optimizers,update','from learning_update_v4 import optimizers\nfrom conditional_update import update\nfrom conditional_exploration import ConditionalExplorationPolicy'),
        ('    args=parser.parse_args()',"    args=parser.parse_args()\n    if args.sampler!='independent' or args.stage!=f'ar_conditional_{args.mode}_20260916':\n        raise ValueError('Only registered conditional-exploration candidates are allowed')\n    if args.mode=='joint' and args.initial_reuse!='joint_independent_refinement_v7':\n        raise ValueError('Reuse the original joint initial validation')"),
        ("'learning_design_v7.md','smooth_policy_candidate.py'","'learning_design_v7.md','conditional_exploration_design.md','conditional_exploration.py','conditional_update.py','conditional_exploration_validation_20260916/receipt.json','build_conditional_runner.py','smooth_policy_candidate.py'"),
        ("validation_runs=84 if args.mode=='joint' and not args.initial_reuse else 72",'validation_runs=12'),
        ("'Four-arm lower-step PPO refinement after saved-buffer diagnosis; paired exploratory study, not independent seeds or final superiority evidence.'","'History-conditional AR Gaussian exploration versus reused original v7 independent 0/60k controls; bounded paired development.'"),
        ("        policy=(SmoothExplorationPolicy(incoming['observation_dim'],mode=args.mode,noise_interval=8)\n                if args.sampler=='gsde8' else JointPolicy(incoming['observation_dim'],mode=args.mode))","        policy=ConditionalExplorationPolicy(incoming['observation_dim'],mode=args.mode)"),
        ("observation_dim=incoming['observation_dim'],mode=args.mode,sampler=args.sampler,noise_interval=8 if args.sampler=='gsde8' else 1)","observation_dim=incoming['observation_dim'],mode=args.mode,sampler='ar_conditional8',\n                        exploration_context=policy.context_state())"),
        ("        batch={key:[] for key in BATCH_KEYS}\n        episode_index=0","        BATCH_KEYS=BATCH_KEYS+['previous_observation','previous_latent','context_live']\n        batch={key:[] for key in BATCH_KEYS}\n        episode_index=0"),
        ("                    batch[key].append(item)\n                episode_return+=reward","                    batch[key].append(item)\n                for key,item in policy.last_context.items():\n                    batch[key].append(item)\n                episode_return+=reward"),
        ("reward=reward,elapsed_s=info['elapsed_s'],terminal=done,reason=info['reason'],","reward=reward,value=value,next_value=next_value,elapsed_s=info['elapsed_s'],terminal=done,reason=info['reason'],\n                    conditional_mean=policy.last_distribution['mean'].tolist(),conditional_std=policy.last_distribution['std'].tolist(),"),
        ("                    if args.sampler=='gsde8':\n                        policy.exploration_age.fill_(policy.noise_interval)\n                    episode_index+=1","                    policy.reset_context()\n                    episode_index+=1"),
        ("                    if args.sampler=='gsde8':\n                        policy.exploration_age.fill_(policy.noise_interval)\n                    costs['ppo_updates']+=1","                    costs['ppo_updates']+=1"),
        ("'v7较小actor步长配对细化一组完成360000新决策；按四组全部检查点判定，旧v6、60k探索及七值固定搜索全部保留。无稳定优势预断，仍需充分训练/多种子、长路径、匹配消融、封存测试及20–25页全PDF验收。'","'时间相关探索及正确条件似然一组已完成60k；两组按同模式原v7的0/60k对照及预注册停止条件分析，保留全部强对照和负结果，不自动扩预算。研究目标未完成。'")]
    for old,new in replacements:s=once(s,old,new)
    s=s.replace('360000','60000').replace('360001','60001').replace('18000','7200')
    s=s.replace('sampler=args.sampler',"sampler='ar_conditional8'")
    s=s.replace('physics_main_ticks=7200000','physics_main_ticks=1200000').replace('terminal_tail_ticks_upper_bound=756000000','terminal_tail_ticks_upper_bound=126000000')
    s='# Generated from frozen v7 by build_conditional_runner.py; physics and public task unchanged.\n'+s
    compile(s,'train_conditional_exploration.py','exec')
    path=BASE/'train_conditional_exploration.py'
    assert not path.exists()
    path.write_text(s,encoding='utf-8')
    print('Generated registered two-arm runner; zero training execution.')


if __name__=='__main__':main()
