"""Decompose actual initial PPO actor gradients on two immutable saved batches."""
import json
from pathlib import Path
import numpy as np
import torch
from joint_policy import JointPolicy, advantages
from resume_codec_v5 import load_resume
from study_io import BASE, register, receipt, write_json, record_progress, sha


def cosine(a, b):
    denominator = torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    return float(torch.dot(a,b)/denominator) if denominator > 0 else None


def main():
    names = ('joint_independent_exploration_learning_v1','fixed3_independent_exploration_learning_v1')
    bindings = json.loads((BASE/'saved_action_credit_v1/external_sources.json').read_text(encoding='utf-8'))
    stage = register('saved_actor_gradient_decomposition_v1',
        [Path(__file__),BASE/'joint_policy.py',BASE/'resume_codec_v5.py',BASE/'learning_update_v4.py',
         BASE/'saved_action_credit_v1/external_sources.json',BASE/'credit_outcome_decomposition_v1/results.json',
         BASE/'credit_horizon_design_v1.md'],
        dict(buffer_loads=2,model_constructions=2,actor_batch_forwards=2,critic_forwards=0,
             autograd_calls_upper=12,optimization_steps=0,physics_ticks=0),
        'Differentiate the existing independent-Gaussian PPO loss at saved parameters; inspect action and outcome components, without changing policy or task.')
    records = []
    calls = 0
    torch.set_num_threads(1)
    for name in names:
        binding = next(b for b in bindings if b['stage']==name)
        source = Path(binding['path'])
        assert sha(source)==binding['sha256']
        state = load_resume(source)
        assert state['decision']==59999 and state['sampler']=='independent'
        batch = state['batch']
        n = len(batch['reward'])
        policy = JointPolicy(state['observation_dim'],mode=state['mode'])
        policy.load_state_dict(state['policy'])
        parameters = list(policy.actor.parameters())+[policy.log_std]
        shared_size = sum(p.numel() for p in list(policy.actor.parameters())[:4])
        obs = torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
        latent = torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
        distribution = policy.distribution(obs)
        per_head_log = distribution.log_prob(latent)*policy.active_mask
        entropy = distribution.entropy()*policy.active_mask
        logp = per_head_log.sum(-1)
        old = torch.as_tensor(batch['log_probability'],dtype=torch.float32)
        ratio = torch.exp(logp-old)
        error = float(torch.max(torch.abs(logp-old)).detach())
        assert error < 1e-3 and bool(((ratio>.8)&(ratio<1.2)).all())
        a,_ = advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'])
        adv = torch.as_tensor(a)
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        surrogate = torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
        losses = -surrogate-.001*entropy.sum(-1)
        def gradient(loss):
            nonlocal calls
            calls += 1
            return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,parameters,retain_graph=True)]).detach()
        total = gradient(losses.mean())
        heads = {}
        for i,label in enumerate(('steering','feedrate','preview')):
            if policy.active_mask[i]:
                heads[label] = gradient(-(ratio.detach()*adv*per_head_log[:,i]).mean()-.001*entropy[:,i].mean())
        head_error = float(torch.max(torch.abs(sum(heads.values())-total)))
        assert head_error < 2e-5
        labels = np.full(n,'censored',dtype='U12')
        batch_start = state['decision']-n+1
        end = 0
        for episode in state['episodes']:
            start=end+1
            end += episode['decisions']
            first,last = max(start,batch_start),min(end,state['decision'])
            if first<=last:
                labels[first-batch_start:last-batch_start+1] = 'success' if episode['success'] else 'failure'
        parts = {}
        for label in ('success','failure','censored'):
            mask = torch.as_tensor(labels==label,dtype=torch.float32)
            if bool(mask.any()):
                parts[label] = gradient((losses*mask).mean())
        partition_error = float(torch.max(torch.abs(sum(parts.values())-total)))
        assert partition_error < 2e-5
        def describe(g):
            return dict(norm=float(torch.linalg.vector_norm(g)),
                shared_norm=float(torch.linalg.vector_norm(g[:shared_size])),
                cosine_with_total=cosine(g,total),
                shared_cosine_with_total=cosine(g[:shared_size],total[:shared_size]))
        pairs = []
        for i,left in enumerate(heads):
            for right in list(heads)[i+1:]:
                pairs.append(dict(left=left,right=right,all_parameters_cosine=cosine(heads[left],heads[right]),
                    shared_parameters_cosine=cosine(heads[left][:shared_size],heads[right][:shared_size])))
        d_rest = cosine(heads['preview'][:shared_size],(heads['steering']+heads['feedrate'])[:shared_size]) if 'preview' in heads else None
        records.append(dict(stage=name,source_sha256=binding['sha256'],rows=n,saved_logp_max_error=error,
            parameter_count=len(total),shared_parameter_count=shared_size,total=describe(total),
            head_gradients={k:describe(v) for k,v in heads.items()},head_cosines=pairs,
            preview_vs_other_shared_cosine=d_rest,
            outcome_gradients={k:dict(rows=int(np.sum(labels==k)),**describe(v)) for k,v in parts.items()},
            additive_checks_max_abs=dict(heads=head_error,outcomes=partition_error)))
        assert sha(source)==binding['sha256']
    result = dict(groups=records,actual_autograd_calls=calls,
        scope='Full-batch gradients of the original PPO actor loss at saved behavior parameters, including entropy and existing advantage normalization. Initial ratios are unclipped. Gradients are before gradient-norm clipping, Adam and subsequent shuffled minibatch updates. Outcome terms keep the full-batch denominator and normalization. Negative cosine indicates local interference only; nonorthogonal norms are not attribution percentages. Historical batches do not prove current-run causation or performance.',
        models_modified=False,optimization_steps=0,physics_ticks=0,training_decisions=0,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_saved_actor_gradient_decomposition',buffer_loads=2,
        model_constructions=2,actor_batch_forwards=2,critic_forwards=0,autograd_calls=calls,
        optimization_steps=0,physics_ticks=0,training_decisions=0))
    record_progress(stage.name,[stage/'receipt.json',stage/'results.json'],
        '两份历史独立高斯保存批的真实PPO actor初始梯度分解完成，分动作与原结局核对可加性。此为clip-grad/Adam前局部梯度，不能用范数当损失比例或冒称当前训练因果；λ=1两组继续既定60k预算。16页报告和18页英文已验收，最终优势与20–25页全文仍未完成。')
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
