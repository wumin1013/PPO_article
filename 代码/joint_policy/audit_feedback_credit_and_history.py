"""Saved-data audit of PPO credit and redundant previous-command sensitivity."""
import time
import shutil
from datetime import datetime, timezone
import numpy as np
import torch
from conditional_feedback_policy import ConditionalFeedbackPolicy
from train_feedback_from_optimized_rollouts import decoded
from search_complete_return import read
from study_io import BASE, register, receipt, sha, write_json


def main():
    torch.set_num_threads(1)
    source=BASE/'bounded_shape_full_feedback_ppo_20260916'
    initial=BASE/'interior_teacher_feedback_learning_20260916/epoch_50.pt'
    data_file=BASE/'interior_teacher_feedback_learning_20260916/training_data.npz'
    stage=register('feedback_credit_history_audit_20260916',[__file__,source/'receipt.json',source/'results.json',initial,data_file],
        dict(physics_ticks=0,training_updates=0,wall_s=120),
        'Audit saved complete MC credit, critic error and same-observation sensitivity to previous held commands. Previous held command is replaced before every physical step; sensitivity alone is not a cause-of-learning claim. No new rollout or sealed test.')
    started=time.perf_counter()
    episodes=read(source/'episode_results.json')['episodes'];updates=read(source/'results.json')['updates']
    groups=[];end=0;batch=[];count=0
    for record in episodes:
        with np.load(source/'episodes'/record['trace']) as z:
            rewards=z['reward'];returns=np.cumsum(rewards[::-1])[::-1]
            batch.append((returns,z['value'],np.full(len(returns),record['success'])))
            count+=len(rewards)
        if count<updates[end]['decisions']:continue
        ret,val,success=(np.concatenate([b[k] for b in batch]) for k in range(3))
        adv=ret-val
        groups.append(dict(decisions=count,success_decisions=int(success.sum()),failed_decisions=int((~success).sum()),
            explained_variance=float(1-np.var(adv)/(np.var(ret)+1e-12)),rmse=float(np.sqrt(np.mean(adv**2))),
            success_adv_mean=float(adv[success].mean()),failure_adv_mean=float(adv[~success].mean()) if (~success).any() else None,
            failure_absolute_advantage_share=float(np.abs(adv[~success]).sum()/max(np.abs(adv).sum(),1e-12)),
            success_return_std=float(ret[success].std())))
        batch=[];end+=1
    with np.load(data_file) as z:x=torch.tensor(z['observation']);y=torch.tensor(z['actions']);path=z['path']
    models=[initial,source/'policy_5283.pt',source/'policy_19716.pt',BASE/'interior_teacher_feedback_learning_20260916/epoch_200.pt']
    sensitivity=[]
    for model in models:
        p=ConditionalFeedbackPolicy(72);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        xx=x.clone().requires_grad_(True);actions=decoded(p.distribution(xx).mean)
        jac=[]
        for j in range(3):
            gradient=torch.autograd.grad(actions[:,j].sum(),xx,retain_graph=j<2)[0]
            jac.append(gradient[:,8:11].detach().numpy()/np.array([np.pi/2,100.,12.]))
        jac=np.stack(jac,axis=1)
        with torch.no_grad():
            default=x.clone();default[:,8:11]=default.new_tensor([0.,.3,.25])
            delta=(decoded(p.distribution(default).mean)-actions).detach().numpy()
        per_path=[]
        for index in sorted(set(path)):
            mask=path==index
            per_path.append(dict(path=int(index),rmse=np.sqrt(np.mean((actions.detach().numpy()[mask]-y.numpy()[mask])**2,axis=0)).tolist(),
                feed_history_gain_quantiles=np.quantile(jac[mask,1,1],[.1,.5,.9]).tolist(),
                feed_target_quantiles=np.quantile(y.numpy()[mask,1],[.1,.5,.9]).tolist(),
                feed_prediction_quantiles=np.quantile(actions.detach().numpy()[mask,1],[.1,.5,.9]).tolist()))
        sensitivity.append(dict(model=str(model.relative_to(BASE)),model_sha256=sha(model),
            diagonal_previous_action_gain_quantiles=np.quantile(np.diagonal(jac,axis1=1,axis2=2),[.1,.5,.9],axis=0).tolist(),
            reset_previous_action_delta_abs_quantiles=np.quantile(abs(delta),[.5,.9,1],axis=0).tolist(),paths=per_path))
    result=dict(credit=groups,actor_history_sensitivity=sensitivity,physics_ticks=0,
        caveat='Derivatives and reset inputs measure model sensitivity only; off-trajectory counterfactuals do not establish closed-loop benefit or training origin.')
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_saved_credit_history_audit',physics_ticks=0,training_updates=0,wall_s=time.perf_counter()-started))
    status_path=BASE/'当前执行状态.json'
    backup=BASE/'before'/('current_state_before_credit_history_'+datetime.now().strftime('%H%M%S')+'.json')
    backup.parent.mkdir(exist_ok=True);shutil.copy2(status_path,backup)
    status=read(status_path)
    for name in ['interior_teacher_feedback_learning_20260916','bounded_shape_return_audit_20260916',source.name,stage.name]:
        if not any(r['stage']==name for r in status['completed']):status['completed'].append(dict(stage=name,receipt_sha256=sha(BASE/name/'receipt.json')))
    status['updated_utc']=datetime.now(timezone.utc).isoformat()
    status['current']=dict(stage=stage.name,state='completed',goal_complete=False)
    status['next']='Use saved credit/history audit to select a bounded mechanism contrast; do not repeat PPO unchanged. Latest PPO12/12 but no95mm/s cruise,150deg width1.2mm and reversal6.7deg; not accepted.'
    status['findings'].append('Bounded full-feedback PPO19716 decisions:12/12 development,18.834s; all four initial turns inward and outward0mm, but reverse5.6-9.8deg,150deg width1.2mm, zero95mm/s cruise. Not adopted.')
    write_json(status_path,status)
    print('CREDIT',groups[0],groups[-1])
    for r in sensitivity:print('HISTORY',r['model'],r['diagonal_previous_action_gain_quantiles'],r['reset_previous_action_delta_abs_quantiles'])


if __name__=='__main__':main()
