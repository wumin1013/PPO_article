"""Reset proposal rate per fresh batch; retain transactional KL backtracking."""
from learning_update_guided_mc import optimizers,update as guided_update

PROPOSAL_RATE=3e-4


def update(policy,optimizer,batch,**kwargs):
    previous=[group['lr'] for group in optimizer['actor'].param_groups]
    for group in optimizer['actor'].param_groups:
        group['lr']=PROPOSAL_RATE
    result=guided_update(policy,optimizer,batch,**kwargs)
    result.update(actor_proposal_learning_rate=PROPOSAL_RATE,
        previous_batch_final_learning_rates=previous,
        actor_final_learning_rates=[group['lr'] for group in optimizer['actor'].param_groups],
        proposal_rate_reset_scope='Once per fresh PPO batch; within-batch backtracking and full KL<=0.03 remain unchanged')
    return result
