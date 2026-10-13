"""Explicit metadata-aware loaders for the current14input research actors."""
def from_saved(saved,seed=922302):
    name=saved.get('policy_class')
    if name=='LocalizedJointPolicy':
        from localized_joint_correction import LocalizedJointPolicy
        policy=LocalizedJointPolicy(seed,count=saved['joint_basis_count'],base_count=saved['base_local_basis_count'])
    elif name=='LocalizedPreviewPolicy':
        from localized_preview_head import LocalizedPreviewPolicy
        policy=LocalizedPreviewPolicy(seed,count=saved['local_basis_count'],left_support=saved.get('left_turn_support',False),positive_only=saved.get('nonnegative_preview_correction',False))
    elif name=='GatedHeadPolicy':
        from gated_head_policy import GatedHeadPolicy
        policy=GatedHeadPolicy(seed,preview_only=saved.get('preview_only_correction',False))
    elif name=='GeometryPreviewPolicy':
        from geometry_preview_head import GeometryPreviewPolicy
        policy=GeometryPreviewPolicy(seed)
    elif saved['interface'][1]==2:
        from direct_two_action_control import DirectTwoPolicy
        policy=DirectTwoPolicy(seed)
    else:
        from compact_fine_policy import FineCompactPolicy
        policy=FineCompactPolicy(seed=seed,feed_gain=saved.get('feed_gain',1.))
    assert saved['interface'][0]==14
    policy.load_state_dict(saved['policy'])
    return policy
