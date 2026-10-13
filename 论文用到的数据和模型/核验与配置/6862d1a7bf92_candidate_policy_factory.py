"""Explicit policy construction for the two recorded experimental residuals."""
from deep_joint_credit_policy import load_credit_policy
from arc_cruise_policy import build_policy
from coordinated_cruise_policy import CoordinatedCruisePolicy
from bend_conditioned_policy import BendConditionedPolicy


def candidate_policy(metadata,selection):
    base=build_policy(load_credit_policy(metadata),selection['source_spec'])
    if 'bend_spec' in selection:return BendConditionedPolicy(base,selection['bend_spec'])
    return CoordinatedCruisePolicy(base,selection['cruise_spec'])
