"""Decline an unavailable finite brake candidate without changing commands."""
from joint_runtime_v3 import JointRuntimeV3


class TerminalCandidateRuntime(JointRuntimeV3):
    def _consider_terminal(self):
        try:
            super()._consider_terminal()
        except ValueError as error:
            if str(error) != 'No feasible nonreversing vector cubic brake within registered horizon':
                raise
            # This search is a candidate query; the policy remains responsible
            # for reaching a state from which an audited stop is available.
            return
