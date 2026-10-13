"""One-change follow-up: direct PPO with shared 15/s heading guidance.

The preceding direct-distance pilot remains immutable. This version changes
only the guidance gain for every fixed and learned configuration. Distances,
policy initialization, reward, optimization and evaluation are unchanged.
"""
from pathlib import Path
import hashlib
import json

import preview_direct_research as study

OUT = study.ROOT / '论文项目/revision_20260908/preview_direct_gain'
BASE_CONFIG = study.config_for
BASE_SOURCE = study.source_state
BASE_PROTOCOL = study.protocol_definition


def config_for(variant, seed):
    cfg = BASE_CONFIG(variant, seed)
    cfg['reward_weights']['control_authority'].update(
        tangent_blend=.015, blend_schedule='constant', warmup_episodes=0, warmup_blend=.015)
    cfg['revision'].update(study='direct-distance-motion-anchor-gain15-v1',
        shared_guidance_gain_per_s=15., shared_guidance_blend=.015)
    return cfg


def source_state():
    state = BASE_SOURCE()
    state['source_files'][Path(__file__).relative_to(study.ROOT).as_posix()] = study.backend.sha(__file__)
    state['source_sha256'] = hashlib.sha256(json.dumps(state['source_files'], sort_keys=True).encode()).hexdigest()
    return state


def protocol_definition():
    value = BASE_PROTOCOL()
    value.update(version='preview-direct-shared-gain15-v1',
        predecessor='preview_direct: six completed pilots, all retained; no formal or holdout data',
        candidate_change='Only shared heading guidance: tangent_blend 0.08 to 0.015 at dt=0.001 s',
        diagnostic_basis=['preview_v2/guidance_gain_probe/protocol.json',
                          'preview_v2/guidance_gain_probe_extension_D3/protocol.json'],
        shared_guidance=dict(tangent_blend=.015, gain_per_s=15., blend_schedule='constant',
                             warmup_episodes=0, attribution='Shared fixed and direct control change, not preview-only benefit'))
    return value


def install():
    """Configure one CLI process; importing this module has no side effects."""
    study.OUT = OUT
    study.config_for = config_for
    study.source_state = source_state
    study.protocol_definition = protocol_definition
    return study


if __name__ == '__main__':
    install().main()
