"""Repeat the registered pilot after unpacking the loader's metadata tuple.

The first harness invocation failed before submitting any actions. Those
records are preserved separately. Executor and weights are unchanged here.
"""
import replay_terminal_v4d as replay


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    original_loader = replay.load_strong_policy
    replay.load_strong_policy = lambda path: original_loader(path)[0]
    replay.OUT = replay.ROOT / 'results/v4d_verified_terminal_preview/pilot_loader_retry'
    replay.SOURCES.append('replay_terminal_pilot_retry_v4d.py')
    for item in replay.register()['cases']:
        if item['historical_split'] in ('train', 'validation'):
            replay.run('pilot', item['id'])
    replay.run('pilot', 'dense')
