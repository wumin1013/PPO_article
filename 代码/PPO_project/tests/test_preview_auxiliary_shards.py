"""Disjoint scheduling must preserve the complete legacy model roster."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from preview_auxiliary_shards import shard_keys


def test_three_shards_cover_models_once_without_overlap():
    items = [dict(baseline=method, seed=seed) for method in ('absolute', 'nnc') for seed in (11, 22, 33, 44, 55)]
    shards = [shard_keys(items, index, 3) for index in range(3)]
    flattened = [key for shard in shards for key in shard]
    assert len(flattened) == len(set(flattened)) == 10
    assert set(flattened) == {(item['baseline'], item['seed']) for item in items}
    assert list(map(len, shards)) == [4, 3, 3]


@pytest.mark.parametrize('shard,shards', [(-1, 3), (3, 3), (0, 0)])
def test_invalid_shard_is_rejected(shard, shards):
    with pytest.raises(ValueError):
        shard_keys([], shard, shards)
