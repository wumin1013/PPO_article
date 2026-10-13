"""Reuse registered direct PPO batch stages with the shared-gain candidate."""
from pathlib import Path

import preview_direct_batch as batch
import preview_direct_gain as candidate


if __name__ == '__main__':
    batch.study = candidate.install()
    batch.SCRIPT = Path(candidate.__file__)
    batch.main()
