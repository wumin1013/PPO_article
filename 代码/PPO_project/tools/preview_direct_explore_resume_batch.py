"""Batch adapter using the separately archived CLI bridge; runtime stays frozen."""
from pathlib import Path

import preview_direct_batch as batch
import preview_direct_explore as candidate
import preview_direct_explore_resume as cli


if __name__=='__main__':
    batch.study=candidate.install()
    batch.SCRIPT=Path(cli.__file__)
    batch.main()
