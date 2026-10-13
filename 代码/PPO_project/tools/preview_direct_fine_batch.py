"""Registered pilot/formal batches for the common fine-witness study."""
from pathlib import Path

import preview_direct_batch as batch
import preview_direct_fine as candidate


if __name__=='__main__':
    batch.study=candidate.install()
    batch.SCRIPT=Path(candidate.__file__)
    batch.main()
