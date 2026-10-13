"""Gradient-flow checks for the independently reported numerical probe."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import preview_exploration_diagnostic as diagnostic


def test_float32_initialization_is_below_hard_minimum():
    row=diagnostic.std_probe(.01)
    assert row['raw_std_u']<row['clipped_std_u']
    assert row['gradient_std_wrt_bias']==0.


@pytest.mark.parametrize('scale',[.010001,.05,.1])
def test_interior_initialization_allows_gradient_flow(scale):
    row=diagnostic.std_probe(scale)
    assert row['gradient_std_wrt_bias']>0.
    assert row['distance_sigma_mm']==pytest.approx(3.8*scale)
