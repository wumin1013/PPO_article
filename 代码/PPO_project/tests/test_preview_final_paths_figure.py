"""Freeze-before-geometry guard and reference metadata checks; no holdout reads."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import preview_final_paths_figure as figure


def test_missing_freeze_blocks_before_manifest_or_geometry(tmp_path,monkeypatch):
    candidate=SimpleNamespace(OUT=tmp_path)
    monkeypatch.setattr(figure.gate,'final_study',lambda name:candidate)
    sentinel=Mock(side_effect=AssertionError('No geometry or manifest read before freeze'))
    monkeypatch.setattr(figure.gate,'read',sentinel)
    monkeypatch.setattr(figure,'checked_points',sentinel)
    monkeypatch.setattr(sys,'argv',['preview_final_paths_figure.py','--study','direct-explore'])
    with pytest.raises(ValueError,match='freeze is required'):
        figure.main()
    sentinel.assert_not_called()
    assert not (tmp_path/'report').exists()


def test_nine_panel_split_excludes_exposed_development():
    rows=[dict(family=family,split=split) for family in figure.FAMILIES for split in figure.SPLITS]
    rows.append(dict(family='mixed_lengths',split='development'))
    selected=figure.select_nine_paths({'paths':rows})
    assert len(selected)==9
    assert all(row['split']!='development' for row in selected)
    with pytest.raises(ValueError,match='exactly one'):
        figure.select_nine_paths({'paths':rows[:-2]})


def test_geometry_statistics_verified_on_toy_reference(tmp_path):
    points=np.array([[0.,0.],[1.,0.],[1.,1.]])
    path=tmp_path/'toy.csv';np.savetxt(path,points,delimiter=',',header='x_mm,y_mm',comments='')
    row=dict(id='toy',source={'kind':'synthetic'},coordinate_unit='mm',plane='XY',reference_file='toy.csv',
        reference_sha256=figure.gate.sha(path),statistics={'segment_count':2,'path_length_mm':2.})
    _,actual=figure.checked_points(row,tmp_path)
    np.testing.assert_array_equal(actual,points)
    row['statistics']['path_length_mm']=3.
    with pytest.raises(ValueError,match='geometry statistics'):
        figure.checked_points(row,tmp_path)
