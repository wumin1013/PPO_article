"""Offline export provenance and certificate-parameter guards."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import machine_data
from src.utils import revision_metrics


def fixture_export(tmp_path):
    points=np.array([[0.,0.],[.1,0.],[.2,0.]])
    path=tmp_path/'command.csv'
    machine_data.export_commands(path,points,[0.,.1,.2],provenance={'data_kind':'simulation'})
    return path,points


def test_export_hash_matches_and_changed_csv_is_rejected(tmp_path):
    path,points=fixture_export(tmp_path)
    result=machine_data.analyze_measurement(path,points,tolerance_mm=1.25)
    assert result['source_sha256']==result['metadata']['csv_sha256']
    with path.open('a',encoding='utf-8') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='csv_sha256'):
        machine_data.analyze_measurement(path,points,tolerance_mm=1.25)


@pytest.mark.parametrize('spacing',[0.,-.01,float('nan'),float('inf'),2.5])
def test_invalid_spacing_rejected_before_input_read(tmp_path,monkeypatch,spacing):
    output=tmp_path/'never_created.json'
    monkeypatch.setattr(sys,'argv',['machine_data.py','analyze',str(tmp_path/'missing.csv'),
        '--reference',str(tmp_path/'missing_reference.csv'),'--one-sided-tolerance-mm','1.25',
        '--witness-spacing-mm',str(spacing),'--output',str(output)])
    with pytest.raises(ValueError,match='Witness spacing'):
        machine_data.main()
    assert not output.exists()


@pytest.mark.parametrize('tolerance',[0.,-1.,float('nan'),float('inf')])
def test_invalid_tolerance_rejected_as_configuration_error(tolerance):
    with pytest.raises(ValueError,match='tolerance'):
        machine_data._validate_analysis_parameters(tolerance,.01)


def test_cli_fine_certificate_applies_and_restores_original_projection(tmp_path,monkeypatch):
    path,points=fixture_export(tmp_path)
    reference=tmp_path/'reference.csv'
    np.savetxt(reference,points,delimiter=',',header='x_mm,y_mm',comments='')
    output=tmp_path/'analysis.json'
    original=revision_metrics.ordered_projection
    monkeypatch.setattr(sys,'argv',['machine_data.py','analyze',str(path),'--reference',str(reference),
        '--one-sided-tolerance-mm','1.25','--witness-spacing-mm','.01','--output',str(output)])
    machine_data.main()
    result=json.loads(output.read_text(encoding='utf-8'))
    assert result['requested_witness_spacing_mm']==.01
    assert result['command']['witness_spacing_mm']==.01
    assert result['command']['witness_match_radius_mm']==1.245
    assert revision_metrics.ordered_projection is original
