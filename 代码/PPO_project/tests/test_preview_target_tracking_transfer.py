"""No training provenance substitution and no hidden model/config changes."""
import copy
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_target_tracking_transfer as transfer


def test_only_feed_metadata_added_to_the_selected_numerical_configuration():
    _, original, metadata = transfer.parent_record()
    before = copy.deepcopy(original)
    candidate = transfer.transfer_configuration(original, metadata)
    assert original == before
    assert candidate['revision']['evaluation_transfer']['additional_training_decisions'] == 0
    candidate['revision'].pop('feed_tracking'); candidate['revision'].pop('evaluation_transfer')
    assert candidate == original


def test_complete_checkpoint_bytes_and_original_training_sources_preserved(tmp_path):
    output = transfer.prepare(tmp_path/'transfer')
    parent, cfg, metadata = transfer.parent_record()
    assert (output/'checkpoint.pt').read_bytes() == (parent/'checkpoint.pt').read_bytes()
    assert (output/'parent_training_source.zip').read_bytes() == (parent/'source.zip').read_bytes()
    assert (output/'parent_training_provenance.json').read_bytes() == (parent/'provenance.json').read_bytes()
    assert not (output/'training_summary.json').exists()
    provenance = transfer.read_json(output/'provenance.json')
    assert provenance['parent_training_source_sha256'] != provenance['evaluation_source_sha256']
    assert provenance['source_sha256'] == provenance['evaluation_source_sha256']
    assert provenance['additional_training_decisions'] == 0
    assert provenance['transfer_tool_source_sha256'] != provenance['source_sha256']
    assert transfer.verify(output)['parent_model_unchanged']
    assert transfer.prepare(output) == output
    modified = bytearray((output/'checkpoint.pt').read_bytes()); modified[-1] ^= 1
    (output/'checkpoint.pt').write_bytes(modified)
    with pytest.raises(ValueError, match='Transfer artifact changed'):
        transfer.verify(output)


def test_parent_selection_must_match_the_registered_selected_model(monkeypatch):
    original = transfer.read_json
    def changed(path):
        value = original(path)
        if path == transfer.SELECTION:
            row = next(row for row in value['records'] if row['selected']['method'] == 'adaptive')
            row['selected']['checkpoint_sha256'] = 'wrong'
        return value
    monkeypatch.setattr(transfer, 'read_json', changed)
    with pytest.raises(ValueError, match='selected full-batch'):
        transfer.parent_record()


def test_failed_trace_has_no_success_time_gain_and_sets_must_match():
    before = [dict(path_id='a', success=True, elapsed_time_s=4., contour_error_max_mm=.2, failure_reasons=[])]
    after = [dict(path_id='a', success=False, elapsed_time_s=1., contour_error_max_mm=2., failure_reasons=['boundary'])]
    row = transfer.comparison_rows(before, after)[0]
    assert row['relative_time_change_pct'] is None and not row['time_comparable']
    assert row['after_failure_reasons'] == ['boundary']
    with pytest.raises(ValueError, match='complete validation path set'):
        transfer.comparison_rows(before, [])
