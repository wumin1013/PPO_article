"""Study selection and fail-closed held-out evaluation gate; no policies run."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import preview_two_step_eval as runner


def study_fixture(tmp_path):
    return SimpleNamespace(OUT=tmp_path,DATA=tmp_path,
        backend=SimpleNamespace(sha=lambda path:'manifest'),
        source_state=lambda:{'source_sha256':'source'},
        manifest=Mock(side_effect=AssertionError('No path access in freeze check')),
        path_points=Mock(side_effect=AssertionError('No path access in freeze check')))


@pytest.mark.parametrize('name,module_name',[('direct-gain','preview_direct_gain'),('direct-fine','preview_direct_fine'),('direct-explore','preview_direct_explore')])
def test_direct_gain_uses_explicit_install(monkeypatch,name,module_name):
    study=object()
    module=SimpleNamespace(install=Mock(return_value=study))
    importer=Mock(return_value=module)
    monkeypatch.setattr(runner.importlib,'import_module',importer)
    assert runner.load_study(name) is study
    importer.assert_called_once_with(module_name)
    module.install.assert_called_once_with()


def test_cli_missing_freeze_rejects_before_path_access(tmp_path,monkeypatch):
    study=study_fixture(tmp_path)
    monkeypatch.setattr(runner,'load_study',lambda name:study)
    monkeypatch.setattr(sys,'argv',['preview_two_step_eval.py','--study','direct-gain','--split','test'])
    with pytest.raises(ValueError,match='Freeze the final direct-gain'):
        runner.main()
    study.manifest.assert_not_called()
    study.path_points.assert_not_called()
    assert not (tmp_path/'two_step').exists()


@pytest.mark.parametrize('source,manifest',[('changed','manifest'),('source','changed')])
def test_changed_freeze_rejects(tmp_path,source,manifest):
    study=study_fixture(tmp_path)
    (tmp_path/'freeze.json').write_text(json.dumps({'source_sha256':source,'path_manifest_sha256':manifest}),encoding='utf-8')
    with pytest.raises(ValueError,match='test baselines refused'):
        runner.require_test_freeze(study,'direct-gain')
    study.path_points.assert_not_called()


def test_matching_freeze_check_does_not_evaluate(tmp_path):
    study=study_fixture(tmp_path)
    (tmp_path/'freeze.json').write_text(json.dumps({'source_sha256':'source','path_manifest_sha256':'manifest'}),encoding='utf-8')
    runner.require_test_freeze(study,'direct-gain')
    study.manifest.assert_not_called()
    study.path_points.assert_not_called()


def test_final_explore_uses_complete_registered_freeze(tmp_path,monkeypatch):
    import preview_unified_baselines as gate
    study=study_fixture(tmp_path)
    (tmp_path/'freeze.json').write_text('{}',encoding='utf-8')
    verifier=Mock(side_effect=ValueError('Complete formal model check'))
    monkeypatch.setattr(gate,'validate_final',verifier)
    with pytest.raises(ValueError,match='Complete formal model check'):
        runner.require_test_freeze(study,'direct-explore')
    verifier.assert_called_once_with(study)
    study.path_points.assert_not_called()
