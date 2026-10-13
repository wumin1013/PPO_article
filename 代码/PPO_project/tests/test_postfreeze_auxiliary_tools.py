"""Small mocked timing/freeze checks and read-only historical registration audit."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'PPO_project/tools'))
import preview_inference_benchmark as benchmark


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def benchmark_fixture(tmp_path,monkeypatch,corrupt=False):
    methods=['selected_fixed','direct','selected_fixed_no_anchor','direct_no_anchor']
    seeds=[11,22,33,44,55]
    protocol={'formal_variants':methods,'formal_seeds':seeds}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    (tmp_path/'selection.json').write_text('{}',encoding='utf-8')
    (tmp_path/'manifest.json').write_text('{}',encoding='utf-8')
    models={};configs={}
    for method in methods:
        for seed in seeds:
            key=f'formal/{method}_seed{seed}';path=tmp_path/key;path.mkdir(parents=True)
            (path/'checkpoint.pt').write_bytes(key.encode())
            (path/'config.yaml').write_text(yaml.safe_dump({'seed':seed,'revision':{'variant':method,'training_path_ids':['training_fixture']}}),encoding='utf-8')
            models[key]=sha(path/'checkpoint.pt');configs[key]=sha(path/'config.yaml')
    frozen={'models':models,'configs':configs,'source_sha256':'source',
        'protocol_sha256':sha(tmp_path/'protocol.json'),'selection_sha256':sha(tmp_path/'selection.json'),
        'path_manifest_sha256':sha(tmp_path/'manifest.json')}
    (tmp_path/'freeze.json').write_text(json.dumps(frozen),encoding='utf-8')
    if corrupt:(tmp_path/sorted(models)[-1]/'checkpoint.pt').write_bytes(b'changed')
    env=SimpleNamespace(action_space_dim=3,step=lambda action:(np.zeros(20,dtype=np.float32),0.,False,{}))
    class Actor:
        def load_state_dict(self,value):pass
        def eval(self):pass
        def __call__(self,state):return torch.zeros((1,3)),torch.ones((1,3))
    backend=SimpleNamespace(sha=sha,write_json=lambda path,value:Path(path).write_text(json.dumps(value),encoding='utf-8'))
    study=SimpleNamespace(OUT=tmp_path,DATA=tmp_path,backend=backend,check_protocol=lambda:None,
        source_state=lambda:{'source_sha256':'source'},make_env=Mock(return_value=(env,np.zeros(20,dtype=np.float32))),
        make_agent=lambda cfg,env:SimpleNamespace(actor=Actor()))
    monkeypatch.setattr(benchmark.importlib,'import_module',lambda name:SimpleNamespace(install=lambda:study))
    monkeypatch.setattr(benchmark.torch,'load',lambda *args,**kwargs:{'actor':{}})
    monkeypatch.setattr(benchmark.torch,'set_num_threads',lambda value:None)
    ticks=itertools.count(step=1000)
    monkeypatch.setattr(benchmark.time,'perf_counter_ns',lambda:next(ticks))
    monkeypatch.setattr(sys,'argv',['preview_inference_benchmark.py','--repeats','100'])
    return study


def test_benchmark_covers_twenty_models_and_correct_microseconds(tmp_path,monkeypatch,capsys):
    benchmark_fixture(tmp_path,monkeypatch)
    benchmark.main()
    result=json.loads((tmp_path/'inference_benchmark/summary.json').read_text(encoding='utf-8'))
    assert len(result['rows'])==20
    assert len({(row['method'],row['seed']) for row in result['rows']})==20
    assert all(row['calls']==100 and row['median_us']==1. and row['p95_us']==1. for row in result['rows'])
    assert result['state_count']==256
    assert 'within-model latency samples' in result['statistical_unit']
    assert result['states_sha256']==sha(tmp_path/'inference_benchmark/shared_training_states.npy')
    assert result['raw_sha256']==sha(tmp_path/'inference_benchmark/latency_samples.npz')


def test_benchmark_checks_last_model_before_any_state_generation(tmp_path,monkeypatch):
    study=benchmark_fixture(tmp_path,monkeypatch,corrupt=True)
    with pytest.raises(ValueError,match='Frozen model or configuration changed'):
        benchmark.main()
    study.make_env.assert_not_called()
    assert not (tmp_path/'inference_benchmark').exists()


def test_historical_registered_points_rebuild_exactly_without_test_access():
    import preview_research_v2 as v2
    directory=ROOT/'论文项目/revision_20260908/preview_direct_explore'
    record=json.loads((directory/'regression_reference_provenance.json').read_text(encoding='utf-8'))
    source=Path(record['historical_config'])
    assert record['historical_config_sha256']==sha(source)
    assert record['protocol_sha256']==sha(directory/'protocol.json')
    specs=yaml.safe_load(source.read_text(encoding='utf-8'))['training']['path_curriculum']['paths']
    assert len(specs)==len(record['references'])==3
    records={row['path_id']:row for row in record['references']}
    for spec in specs:
        stored=records[spec['name']];path=directory/stored['file']
        assert stored['sha256']==sha(path)
        points=np.asarray(v2.backend._build_path(spec),dtype=float)
        np.testing.assert_array_equal(points,np.loadtxt(path,delimiter=',',skiprows=1))
        assert stored['samples']==len(points)
        assert np.isclose(stored['length_mm'],np.linalg.norm(np.diff(points,axis=0),axis=1).sum(),rtol=0,atol=1e-10)
