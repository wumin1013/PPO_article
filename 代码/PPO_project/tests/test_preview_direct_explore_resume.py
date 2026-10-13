"""The CLI bridge accepts only registered methods and delegates unchanged."""
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import preview_direct_explore_resume as cli


@pytest.mark.parametrize('seed',[11,22,33,44,55])
def test_registered_beta_zero_formal_cli_is_accepted(seed):
    args=cli.parse_args(['train','--variant','selected_fixed_no_anchor','--seed',str(seed),'--stage','formal','--steps','131072'])
    assert (args.variant,args.seed,args.stage,args.steps)==('selected_fixed_no_anchor',seed,'formal',131072)


def test_invalid_method_still_rejected():
    with pytest.raises(SystemExit):cli.parse_args(['train','--variant','unknown','--seed','11'])


def test_training_dispatch_preserves_exact_original_arguments(monkeypatch):
    calls=[]
    study=SimpleNamespace(activate_backend=lambda:calls.append('activate'),
        train=lambda *args:calls.append(args))
    monkeypatch.setattr(cli.candidate,'install',lambda:study)
    monkeypatch.setattr(cli,'archive_launcher',lambda *args:calls.append('archive'))
    cli.main(['train','--variant','selected_fixed_no_anchor','--seed','22','--stage','formal','--steps','131072'])
    assert calls==['activate','archive',('selected_fixed_no_anchor',22,'formal',131072)]


def test_evaluation_uses_existing_api_and_split(monkeypatch):
    calls=[]
    study=SimpleNamespace(activate_backend=lambda:None,evaluate=lambda *args:calls.append(args))
    monkeypatch.setattr(cli.candidate,'install',lambda:study)
    cli.main(['evaluate','--run','example','--split','validation'])
    assert calls==[(Path('example'),'validation','nominal')]


def test_importing_bridge_preserves_protocol_and_source_hash():
    script="""import sys, tempfile, json
from pathlib import Path
sys.path.insert(0,'PPO_project/tools')
import preview_direct_explore as c
s=c.install()
archived_protocol=s.OUT/'protocol.json'
archived_bytes=archived_protocol.read_bytes()
# The working source may have a documented post-experiment encoding cleanup.
# Test import isolation against a fresh temporary protocol, preserving archives.
with tempfile.TemporaryDirectory() as directory:
    s.OUT=Path(directory)
    s.prepare()
    before=s.check_protocol()
    source_before=s.source_state()['source_sha256']
    import preview_direct_explore_resume
    assert before==s.check_protocol()
    assert source_before==s.source_state()['source_sha256']
    assert 'PPO_project/tools/preview_direct_explore_resume.py' not in s.source_state()['source_files']
    changed=dict(before,source_sha256='deliberately_invalid_fixture')
    (s.OUT/'protocol.json').write_text(json.dumps(changed),encoding='utf-8')
    try:
        s.check_protocol()
    except ValueError as error:
        assert 'Protocol/source changed' in str(error)
    else:
        raise AssertionError('The protocol guard must still reject changed source')
assert archived_protocol.read_bytes()==archived_bytes
"""
    subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[2],check=True)
