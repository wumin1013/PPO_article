"""Check unchanged plotted data and reject missing or changed plotting dependencies."""
import contextlib
import copy
import io
import numpy as np
import main_evidence_v5 as evidence
import check_manuscript_binding_v5 as binding
from common import ROOT, sha, utc, write_json


def main():
    before = ROOT / 'archive/main_model_preview_before_style_and_dependency/main_model_v5'
    after = ROOT / 'shared_manuscript/main_model_v5'
    arrays = []
    with np.load(before/'figure8_source.npz') as old, np.load(after/'figure8_source.npz') as new:
        assert set(old.files) == set(new.files)
        for name in old.files:
            assert np.array_equal(old[name], new[name]), name
            arrays.append(name)
    equal_files = ['figure8_local.csv', 'abstract_numbers.tex', 'highlights_numbers.txt']
    assert all(sha(before/name) == sha(after/name) for name in equal_files)
    old_metrics, new_metrics = [evidence.read(p/'metrics.json') for p in (before,after)]
    assert old_metrics['runs'] == new_metrics['runs']
    report = []
    def reject(name, call):
        try:
            call()
        except ValueError as exc:
            report.append(dict(name=name,rejected=True,reason=str(exc)))
        else:
            raise AssertionError('Invalid plotting dependency accepted: '+name)
    manifest = evidence.read(evidence.MANIFEST)
    helper = evidence.FIGURE_DEPENDENCIES['local_frame']
    for name, mutation in [
        ('missing_dependency_set', lambda m: m.pop('figure_dependencies')),
        ('unbound_helper', lambda m: m['sources'].pop(helper)),
        ('wrong_helper_pointer', lambda m: m['figure_dependencies'].update(local_frame='unrelated.py')),
    ]:
        altered=copy.deepcopy(manifest)
        mutation(altered)
        reject(name,lambda: evidence.verify(altered,verify_files=False))
    original_sha=evidence.sha
    try:
        evidence.sha=lambda path: '0'*64 if path == ROOT/helper else original_sha(path)
        reject('changed_helper_file',lambda: evidence.verify())
    finally:
        evidence.sha=original_sha
    receipt=evidence.read(after/'generation_receipt.json')
    original_read=binding.read
    for name, field in [('missing_receipt_dependency',None),('wrong_receipt_dependency',{helper:'0'*64})]:
        altered=copy.deepcopy(receipt)
        if field is None:altered.pop('generator_dependency_sha256')
        else:altered['generator_dependency_sha256']=field
        try:
            binding.read=lambda path: altered if path == after/'generation_receipt.json' else original_read(path)
            reject(name,lambda: binding.validate('preview'))
        finally:
            binding.read=original_read
    with contextlib.redirect_stdout(io.StringIO()):
        binding.validate('preview')
    write_json(ROOT/'audit/main_presentation_update_v5.json',dict(created_utc=utc(),
        unchanged_array_count=len(arrays),unchanged_arrays=arrays,
        byte_identical_data_and_numeric_fragments=equal_files,
        all_eight_run_metrics_identical=True,dependency_negative_checks=report,
        helper_sha256=sha(ROOT/helper),source_sha256={name:sha(ROOT/name) for name in (
            'src/main_evidence_v5.py','src/check_manuscript_binding_v5.py',
            'manuscript/generate_main_bound_v5.py','src/audit_main_presentation_v5.py')},
        scope='Presentation wording and explicit helper binding; original model, executor, cases and plotted samples unchanged'))
    print(f'{len(arrays)} arrays unchanged; all six dependency substitutions rejected.')


if __name__ == '__main__':
    main()
