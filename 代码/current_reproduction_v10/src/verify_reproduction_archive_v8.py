"""Exercise restoration integrity checks with small retained local fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile
from common import ROOT, utc, write_json
from reproduce_representative_v8 import extract_checked, safe_name, validate_manifest_names


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def archive(path, entries):
    with zipfile.ZipFile(path, 'x') as bundle:
        for name, value, mode in entries:
            info = zipfile.ZipInfo(name)
            if mode is not None:
                info.create_system = 3
                info.external_attr = mode << 16
            bundle.writestr(info, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = []

    def reject(name, function):
        try:
            function()
        except ValueError as error:
            cases.append(dict(case=name, passed=True, expected_rejection=str(error)))
        else:
            raise AssertionError('Malformed archive accepted: '+name)

    for index, name in enumerate(('../outside.txt', '/absolute.txt', 'D:/drive.txt',
                                 'a\\b', 'a//b', 'a/./b', 'a/../b', 'aux.txt',
                                 'a/NUL.bin', 'trailing.', 'space ', 'COM\u00b9.txt',
                                 'wild*.txt', 'a\x00b', 'a\nb')):
        reject('unsafe_name_'+str(index), lambda name=name:safe_name(name))
    reject('cross_manifest_case_alias', lambda:validate_manifest_names({'A.txt':'x'}, {'a.txt':'x'}))

    manifest_name = 'MANIFEST.json'
    content = b'preserved original evidence\n'
    expected = {'folder/data.txt':sha_bytes(content)}
    manifest = json.dumps(expected).encode('utf-8')
    good = output/'valid.zip'
    archive(good, [('folder/data.txt',content,None), (manifest_name,manifest,None)])
    destination = output/'restored'
    extract_checked(good, destination, expected, manifest_name)
    extract_checked(good, destination, expected, manifest_name)
    assert (destination/'folder/data.txt').read_bytes() == content
    cases.append(dict(case='valid_restore_and_identical_collision', passed=True))

    changed = b'unapproved changed evidence\n'
    conflict = output/'changed.zip'
    archive(conflict, [('folder/data.txt',changed,None), (manifest_name,manifest,None)])
    reject('different_existing_bytes', lambda:extract_checked(conflict,destination,
        {'folder/data.txt':sha_bytes(changed)},manifest_name))
    assert (destination/'folder/data.txt').read_bytes() == content
    reject('checksum_mismatch', lambda:extract_checked(good,output/'checksum',
        {'folder/data.txt':'0'*64},manifest_name))
    reject('missing_manifest_member', lambda:extract_checked(good,output/'missing',
        {**expected,'missing.txt':'0'*64},manifest_name))

    malformed = [
        ('traversal', [('folder/data.txt',content,None), ('../outside.txt',b'x',None)]),
        ('case_alias', [('folder/data.txt',content,None), ('folder/DATA.txt',content,None)]),
        ('symlink', [('folder/data.txt',b'../../outside.txt',stat.S_IFLNK|0o777)]),
    ]
    for name, entries in malformed:
        path=output/(name+'.zip')
        archive(path, entries+[(manifest_name,manifest,None)])
        reject(name, lambda path=path,name=name:extract_checked(path,output/name,expected,manifest_name))
    assert not (output/'outside.txt').exists()
    receipt=dict(created_utc=utc(), status='passed', tests=len(cases), cases=cases,
        original_evidence_unchanged=True, fixtures_retained=True,
        fixture_directory=output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else str(output))
    write_json(output/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],tests=len(cases),receipt=str(output/'receipt.json')),indent=2))


if __name__=='__main__':
    main()
