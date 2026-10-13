"""Safely restore the immutable v6 base plus v8 increment to a new workspace."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def safe_name(name):
    """Reject traversal, Windows aliases and archive names with ambiguous paths."""
    path = PurePosixPath(name)
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{p}{i}' for p in ('COM', 'LPT') for i in range(1, 10)}
    reserved |= {p+n for p in ('COM', 'LPT') for n in ('\u00b9', '\u00b2', '\u00b3')}
    if (not name or '\\' in name or ':' in name or name.startswith('/')
            or any(ord(char) < 32 or char in '<>"|?*' for char in name)
            or any(x in ('', '.', '..') for x in name.rstrip('/').split('/'))
            or any(x.endswith((' ', '.')) or x.split('.')[0].upper() in reserved for x in path.parts)):
        raise ValueError('Unsafe archive name: ' + name)
    return path.as_posix().rstrip('/')


def validate_manifest_names(*manifests):
    seen = {}
    for manifest in manifests:
        for name in manifest:
            normalized = safe_name(name)
            if normalized != name:
                raise ValueError('Checksum manifests contain file names only: '+name)
            key = name.casefold()
            if key in seen and seen[key] != name:
                raise ValueError('Case-aliased manifest paths: '+seen[key]+' / '+name)
            seen[key] = name


def validate_destination_length(destination, *manifests):
    if os.name != 'nt':
        return
    longest = max((destination/name for manifest in manifests for name in manifest),
                  key=lambda path:len(str(path)), default=destination)
    if len(str(longest).encode('utf-16-le'))//2 >= 260:
        raise ValueError('Use a short Windows output directory, for example D:\\PPO_repro_v8; '
                         'a retained historical file would exceed MAX_PATH: '+str(longest))


def extract_checked(archive, destination, expected, manifest_name):
    """All collisions must be byte-identical; extraction never follows links."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = []
        seen = set()
        file_names = set()
        for member in bundle.infolist():
            name = safe_name(member.filename)
            key = name.casefold()
            if key in seen or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Duplicate or symbolic-link archive entry: ' + name)
            seen.add(key)
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive path escapes destination: ' + name)
            if not member.is_dir():
                file_names.add(name)
            entries.append((member, name, target))
        if file_names != set(expected) | {manifest_name}:
            raise ValueError('Archive members differ from its complete checksum manifest')
        for member, name, target in entries:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member_digest = hashlib.sha256()
            if target.exists():
                if not target.is_file() or target.is_symlink():
                    raise ValueError('Invalid existing extraction target: ' + name)
                with bundle.open(member) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        member_digest.update(block)
                if digest(target) != member_digest.hexdigest():
                    raise ValueError('Different bytes at existing path: ' + name)
            else:
                with bundle.open(member) as source, target.open('xb') as output:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        member_digest.update(block)
                        output.write(block)
            if name != manifest_name and member_digest.hexdigest() != expected[name]:
                raise ValueError('Extracted member checksum differs: ' + name)
        for name, expected_digest in expected.items():
            if digest(destination / safe_name(name)) != expected_digest:
                raise ValueError('Extracted file checksum differs: ' + name)


def main():
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--delta', type=Path)
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    base = args.base or directory / 'reproduction_payload.zip'
    delta = args.delta or directory / 'reproduction_delta_v8.zip'
    receipt = read(delta.with_suffix('.json'))
    if digest(base) != receipt['base_archive_sha256'] or digest(delta) != receipt['sha256']:
        raise ValueError('Base or delta archive checksum differs')
    with zipfile.ZipFile(base) as bundle:
        base_files = json.loads(bundle.read('PAYLOAD_SHA256.json').decode('utf-8'))
    with zipfile.ZipFile(delta) as bundle:
        manifest = json.loads(bundle.read('DELTA_V8_MANIFEST.json').decode('utf-8'))
    if manifest['base_archive_sha256'] != receipt['base_archive_sha256']:
        raise ValueError('Delta was not bound to this immutable base')
    validate_manifest_names(base_files, manifest['files'])
    conflicts = [name for name,value in manifest['files'].items()
                 if name in base_files and base_files[name] != value]
    if conflicts:
        raise ValueError('Increment changes an immutable base file: '+', '.join(conflicts))
    destination = args.output.resolve()
    validate_destination_length(destination, base_files, manifest['files'])
    if destination.exists():
        raise FileExistsError('Use a new output directory: ' + str(destination))
    destination.mkdir(parents=True)
    extract_checked(base, destination, base_files, 'PAYLOAD_SHA256.json')
    extract_checked(delta, destination, manifest['files'], 'DELTA_V8_MANIFEST.json')
    # A same-name increment can only contain the original base bytes.
    for name, expected in base_files.items():
        if digest(destination / safe_name(name)) != expected:
            raise ValueError('Increment modified an immutable base file: ' + name)
    extraction = dict(base_archive_sha256=digest(base), delta_archive_sha256=digest(delta),
        base_files_verified=len(base_files), delta_files_verified=len(manifest['files']),
        base_bytes_preserved=True, destination=str(destination))
    (destination / 'EXTRACTION_RECEIPT_V8.json').write_text(
        json.dumps(extraction, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if not args.extract_only:
        subprocess.run([sys.executable, '-B', 'src/regenerate_representative_v8.py'],
                       cwd=destination, check=True)
    print('Verified v8 reproduction workspace: ' + str(destination))


if __name__ == '__main__':
    main()
