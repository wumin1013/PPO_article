"""Checksums and strict extraction shared by the v9 internal reproduction tools."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def write_new(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    reserved = {'CON','PRN','AUX','NUL'} | {p+str(i) for p in ('COM','LPT') for i in range(1,10)}
    reserved |= {p+n for p in ('COM','LPT') for n in ('\u00b9','\u00b2','\u00b3')}
    if (not name or '\\' in name or ':' in name or name.startswith('/')
        or any(ord(c)<32 or c in '<>"|?*' for c in name)
        or any(x in ('','.','..') for x in name.rstrip('/').split('/'))
        or any(x.endswith((' ','.')) or x.split('.')[0].upper() in reserved for x in path.parts)):
        raise ValueError('Unsafe archive path: '+name)
    return path.as_posix().rstrip('/')


def validate_names(*maps):
    seen = {}
    for mapping in maps:
        for name in mapping:
            if safe_name(name) != name:
                raise ValueError('Manifest accepts file names only: '+name)
            key = name.casefold()
            if key in seen and seen[key] != name:
                raise ValueError('Case-aliased manifest paths: '+seen[key]+' / '+name)
            seen[key] = name


def verify_files(root,mapping):
    for name,expected in mapping.items():
        if digest(Path(root)/safe_name(name)) != expected:
            raise ValueError('Changed file: '+name)


def validate_length(destination,*maps):
    if os.name == 'nt':
        longest = max((Path(destination)/name for mapping in maps for name in mapping),key=lambda p:len(str(p)))
        if len(str(longest).encode('utf-16-le'))//2 >= 260:
            raise ValueError('Use a short output path, such as D:\\PPO_r9: '+str(longest))


def extract_checked(archive,destination,expected,manifest_name):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries,seen,files = [],set(),set()
        for member in bundle.infolist():
            name = safe_name(member.filename)
            if name.casefold() in seen or stat.S_ISLNK(member.external_attr>>16):
                raise ValueError('Duplicate path or symlink: '+name)
            seen.add(name.casefold())
            target = (destination/name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive path escapes destination: '+name)
            if not member.is_dir():
                files.add(name)
            entries.append((member,name,target))
        if files != set(expected)|{manifest_name}:
            raise ValueError('Archive member set differs from its manifest.')
        for member,name,target in entries:
            if member.is_dir():
                target.mkdir(parents=True,exist_ok=True)
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            h = hashlib.sha256()
            if target.exists():
                if not target.is_file() or target.is_symlink():
                    raise ValueError('Invalid existing destination: '+name)
                with bundle.open(member) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        h.update(block)
                if digest(target) != h.hexdigest():
                    raise ValueError('Refusing different-byte overwrite: '+name)
            else:
                with bundle.open(member) as stream,target.open('xb') as output:
                    for block in iter(lambda:stream.read(1024*1024),b''):
                        h.update(block)
                        output.write(block)
            if name != manifest_name and h.hexdigest() != expected[name]:
                raise ValueError('Bad member checksum: '+name)
    verify_files(destination,expected)
