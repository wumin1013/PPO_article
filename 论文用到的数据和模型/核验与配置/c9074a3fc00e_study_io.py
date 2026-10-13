"""Small, immutable stage registration and atomic progress records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time
import uuid

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    atomic_replace(temporary,path)


def atomic_replace(source,destination):
    # Windows indexers/sync clients may briefly open the destination without delete sharing.
    for attempt in range(21):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt==20:
                raise
            time.sleep(.05)


def register(name, sources, budget, purpose):
    stage = BASE / name
    stage.mkdir(exist_ok=False)
    (stage / 'source').mkdir()
    bindings = {}
    for index, source in enumerate(sources):
        source = Path(source).resolve()
        target = stage / 'source' / (str(index) + '_' + source.name)
        shutil.copy2(source, target)
        bindings[source.relative_to(ROOT).as_posix()] = sha(source)
    write_json(stage / 'registration.json', dict(
        status='registered_before_execution', created_utc=datetime.now(timezone.utc).isoformat(),
        purpose=purpose, budget=budget, source_sha256=bindings))
    return stage


def receipt(stage, value):
    write_json(stage / 'receipt.json', dict(value,
        registration_sha256=sha(stage / 'registration.json'),
        completed_utc=datetime.now(timezone.utc).isoformat()))


def record_progress(name, source_receipts, next_action):
    """The historical J writer remains the only writer of unified records."""
    import subprocess
    import sys
    import time
    tracker = ROOT / '论文项目/revision_20260913_preview_efficiency_v2/ts_local_revision_v1'
    destination = tracker / (name + '.json')
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, dict(status='joint_stage_observed', goal_complete=False,
        receipts=[dict(path=Path(p).relative_to(ROOT).as_posix(), sha256=sha(p))
                  for p in source_receipts]))
    lock = tracker/'joint_progress_writer.lock'
    deadline = time.monotonic()+60.
    while True:
        try:
            handle = lock.open('x',encoding='utf-8')
            handle.close()
            break
        except FileExistsError:
            if time.monotonic()>deadline:
                raise TimeoutError('Unified progress writer busy; receipt is preserved')
            time.sleep(.2)
    try:
        result = subprocess.run([sys.executable, '-B', str(tracker/'progress.py'),
            '--evidence-receipt', str(destination), '--next-action', next_action],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    finally:
        lock.unlink()
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:] + result.stdout[-2500:])
    return destination
