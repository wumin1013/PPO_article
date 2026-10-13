"""Small shared provenance and result helpers for the JMP revision."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_path(name):
    return np.loadtxt(ROOT / 'data' / (name + '.csv'), delimiter=',', skiprows=1)


def manifest():
    return json.loads((ROOT / 'data' / 'manifest.json').read_text('utf-8'))
