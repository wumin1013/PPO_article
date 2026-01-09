from __future__ import annotations
import math
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

class DebugCollector:
    """
    Debug trace collector for CNC Environment (formerly P7.3).
    Handles NaN/Inf detection and trajectory tracing for debugging.
    """
    def __init__(self, trace_ring_size: int = 200, dump_dir: Optional[Path] = None):
        self.trace_ring_size = max(50, min(int(trace_ring_size), 2000))
        self.trace_ring: List[Dict[str, Any]] = []
        
        if dump_dir is None:
            # Default to PPO_project/out/p7_3_nan_dumps
            # Assuming this file is in src/utils/debug.py -> parents[2] is PPO_project
            self.dump_dir = Path(__file__).resolve().parents[2] / "out" / "p7_3_nan_dumps"
        else:
            self.dump_dir = Path(dump_dir)
            
    def reset(self):
        """Clear trace ring on reset."""
        self.trace_ring = []

    def append_trace(self, step: int, pos: np.ndarray, progress: float, 
                    contour_error: float, p4_status: Dict[str, float]) -> None:
        """Append a step to the circular trace buffer."""
        try:
            entry = {
                "step": int(step),
                "pos": [float(pos[0]), float(pos[1])],
                "progress": float(progress),
                "contour_error": float(contour_error),
                "v_exec": float(p4_status.get("v_exec", float("nan"))),
                "omega_exec": float(p4_status.get("omega_exec", float("nan"))),
                "v_ratio_exec": float(p4_status.get("v_ratio_exec", float("nan"))),
                "v_ratio_cap": float(p4_status.get("v_ratio_cap", float("nan"))),
                "kappa_exec": float(p4_status.get("kappa_exec", float("nan"))),
                "dkappa_exec": float(p4_status.get("dkappa_exec", float("nan"))),
                "alpha": float(p4_status.get("alpha", float("nan"))),
            }
            self.trace_ring.append(entry)
            
            if len(self.trace_ring) > self.trace_ring_size:
                del self.trace_ring[: max(0, len(self.trace_ring) - self.trace_ring_size)]
        except Exception:
            return

    def dump_trace(self, reason: str, step: int, p4_status: Dict[str, float], extra: Optional[Dict[str, object]] = None) -> None:
        """Dump current trace ring to JSON on error."""
        try:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000.0)
            path = self.dump_dir / f"dump_{stamp}_{reason}.json"

            payload = {
                "reason": str(reason),
                "step": int(step),
                "p4_status": {k: float(v) for k, v in (p4_status or {}).items() if isinstance(v, (int, float, np.floating))},
                "trace_tail": list(self.trace_ring),
            }
            if extra:
                payload["extra"] = extra
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def assert_finite(self, name: str, value: float, step: int, p4_status: Dict[str, float]) -> None:
        """Assert value is finite, else dump and raise."""
        v = float(value)
        if math.isfinite(v):
            return
        self.dump_trace(reason=f"non_finite_{name}", step=step, p4_status=p4_status, extra={"value": str(value)})
        raise AssertionError(f"[Debug] non-finite {name}: {value}")

    def assert_finite_array(self, name: str, arr: np.ndarray, step: int, p4_status: Dict[str, float]) -> None:
        """Assert array is all finite, else dump and raise."""
        try:
            a = np.asarray(arr, dtype=float)
        except Exception:
            self.dump_trace(reason=f"non_numeric_{name}", step=step, p4_status=p4_status)
            raise AssertionError(f"[Debug] non-numeric {name}")

        finite = np.isfinite(a)
        if bool(np.all(finite)):
            return

        nan_count = int(a.size - int(np.count_nonzero(finite)))
        self.dump_trace(
            reason=f"non_finite_{name}",
            step=step,
            p4_status=p4_status,
            extra={"shape": list(a.shape), "nan_count": int(nan_count)},
        )
        raise AssertionError(f"[Debug] non-finite {name}: nan_count={nan_count}")
