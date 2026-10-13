"""Expose the eight endpoint regressions to the project's unittest discovery."""
from pathlib import Path
import sys

DIAGNOSTICS = Path(__file__).resolve().parents[2] / "论文项目/revision_20260908/endpoint_diagnostics"
sys.path.insert(0, str(DIAGNOSTICS))
from test_endpoint import EndpointTests  # noqa: F401,E402
