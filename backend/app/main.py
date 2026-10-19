"""FastAPI backend for the FedMedGuard dashboard (Objective 9, FR8).

Serves the single-file React dashboard and a small results API so the frontend can read
the real experiment outputs. Run from the repo root:

    uvicorn backend.app.main:app --reload
    # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DASHBOARD = REPO / "frontend" / "dashboard.html"

app = FastAPI(title="FedMedGuard Dashboard API")


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.get("/")
def index():
    """Serve the dashboard."""
    return FileResponse(str(DASHBOARD))


@app.get("/api/results")
def results():
    """Return every available result file (core summary, defence sweep, tradeoff, audit)."""
    payload = {
        "summary": _load(RESULTS / "summary.json"),
        "sweep": _load(RESULTS / "sweep_results.json"),
        "tradeoff": _load(RESULTS / "tradeoff" / "sweep_results.json"),
        "audit": _load(RESULTS / "audit" / "audit_results.json"),
    }
    return JSONResponse({k: v for k, v in payload.items() if v is not None})


@app.get("/api/scenarios")
def scenarios():
    """Describe the scenarios the dashboard visualizes and which experiment each maps to."""
    return {
        "baseline": {"exp": "EXP-003", "scale": "core (3 nodes, 224px, ~30k)",
                     "shows": "federated vs isolated vs centralized"},
        "dp": {"exp": "EXP-004", "scale": "defence (6 nodes, 128px, ~12k)",
               "shows": "privacy-utility trade-off"},
        "attack": {"exp": "EXP-005", "scale": "defence",
                   "shows": "FedAvg collapse vs robust aggregation under model-replacement"},
        "composition": {"exp": "EXP-006", "scale": "defence",
                        "shows": "DP x robust aggregation destructive composition (G1)"},
    }


# Serve result plots (if present) at /plots for optional embedding.
_plots = RESULTS / "plots"
if _plots.exists():
    app.mount("/plots", StaticFiles(directory=str(_plots)), name="plots")
