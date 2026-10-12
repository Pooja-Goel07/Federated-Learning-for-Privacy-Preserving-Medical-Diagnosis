"""Turn results/demo/summary.json into presentation charts + REPORT.md.

Thin wrapper around src.evaluation.report (shared with the Colab run). Run after
run_demo.py:
    python results/demo/make_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parents[1]))  # repo root, so `import src...` works

from src.evaluation.report import generate_report  # noqa: E402

if __name__ == "__main__":
    generate_report(DEMO_DIR)
