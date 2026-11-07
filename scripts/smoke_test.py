#!/usr/bin/env python3
"""Offline smoke test: validate + demo artifacts (no LLM call)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_validate import build_effective_config
from src.demo_artifacts import load_sample_tables, sample_dir_ready
from src.paths import BASELINE_CUSTOM


def main() -> int:
    assert sample_dir_ready(), "sample_data/run_demo missing"
    tables = load_sample_tables()
    assert "mse_by_horizon" in tables and not tables["mse_by_horizon"].empty

    user_req = {
        "status": "ready",
        "country": "GBRTENW",
        "gender": "male",
        "ages": {"min": 50, "max": 89},
        "fit_period": {"start": 1960, "end": 1990},
        "test_period": {"start": 1991, "end": 2015},
        "models": ["LC", "RH", "APC", "CBD", "M7", "PLAT"],
        "outcomes": ["mse", "forecast_rates"],
    }
    baseline = json.loads(BASELINE_CUSTOM.read_text(encoding="utf-8"))
    effective = build_effective_config(baseline, user_req)
    assert effective["hmd"]["country"] == "GBRTENW"
    assert effective["run_mode"] == "evaluation"
    print("smoke_test OK")
    print("tables:", ", ".join(tables.keys()))
    print("effective country:", effective["hmd"]["country"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
