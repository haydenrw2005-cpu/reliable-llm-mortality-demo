"""Load bundled sample CoMoMo outputs for free public demo mode."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .paths import ROOT, SAMPLE_DIR

TABLE_FILES = {
    "mse_by_horizon": "mse_by_horizon.csv",
    "mse_average_ranking": "mse_average_ranking.csv",
    "mse_summary": "mse_summary.csv",
    "weights_table": "weights_table.csv",
    "forecast_rates_preview": "forecast_rates_preview.csv",
}


def sample_dir_ready() -> bool:
    return SAMPLE_DIR.is_dir() and (SAMPLE_DIR / "mse_by_horizon.csv").is_file()


def list_sample_files() -> List[str]:
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(p.name for p in SAMPLE_DIR.iterdir() if p.is_file())


def load_sample_tables(max_rows: int = 100) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in TABLE_FILES.items():
        path = SAMPLE_DIR / filename
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        tables[key] = df.head(max_rows)
    return tables


def sample_run_manifest() -> Dict[str, object]:
    return {
        "mode": "demo",
        "source": str(SAMPLE_DIR.relative_to(ROOT)),
        "note": (
            "Demo mode returns cached CoMoMo sample outputs so the public site "
            "stays fast. LLM extraction + config merge still run live."
        ),
        "files": list_sample_files(),
    }
