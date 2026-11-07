from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
SAMPLE_DIR = ROOT / "sample_data" / "run_demo"
STATE_DIR = ROOT / "state"
OUTPUT_DIR = ROOT / "output"
R_DIR = ROOT / "r"
DEFAULT_R_SCRIPT = R_DIR / "rolling_window_check_mse_comomo_api_fixed_forecast_final.R"

EXTRACTOR_SYSTEM = PROMPTS_DIR / "extractor_system.txt"
BASELINE_LOCKED = CONFIG_DIR / "baseline_effective.json"
BASELINE_CUSTOM = CONFIG_DIR / "paper_uk_male_effective_customised.json"
