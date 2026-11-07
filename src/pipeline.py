"""Orchestrate: extract → validate → demo artifacts (or optional local R)."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from .config_validate import build_effective_config
from .demo_artifacts import load_sample_tables, sample_run_manifest
from .llm_extract import extract_user_request
from .paths import BASELINE_CUSTOM, OUTPUT_DIR, ROOT, STATE_DIR


PipelineMode = Literal["demo", "full"]


@dataclass
class RunResult:
    run_id: str
    status: str
    mode: str
    started_at: str
    ended_at: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    user_request: Dict[str, Any] = field(default_factory=dict)
    effective_config: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_mode(requested: Optional[str] = None) -> PipelineMode:
    raw = (requested or os.environ.get("PIPELINE_MODE", "demo")).strip().lower()
    return "full" if raw == "full" else "demo"


def parse_query_as_json(query: str) -> Dict[str, Any]:
    return json.loads(query)


def run_extract_and_validate(
    query: str,
    *,
    use_llm: bool,
    baseline_path: Path,
    logs: Optional[List[str]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    log = logs if logs is not None else []
    if use_llm:
        log.append("Calling cloud LLM for NL → JSON extraction…")
        user_req = extract_user_request(query)
        log.append("LLM extraction complete.")
    else:
        log.append("Parsing query as raw JSON…")
        user_req = parse_query_as_json(query)

    if user_req.get("status") == "need_clarification":
        raise ValueError(
            "Model needs clarification: "
            + json.dumps(user_req.get("questions") or user_req.get("missing") or user_req)
        )

    baseline = _read_json(baseline_path)
    log.append(f"Merging with baseline `{baseline_path.name}`…")
    effective = build_effective_config(baseline, user_req)
    log.append("Effective config ready.")
    return user_req, effective


def _run_optional_r(
    effective_path: Path,
    logs: List[str],
) -> Path:
    """Optional full mode: shell out to the original R entry script."""
    r_script = os.environ.get("R_PIPELINE_SCRIPT", "").strip()
    if not r_script:
        raise RuntimeError(
            "PIPELINE_MODE=full but R_PIPELINE_SCRIPT is not set. "
            "Point it at rolling_window_check_mse_comomo_api_fixed_forecast_final.R "
            "from the original submission folder."
        )
    script = Path(r_script)
    if not script.is_file():
        raise FileNotFoundError(f"R script not found: {script}")

    out_root = OUTPUT_DIR / "comomo_demo"
    out_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MORTALITY_EFFECTIVE_CONFIG"] = str(effective_path)
    env["MORTALITY_OUTPUT_DIR"] = str(out_root)
    timeout = int(os.environ.get("R_PIPELINE_TIMEOUT_SEC", "28800"))

    logs.append(f"$ Rscript {script}")
    proc = subprocess.run(
        ["Rscript", str(script)],
        cwd=str(script.parent if script.is_absolute() else ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.stdout:
        for line in proc.stdout.splitlines()[-40:]:
            logs.append(line)
    if proc.stderr:
        for line in proc.stderr.splitlines()[-40:]:
            logs.append(line)
    if proc.returncode != 0:
        raise RuntimeError(f"R pipeline failed with code {proc.returncode}")

    runs = sorted(
        [p for p in out_root.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise RuntimeError("R finished but no run_* output folder was found.")
    return runs[0]


def _tables_from_run_dir(run_dir: Path, max_rows: int = 100) -> Dict[str, pd.DataFrame]:
    specs = {
        "mse_by_horizon": ("mse_by_horizon_output_comomo_api_*.csv", "mse_by_horizon_*.csv"),
        "mse_average_ranking": (
            "mse_average_ranking_output_comomo_api_*.csv",
            "mse_average_ranking_*.csv",
        ),
        "forecast_rates_preview": (
            "forecast_future_rates_*.csv",
            "forecast_rates_with_mse_output_comomo_api_*.csv",
        ),
    }
    out: Dict[str, pd.DataFrame] = {}
    for key, patterns in specs.items():
        candidates: List[Path] = []
        for pat in patterns:
            candidates.extend(run_dir.glob(pat))
        if not candidates:
            continue
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        out[key] = pd.read_csv(path).head(max_rows)
    return out


def execute_run(
    query: str,
    *,
    use_llm: bool = True,
    baseline_path: Optional[Path] = None,
    mode: Optional[str] = None,
) -> RunResult:
    run_id = uuid.uuid4().hex[:12]
    started = _now()
    logs: List[str] = []
    pipeline_mode = resolve_mode(mode)
    baseline = baseline_path or BASELINE_CUSTOM
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        run_id=run_id,
        status="running",
        mode=pipeline_mode,
        started_at=started,
        logs=logs,
    )

    try:
        user_req, effective = run_extract_and_validate(
            query,
            use_llm=use_llm,
            baseline_path=baseline,
            logs=logs,
        )
        user_path = STATE_DIR / f"{run_id}_user_request.json"
        eff_path = STATE_DIR / f"{run_id}_effective_config.json"
        _write_json(user_path, user_req)
        _write_json(eff_path, effective)
        result.user_request = user_req
        result.effective_config = effective

        if pipeline_mode == "demo":
            logs.append("Demo mode: loading bundled sample CoMoMo artifacts…")
            result.tables = load_sample_tables()
            result.files = list(result.tables.keys())
            result.metadata = sample_run_manifest()
            result.metadata["baseline"] = baseline.name
            extract_label = "LLM extract" if use_llm else "JSON parse"
            result.message = (
                f"{extract_label} + config merge succeeded. "
                "Tables below are cached sample CoMoMo outputs (demo mode)."
            )
        else:
            logs.append("Full mode: launching R CoMoMo/StMoMo pipeline…")
            run_dir = _run_optional_r(eff_path, logs)
            result.tables = _tables_from_run_dir(run_dir)
            result.files = [p.name for p in sorted(run_dir.iterdir()) if p.is_file()]
            result.metadata = {
                "mode": "full",
                "run_dir": str(run_dir),
                "baseline": baseline.name,
            }
            result.message = "Full R pipeline completed successfully."

        result.status = "completed"
    except Exception as ex:  # noqa: BLE001
        result.status = "failed"
        result.error = type(ex).__name__
        result.message = str(ex)
        logs.append(f"ERROR: {ex}")

    result.ended_at = _now()
    result.logs = logs
    return result
