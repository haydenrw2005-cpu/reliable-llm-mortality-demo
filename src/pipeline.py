"""Orchestrate: extract → validate → R (or demo) → sanity-check → plain-English answer."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

import pandas as pd

from .answer_summary import build_plain_english_answer, interpretation_snapshot
from .config_validate import build_effective_config
from .demo_artifacts import load_sample_tables, sample_run_manifest
from .llm_extract import extract_user_request
from .paths import BASELINE_CUSTOM, DEFAULT_R_SCRIPT, OUTPUT_DIR, R_DIR, ROOT, STATE_DIR
from .request_validate import validate_user_request
from .result_check import run_result_checks

PipelineMode = Literal["demo", "full"]
LogFn = Callable[[str], None]

_RUN_LOCK = threading.Lock()
_ACTIVE_RUN_ID: Optional[str] = None
_RUNS: Dict[str, "RunResult"] = {}


@dataclass
class RunResult:
    run_id: str
    status: str
    mode: str
    stage: str = "queued"
    started_at: str = ""
    ended_at: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    user_request: Dict[str, Any] = field(default_factory=dict)
    effective_config: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    plain_english_answer: str = ""
    interpretation: Dict[str, str] = field(default_factory=dict)
    sanity_check: Dict[str, Any] = field(default_factory=dict)
    clarifying_questions: List[str] = field(default_factory=list)


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


def _append_log(result: RunResult, line: str) -> None:
    clean = line.rstrip()
    if not clean:
        return
    # Redact HMD secrets if present in env.
    for key in ("HMD_PASSWORD", "HMD_PASS", "HMD_USERNAME", "HMD_USER"):
        val = os.environ.get(key, "")
        if val:
            clean = clean.replace(val, "***")
    result.logs.append(clean)
    if len(result.logs) > 500:
        result.logs[:] = result.logs[-500:]


def resolve_mode(requested: Optional[str] = None) -> PipelineMode:
    raw = (requested or os.environ.get("PIPELINE_MODE", "full")).strip().lower()
    return "demo" if raw == "demo" else "full"


def parse_query_as_json(query: str) -> Dict[str, Any]:
    return json.loads(query)


def get_run(run_id: str) -> Optional[RunResult]:
    return _RUNS.get(run_id)


def active_run_id() -> Optional[str]:
    return _ACTIVE_RUN_ID


def run_extract_and_validate(
    query: str,
    *,
    use_llm: bool,
    baseline_path: Path,
    result: RunResult,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    result.stage = "extract"
    if use_llm:
        _append_log(result, "Calling cloud LLM for NL → JSON extraction…")
        user_req = extract_user_request(query)
        _append_log(result, "LLM extraction complete.")
    else:
        _append_log(result, "Parsing query as raw JSON…")
        user_req = parse_query_as_json(query)

    result.stage = "validate"
    _append_log(result, "Validating extracted request…")
    ok, issues, questions = validate_user_request(user_req)
    if not ok:
        result.clarifying_questions = questions
        detail = "; ".join(issues) if issues else "needs clarification"
        raise ValueError(f"Request validation failed: {detail}")

    baseline = _read_json(baseline_path)
    _append_log(result, f"Merging with baseline `{baseline_path.name}`…")
    effective = build_effective_config(baseline, user_req)
    if effective.get("run_mode"):
        os.environ["MORTALITY_RUN_MODE"] = str(effective["run_mode"])
    _append_log(result, "Effective config ready.")
    return user_req, effective


def _default_r_script() -> Path:
    env = os.environ.get("R_PIPELINE_SCRIPT", "").strip()
    if env:
        return Path(env)
    return DEFAULT_R_SCRIPT


def _run_r_pipeline(effective_path: Path, result: RunResult) -> Path:
    """Shell out to Rscript with live log pumping."""
    result.stage = "r_running"
    script = _default_r_script()
    if not script.is_file():
        raise FileNotFoundError(
            f"R script not found: {script}. Expected under r/ or set R_PIPELINE_SCRIPT."
        )

    out_root = OUTPUT_DIR / "comomo_live"
    out_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MORTALITY_EFFECTIVE_CONFIG"] = str(effective_path.resolve())
    env["MORTALITY_OUTPUT_DIR"] = str(out_root.resolve())
    # Prefer LLM/effective forecast year over a stale env horizon override.
    env.pop("MORTALITY_FORECAST_HORIZON", None)
    timeout = int(os.environ.get("R_PIPELINE_TIMEOUT_SEC", "28800"))

    try:
        use_r_dir = script.resolve().is_relative_to(R_DIR.resolve())
    except (ValueError, AttributeError):
        use_r_dir = str(script.resolve()).startswith(str(R_DIR.resolve()))
    cwd = str(R_DIR if use_r_dir else script.parent)
    _append_log(result, f"$ Rscript {script.name} (cwd={cwd})")
    _append_log(result, f"Output dir: {out_root}")

    proc = subprocess.Popen(
        ["Rscript", str(script.resolve())],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if line == "":
                break
            _append_log(result, line.rstrip("\n"))
        proc.stdout.close()

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        raise RuntimeError(f"R pipeline timed out after {timeout}s")
    pump.join(timeout=30)
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


def _tables_from_run_dir(run_dir: Path, max_rows: int = 200) -> Dict[str, pd.DataFrame]:
    specs = {
        "mse_by_horizon": ("mse_by_horizon_output_comomo_api_*.csv", "mse_by_horizon_*.csv"),
        "mse_average_ranking": (
            "mse_average_ranking_output_comomo_api_*.csv",
            "mse_average_ranking_*.csv",
            "mse_summary_table_*.csv",
        ),
        "forecast_rates_preview": (
            "forecast_future_rates_*.csv",
            "forecast_rates_with_mse_output_comomo_api_*.csv",
        ),
        "weights_table": ("weights_table_*.csv",),
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


def _finalize_answer(
    result: RunResult,
    *,
    query: str,
    user_req: Dict[str, Any],
    effective: Dict[str, Any],
    pipeline_mode: PipelineMode,
) -> None:
    result.interpretation = interpretation_snapshot(user_req, effective)
    result.stage = "sanity_check"
    _append_log(result, "Running deterministic + LLM sanity checks…")
    result.sanity_check = run_result_checks(
        query=query,
        user_request=user_req,
        effective_config=effective,
        tables=result.tables,
        files=result.files,
    )
    ok = bool(result.sanity_check.get("ok"))
    _append_log(
        result,
        "Sanity check "
        + ("PASSED" if ok else "FAILED: " + "; ".join(result.sanity_check.get("issues") or [])),
    )

    result.stage = "answer"
    _append_log(result, "Writing plain-English final answer…")
    result.plain_english_answer = build_plain_english_answer(
        query=query,
        user_request=user_req,
        effective_config=effective,
        tables=result.tables,
        mode=pipeline_mode,
        use_llm=True,
        sanity_check=result.sanity_check,
    )
    _append_log(result, "Plain-English answer ready.")

    if ok:
        result.status = "completed"
        result.message = (
            "Live R pipeline completed; sanity check passed."
            if pipeline_mode == "full"
            else "Demo run completed; sanity check passed."
        )
    else:
        result.status = "completed_with_issues"
        result.message = (
            "Pipeline finished but sanity check reported issues. "
            "Review the Sanity check panel before trusting the numbers."
        )


def execute_run(
    query: str,
    *,
    use_llm: bool = True,
    baseline_path: Optional[Path] = None,
    mode: Optional[str] = None,
) -> RunResult:
    """Synchronous full pipeline (suitable for demo / short jobs)."""
    run_id = uuid.uuid4().hex[:12]
    pipeline_mode = resolve_mode(mode)
    baseline = baseline_path or BASELINE_CUSTOM
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    result = RunResult(
        run_id=run_id,
        status="running",
        mode=pipeline_mode,
        stage="queued",
        started_at=_now(),
    )
    _RUNS[run_id] = result

    try:
        user_req, effective = run_extract_and_validate(
            query,
            use_llm=use_llm,
            baseline_path=baseline,
            result=result,
        )
        user_path = STATE_DIR / f"{run_id}_user_request.json"
        eff_path = STATE_DIR / f"{run_id}_effective_config.json"
        _write_json(user_path, user_req)
        _write_json(eff_path, effective)
        result.user_request = user_req
        result.effective_config = effective

        if pipeline_mode == "demo":
            result.stage = "demo_artifacts"
            _append_log(result, "Demo mode: loading bundled sample CoMoMo artifacts…")
            result.tables = load_sample_tables()
            result.files = list(result.tables.keys())
            result.metadata = sample_run_manifest()
            result.metadata["baseline"] = baseline.name
        else:
            _append_log(result, "Full mode: launching R CoMoMo/StMoMo pipeline…")
            run_dir = _run_r_pipeline(eff_path, result)
            result.tables = _tables_from_run_dir(run_dir)
            result.files = [p.name for p in sorted(run_dir.iterdir()) if p.is_file()]
            result.metadata = {
                "mode": "full",
                "run_dir": str(run_dir),
                "baseline": baseline.name,
            }

        _finalize_answer(
            result,
            query=query,
            user_req=user_req,
            effective=effective,
            pipeline_mode=pipeline_mode,
        )
    except ValueError as ex:
        msg = str(ex)
        if result.clarifying_questions or msg.startswith("Request validation failed"):
            result.status = "needs_clarification"
            result.error = "ValidationError"
            result.message = msg
            result.stage = "validate"
        else:
            result.status = "failed"
            result.error = "ValueError"
            result.message = msg
        _append_log(result, f"ERROR: {ex}")
    except json.JSONDecodeError as ex:
        result.status = "failed"
        result.error = "JSONDecodeError"
        result.message = (
            "The LLM returned invalid JSON while reading your request. "
            f"Please click Run again. Details: {ex}"
        )
        _append_log(result, f"ERROR: {ex}")
    except Exception as ex:  # noqa: BLE001
        result.status = "failed"
        result.error = type(ex).__name__
        result.message = str(ex)
        _append_log(result, f"ERROR: {ex}")

    result.ended_at = _now()
    return result


def start_run_async(
    query: str,
    *,
    use_llm: bool = True,
    baseline_path: Optional[Path] = None,
    mode: Optional[str] = None,
) -> RunResult:
    """Start a background run (for long full R jobs). One active run at a time."""
    global _ACTIVE_RUN_ID
    with _RUN_LOCK:
        if _ACTIVE_RUN_ID is not None:
            active = _RUNS.get(_ACTIVE_RUN_ID)
            if active and active.status == "running":
                raise RuntimeError(
                    f"Another run is already active ({_ACTIVE_RUN_ID}). Wait for it to finish."
                )
        run_id = uuid.uuid4().hex[:12]
        pipeline_mode = resolve_mode(mode)
        result = RunResult(
            run_id=run_id,
            status="running",
            mode=pipeline_mode,
            stage="queued",
            started_at=_now(),
        )
        _RUNS[run_id] = result
        _ACTIVE_RUN_ID = run_id

    baseline = baseline_path or BASELINE_CUSTOM
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _worker() -> None:
        global _ACTIVE_RUN_ID
        try:
            # Reuse synchronous body by calling execute pieces with this result object.
            user_req, effective = run_extract_and_validate(
                query,
                use_llm=use_llm,
                baseline_path=baseline,
                result=result,
            )
            user_path = STATE_DIR / f"{run_id}_user_request.json"
            eff_path = STATE_DIR / f"{run_id}_effective_config.json"
            _write_json(user_path, user_req)
            _write_json(eff_path, effective)
            result.user_request = user_req
            result.effective_config = effective

            if pipeline_mode == "demo":
                result.stage = "demo_artifacts"
                _append_log(result, "Demo mode: loading bundled sample CoMoMo artifacts…")
                result.tables = load_sample_tables()
                result.files = list(result.tables.keys())
                result.metadata = sample_run_manifest()
                result.metadata["baseline"] = baseline.name
            else:
                _append_log(result, "Full mode: launching R CoMoMo/StMoMo pipeline…")
                run_dir = _run_r_pipeline(eff_path, result)
                result.tables = _tables_from_run_dir(run_dir)
                result.files = [p.name for p in sorted(run_dir.iterdir()) if p.is_file()]
                result.metadata = {
                    "mode": "full",
                    "run_dir": str(run_dir),
                    "baseline": baseline.name,
                }

            _finalize_answer(
                result,
                query=query,
                user_req=user_req,
                effective=effective,
                pipeline_mode=pipeline_mode,
            )
        except ValueError as ex:
            msg = str(ex)
            if result.clarifying_questions or msg.startswith("Request validation failed"):
                result.status = "needs_clarification"
                result.error = "ValidationError"
                result.message = msg
                result.stage = "validate"
            else:
                result.status = "failed"
                result.error = "ValueError"
                result.message = msg
            _append_log(result, f"ERROR: {ex}")
        except Exception as ex:  # noqa: BLE001
            result.status = "failed"
            result.error = type(ex).__name__
            result.message = str(ex)
            _append_log(result, f"ERROR: {ex}")
        finally:
            result.ended_at = _now()
            with _RUN_LOCK:
                if _ACTIVE_RUN_ID == run_id:
                    _ACTIVE_RUN_ID = None

    threading.Thread(target=_worker, daemon=True).start()
    return result
