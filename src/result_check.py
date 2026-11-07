"""Deterministic + LLM sanity checks on CoMoMo output tables."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .llm_extract import _chat_completion, parse_first_json_object


def _finite_mse(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().all():
        return False
    return bool(vals.dropna().apply(lambda x: abs(float(x)) < 1e6).all())


def deterministic_checks(
    *,
    user_request: Dict[str, Any],
    effective_config: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    files: List[str],
) -> List[str]:
    issues: List[str] = []
    locked = bool(effective_config.get("lock_to_baseline", False))

    rank = tables.get("mse_average_ranking")
    if not isinstance(rank, pd.DataFrame) or rank.empty:
        rank = tables.get("mse_summary")
    mse = tables.get("mse_by_horizon")

    if not isinstance(rank, pd.DataFrame) or rank.empty:
        issues.append("Missing or empty average-ranking / MSE summary table.")
    else:
        cols = {c.lower(): c for c in rank.columns}
        mse_col = cols.get("avg_mse") or cols.get("mse")
        model_col = cols.get("model")
        if mse_col is None or not _finite_mse(rank[mse_col]):
            issues.append("Ranking table MSE values are missing or not finite.")
        requested = {str(m).upper() for m in (user_request.get("models") or [])}
        if requested and model_col:
            present = {str(v).upper() for v in rank[model_col].tolist()}
            missing = sorted(requested - present)
            # Stacking ensembles may dominate; still expect named base models in ranking output.
            if missing:
                issues.append(
                    "Requested models not found in ranking output: " + ", ".join(missing)
                )

    if not isinstance(mse, pd.DataFrame) or mse.empty:
        issues.append("Missing or empty MSE-by-horizon table.")
    else:
        cols = {c.lower(): c for c in mse.columns}
        mse_col = cols.get("mse")
        if mse_col is None or not _finite_mse(mse[mse_col]):
            issues.append("MSE-by-horizon values are missing or not finite.")

    target = effective_config.get("forecast_target_year") or user_request.get(
        "forecast_target_year"
    )
    if target not in (None, ""):
        fc = tables.get("forecast_rates_preview")
        has_fc_file = any("forecast" in f.lower() for f in files)
        if (not isinstance(fc, pd.DataFrame) or fc.empty) and not has_fc_file:
            issues.append(
                "forecast_target_year was set but no forecast rates output was found."
            )

    if not locked:
        hmd = effective_config.get("hmd") or {}
        req_country = str(user_request.get("country") or "").strip()
        req_gender = str(user_request.get("gender") or "").strip().lower()
        if req_country and str(hmd.get("country") or "").strip() != req_country:
            issues.append(
                f"Effective country {hmd.get('country')!r} does not match request {req_country!r}."
            )
        if req_gender and str(hmd.get("sex") or "").strip().lower() != req_gender:
            issues.append(
                f"Effective sex {hmd.get('sex')!r} does not match request {req_gender!r}."
            )

    return issues


def _table_digest(tables: Dict[str, pd.DataFrame], max_rows: int = 8) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, df in tables.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        out[key] = json.loads(df.head(max_rows).to_json(orient="records"))
    return out


def llm_sanity_check(
    *,
    query: str,
    user_request: Dict[str, Any],
    effective_config: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    deterministic_issues: List[str],
) -> Dict[str, Any]:
    """Return {ok, issues, summary_points} from the LLM (with fallback)."""
    payload = {
        "user_query": query,
        "user_request": user_request,
        "effective_config_hmd": effective_config.get("hmd"),
        "run_mode": effective_config.get("run_mode"),
        "forecast_target_year": effective_config.get("forecast_target_year"),
        "deterministic_issues": deterministic_issues,
        "table_previews": _table_digest(tables),
    }
    system = (
        "You sanity-check mortality CoMoMo pipeline outputs. "
        "Reply with ONLY JSON: "
        '{"ok": true|false, "issues": ["..."], "summary_points": ["..."]}. '
        "ok=false if results contradict the request, tables look empty/corrupt, "
        "or deterministic_issues are material. "
        "summary_points should be short factual bullets useful for a plain-English answer. "
        "Do not invent MSE numbers not present in table_previews."
    )
    try:
        raw = _chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)[:12000]},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        parsed = parse_first_json_object(raw)
    except Exception as ex:  # noqa: BLE001
        return {
            "ok": len(deterministic_issues) == 0,
            "issues": list(deterministic_issues)
            + [f"LLM sanity-check unavailable: {ex}"],
            "summary_points": [],
        }

    ok = bool(parsed.get("ok"))
    issues = [str(x) for x in (parsed.get("issues") or [])]
    # Always include deterministic issues.
    for d in deterministic_issues:
        if d not in issues:
            issues.append(d)
    if deterministic_issues:
        ok = False
    points = [str(x) for x in (parsed.get("summary_points") or [])]
    return {"ok": ok, "issues": issues, "summary_points": points}


def run_result_checks(
    *,
    query: str,
    user_request: Dict[str, Any],
    effective_config: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    files: List[str],
) -> Dict[str, Any]:
    det = deterministic_checks(
        user_request=user_request,
        effective_config=effective_config,
        tables=tables,
        files=files,
    )
    return llm_sanity_check(
        query=query,
        user_request=user_request,
        effective_config=effective_config,
        tables=tables,
        deterministic_issues=det,
    )
