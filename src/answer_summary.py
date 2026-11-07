"""Build a plain-English final answer from request + mortality tables."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import requests

from .llm_extract import _resolve_credentials

COUNTRY_NAMES = {
    "GBRTENW": "England & Wales (UK)",
    "GBRCENW": "England & Wales (civilian)",
    "GBR_NP": "United Kingdom",
    "GBR_SCO": "Scotland",
    "GBR_NIR": "Northern Ireland",
    "USA": "United States",
    "SWE": "Sweden",
    "AUS": "Australia",
    "CAN": "Canada",
    "NOR": "Norway",
    "DNK": "Denmark",
    "NLD": "Netherlands",
    "FRA": "France",
    "FRATNP": "France",
    "DEUTNP": "Germany",
    "ITA": "Italy",
    "ESP": "Spain",
    "JPN": "Japan",
}


def _country_label(code: str) -> str:
    code = (code or "").strip()
    return COUNTRY_NAMES.get(code, code or "the selected population")


def _pick_ranking(tables: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    for key in ("mse_average_ranking", "mse_summary"):
        df = tables.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    return None


def _top_models(rank_df: pd.DataFrame, n: int = 3) -> list[tuple[str, float, int]]:
    df = rank_df.copy()
    cols = {c.lower(): c for c in df.columns}
    model_col = cols.get("model")
    mse_col = cols.get("avg_mse") or cols.get("mse")
    rank_col = cols.get("ranking") or cols.get("rank")
    if model_col is None:
        return []
    if rank_col:
        df = df.sort_values(rank_col, ascending=True)
    elif mse_col:
        df = df.sort_values(mse_col, ascending=True)
    out: list[tuple[str, float, int]] = []
    for i, (_, row) in enumerate(df.head(n).iterrows(), start=1):
        name = str(row[model_col])
        mse = float(row[mse_col]) if mse_col and pd.notna(row[mse_col]) else float("nan")
        rank = int(row[rank_col]) if rank_col and pd.notna(row.get(rank_col)) else i
        out.append((name, mse, rank))
    return out


def build_factual_summary(
    *,
    query: str,
    user_request: Dict[str, Any],
    effective_config: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    mode: str,
) -> str:
    hmd = effective_config.get("hmd") or {}
    country = _country_label(str(hmd.get("country") or user_request.get("country") or ""))
    sex = str(hmd.get("sex") or user_request.get("gender") or "unspecified")
    ages = effective_config.get("ages") or user_request.get("ages") or {}
    fit = effective_config.get("fit_period") or user_request.get("fit_period") or {}
    rolling = effective_config.get("rolling_origins") or {}
    test = user_request.get("test_period") or {}
    models = effective_config.get("models") or user_request.get("models") or []
    target = effective_config.get("forecast_target_year") or user_request.get(
        "forecast_target_year"
    )
    run_mode = effective_config.get("run_mode", "evaluation")

    age_bit = ""
    if isinstance(ages, dict) and ages.get("min") is not None and ages.get("max") is not None:
        age_bit = f" ages {ages['min']}–{ages['max']},"

    fit_bit = ""
    if isinstance(fit, dict) and fit.get("start") is not None and fit.get("end") is not None:
        fit_bit = f" fitted on {fit['start']}–{fit['end']},"

    test_bit = ""
    period = rolling if rolling else test
    if isinstance(period, dict) and period.get("start") is not None and period.get("end") is not None:
        test_bit = f" evaluated over {period['start']}–{period['end']},"

    model_bit = ", ".join(str(m) for m in models) if models else "the baseline model set"

    lines = [
        f"You asked about mortality for {country} ({sex}),{age_bit}{fit_bit}{test_bit} "
        f"using models: {model_bit}.",
    ]

    if target:
        lines.append(
            f"A forward forecast target year of {target} was recognised "
            f"(run mode: {run_mode})."
        )
    else:
        lines.append(
            "No specific future calendar year was requested, so this run focuses on "
            f"backtesting / model comparison (run mode: {run_mode})."
        )

    rank_df = _pick_ranking(tables)
    tops = _top_models(rank_df, 5) if rank_df is not None else []
    if tops:
        best_name, best_mse, best_rank = tops[0]
        lines.append(
            f"On average MSE across horizons, the best-performing method is {best_name} "
            f"(rank {best_rank}, average MSE ≈ {best_mse:.6f})."
        )
        board = "; ".join(
            f"{name}: rank {rank}, avg MSE ≈ {mse:.6f}" for name, mse, rank in tops
        )
        lines.append(f"Top methods: {board}.")

        # Highlight any base models the user asked for, if present in ranking.
        requested = {str(m).upper() for m in (user_request.get("models") or [])}
        if requested and rank_df is not None:
            cols = {c.lower(): c for c in rank_df.columns}
            model_col = cols.get("model")
            mse_col = cols.get("avg_mse") or cols.get("mse")
            rank_col = cols.get("ranking") or cols.get("rank")
            if model_col and mse_col:
                highlights = []
                for _, row in rank_df.iterrows():
                    name = str(row[model_col])
                    if name.upper() in requested:
                        mse = float(row[mse_col])
                        rk = int(row[rank_col]) if rank_col and pd.notna(row[rank_col]) else "?"
                        highlights.append(f"{name}: rank {rk}, avg MSE ≈ {mse:.6f}")
                if highlights:
                    lines.append(
                        "Among the models named in the request: " + "; ".join(highlights) + "."
                    )
        lines.append(
            "Lower average MSE means predicted mortality rates were closer to observed rates "
            "on the evaluation windows."
        )
    else:
        lines.append("Ranking tables were not available for this run.")

    if mode == "demo":
        lines.append(
            "Note: this public demo uses live language understanding for your request, "
            "but the numerical tables are cached sample CoMoMo outputs so the site stays fast. "
            "Treat the numbers as an illustrative worked example, not a fresh HMD download."
        )

    lines.append(f'Original request: “{query.strip()}”')
    return "\n\n".join(lines)


def llm_plain_english_answer(facts: str, query: str) -> str:
    """Ask the cloud LLM to rewrite facts as a short plain-English answer."""
    api_key, base_url, model = _resolve_credentials()
    system = (
        "You are a mortality-forecasting assistant. Write a clear plain-English final answer "
        "for a non-technical reader. Use ONLY the verified facts provided. "
        "Do not invent numbers. Do not say data is missing if the facts already list ranks or MSE. "
        "Paragraph 1 MUST name the exact country, sex, ages, years, and models from THIS request "
        "(so different inputs produce clearly different answers). "
        "Paragraph 2: among models the user named, report their ranks/MSE from the facts; "
        "then mention the overall best method in the sample ranking. "
        "Paragraph 3: if facts say demo/cached sample, say clearly that the MSE numbers are an "
        "illustrative sample and would change with a full live CoMoMo/HMD run. "
        "2–4 short paragraphs. No JSON. No markdown headings. Bold one model name once."
    )
    user = (
        f"User question:\n{query}\n\n"
        f"Verified facts from the pipeline:\n{facts}\n\n"
        "Write the final plain-English answer now. Make paragraph 1 specific to this request."
    )
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.2,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=90,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM summary error {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def interpretation_snapshot(
    user_request: Dict[str, Any], effective_config: Dict[str, Any]
) -> Dict[str, str]:
    """Compact fields for the UI so users see their input was understood."""
    hmd = effective_config.get("hmd") or {}
    ages = effective_config.get("ages") or user_request.get("ages") or {}
    fit = effective_config.get("fit_period") or user_request.get("fit_period") or {}
    test = user_request.get("test_period") or effective_config.get("rolling_origins") or {}
    models = effective_config.get("models") or user_request.get("models") or []
    target = effective_config.get("forecast_target_year") or user_request.get(
        "forecast_target_year"
    )
    return {
        "country": _country_label(str(hmd.get("country") or user_request.get("country") or "")),
        "sex": str(hmd.get("sex") or user_request.get("gender") or "—"),
        "ages": (
            f"{ages.get('min')}–{ages.get('max')}"
            if isinstance(ages, dict) and ages.get("min") is not None
            else "—"
        ),
        "fit_period": (
            f"{fit.get('start')}–{fit.get('end')}"
            if isinstance(fit, dict) and fit.get("start") is not None
            else "—"
        ),
        "test_period": (
            f"{test.get('start')}–{test.get('end')}"
            if isinstance(test, dict) and test.get("start") is not None
            else "—"
        ),
        "models": ", ".join(str(m) for m in models) if models else "—",
        "forecast_target_year": str(target) if target not in (None, "") else "none (MSE / backtest)",
    }


def build_plain_english_answer(
    *,
    query: str,
    user_request: Dict[str, Any],
    effective_config: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    mode: str,
    use_llm: bool = True,
) -> str:
    facts = build_factual_summary(
        query=query,
        user_request=user_request,
        effective_config=effective_config,
        tables=tables,
        mode=mode,
    )
    # Demo mode: keep the deterministic facts as the answer so changing the prompt
    # always changes the visible text (LLM polish can blur request-specific details).
    if mode == "demo" or not use_llm:
        return facts
    has_key = bool(
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("GROQ_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not has_key:
        return facts
    try:
        return llm_plain_english_answer(facts, query)
    except Exception:  # noqa: BLE001
        return facts
