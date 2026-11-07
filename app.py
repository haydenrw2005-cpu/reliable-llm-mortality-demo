"""Reliable LLM mortality demo — single Streamlit app (cloud LLM + demo/full pipeline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_load import load_dotenv  # noqa: E402
from src.llm_extract import llm_status  # noqa: E402
from src.paths import BASELINE_CUSTOM, BASELINE_LOCKED  # noqa: E402
from src.pipeline import execute_run, resolve_mode  # noqa: E402

load_dotenv()


def _apply_streamlit_secrets() -> None:
    """Map Streamlit Cloud secrets into os.environ (no overwrite if already set)."""
    import os

    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001
        return
    for key in (
        "GROQ_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "PIPELINE_MODE",
        "R_PIPELINE_SCRIPT",
        "HMD_USERNAME",
        "HMD_PASSWORD",
    ):
        try:
            val = secrets.get(key)
        except Exception:  # noqa: BLE001
            continue
        if val and not os.environ.get(key):
            os.environ[key] = str(val)


_apply_streamlit_secrets()

DEFAULT_QUERY = (
    "Predict mortality for GBRTENW male, ages 50-89, fit 1960-1990, "
    "test 1991-2015, models LC RH APC CBD M7 PLAT, return mse and forecast rates."
)

st.set_page_config(
    page_title="Reliable LLM · Mortality Demo",
    page_icon="📉",
    layout="wide",
)

st.title("Reliable LLM — Mortality Forecast Demo")
st.caption(
    "Natural language → structured config → CoMoMo/StMoMo results. "
    "Cloud LLM extraction; demo mode uses cached sample outputs for free hosting."
)

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "query_text" not in st.session_state:
    st.session_state["query_text"] = DEFAULT_QUERY

with st.sidebar:
    st.header("Settings")
    mode_default = resolve_mode()
    mode = st.radio(
        "Pipeline mode",
        ["demo", "full"],
        index=0 if mode_default == "demo" else 1,
        help=(
            "Demo = live LLM + cached sample tables (best for public free sites). "
            "Full = local Rscript + HMD credentials (slow)."
        ),
    )
    baseline_choice = st.selectbox(
        "Baseline config",
        ["customisable", "locked"],
        format_func=lambda x: (
            "Customisable (UK male defaults)"
            if x == "customisable"
            else "Locked baseline"
        ),
    )
    baseline_path = BASELINE_CUSTOM if baseline_choice == "customisable" else BASELINE_LOCKED
    use_llm = st.checkbox("Use cloud LLM extraction", value=True)

    status = llm_status()
    st.markdown("### LLM")
    if status["configured"] == "yes":
        st.success("API key detected")
    else:
        st.warning("No API key — set GROQ_API_KEY or LLM_API_KEY")
    st.caption(f"Model: `{status['model']}`")
    st.caption(f"Endpoint: `{status['base_url']}`")
    st.caption(status["provider_hint"])

    if mode == "full":
        st.info(
            "Full mode needs R, CoMoMo/StMoMo, HMD credentials, and "
            "`R_PIPELINE_SCRIPT` pointing at the original R entry script."
        )

st.text_area("Natural-language request", height=140, key="query_text")

col_run, col_ex = st.columns([1, 1])
with col_run:
    run_clicked = st.button("Run", type="primary", width="stretch")
with col_ex:
    if st.button("Load JSON example", width="stretch"):
        example = {
            "status": "ready",
            "country": "USA",
            "gender": "female",
            "ages": {"min": 60, "max": 90},
            "fit_period": {"start": 1970, "end": 2000},
            "test_period": {"start": 2001, "end": 2010},
            "models": ["LC", "APC"],
            "outcomes": ["mse", "forecast_rates"],
        }
        st.session_state["query_text"] = json.dumps(example, indent=2)
        st.rerun()

if run_clicked:
    query = (st.session_state.get("query_text") or "").strip()
    if not query:
        st.error("Enter a prompt or JSON.")
    else:
        with st.spinner("Running extract → validate → artifacts…"):
            result = execute_run(
                query,
                use_llm=use_llm,
                baseline_path=baseline_path,
                mode=mode,
            )
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result is None:
    st.info("Enter a request and click **Run**.")
    st.stop()

st.subheader(f"Run `{result.run_id}`")
meta_cols = st.columns(4)
meta_cols[0].metric("Status", result.status)
meta_cols[1].metric("Mode", result.mode)
meta_cols[2].write(f"**Started**\n\n{result.started_at}")
meta_cols[3].write(f"**Ended**\n\n{result.ended_at or '—'}")

if result.status == "completed":
    st.success(result.message or "Completed.")
else:
    st.error(f"{result.error}: {result.message}")

with st.expander("Execution log", expanded=result.status != "completed"):
    for line in result.logs:
        st.code(line, language=None)

tab_req, tab_cfg, tab_mse, tab_rank, tab_fc, tab_w = st.tabs(
    [
        "User request",
        "Effective config",
        "MSE by horizon",
        "Average ranking",
        "Forecast preview",
        "Weights",
    ]
)

with tab_req:
    if result.user_request:
        st.json(result.user_request)
    else:
        st.caption("No user request (run failed before extract finished).")

with tab_cfg:
    if result.effective_config:
        st.json(result.effective_config)
    else:
        st.caption("No effective config.")

with tab_mse:
    df = result.tables.get("mse_by_horizon")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, width="stretch")
    else:
        st.caption("No MSE-by-horizon table.")

with tab_rank:
    df = result.tables.get("mse_average_ranking")
    if not isinstance(df, pd.DataFrame) or df.empty:
        df = result.tables.get("mse_summary")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, width="stretch")
    else:
        st.caption("No ranking table.")

with tab_fc:
    df = result.tables.get("forecast_rates_preview")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, width="stretch")
        st.caption("Preview rows only (demo/full UI cap).")
    else:
        st.caption("No forecast preview.")

with tab_w:
    df = result.tables.get("weights_table")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, width="stretch")
    else:
        st.caption("No weights table in this run.")

if result.metadata:
    with st.expander("Run metadata"):
        st.json(result.metadata)
