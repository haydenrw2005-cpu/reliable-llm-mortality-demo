"""Reliable LLM mortality demo — Streamlit UI (live R on VM; demo fallback)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_load import load_dotenv  # noqa: E402
from src.llm_extract import llm_status  # noqa: E402
from src.paths import BASELINE_CUSTOM, BASELINE_LOCKED  # noqa: E402
from src.pipeline import (  # noqa: E402
    active_run_id,
    get_run,
    resolve_mode,
    start_run_async,
)

load_dotenv()


def _apply_streamlit_secrets() -> None:
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
        "R_PIPELINE_TIMEOUT_SEC",
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

STAGE_LABELS = {
    "queued": "Queued",
    "extract": "1. Extract request (LLM)",
    "validate": "2. Validate inputs",
    "demo_artifacts": "3. Load demo artifacts",
    "r_running": "3. Run CoMoMo R pipeline",
    "sanity_check": "4. Sanity-check results",
    "answer": "5. Plain-English answer",
}

st.set_page_config(
    page_title="Reliable LLM · Mortality Demo",
    page_icon="📉",
    layout="wide",
)

st.title("Reliable LLM — Mortality Forecast")
st.caption(
    "Plain English → LLM validates config → CoMoMo/StMoMo (R) → "
    "LLM sanity-check → plain-English answer."
)

if "last_run_id" not in st.session_state:
    st.session_state["last_run_id"] = ""
if "query_text" not in st.session_state:
    st.session_state["query_text"] = DEFAULT_QUERY

with st.sidebar:
    st.header("Settings")
    mode_default = resolve_mode()
    mode = st.radio(
        "Pipeline mode",
        ["full", "demo"],
        index=0 if mode_default == "full" else 1,
        help=(
            "full = live R CoMoMo on this host (VM). "
            "demo = cached sample tables (Streamlit Cloud / smoke tests)."
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

    if mode == "full":
        st.info(
            "Full mode runs the paper-style R pipeline (can take a long time). "
            "Needs R, CoMoMo/StMoMo, and HMD credentials on this machine/VM."
        )

st.text_area("Natural-language request", height=140, key="query_text")

busy = False
aid = active_run_id()
if aid:
    ar = get_run(aid)
    busy = bool(ar and ar.status == "running")

col_run, col_ex = st.columns([1, 1])
with col_run:
    run_clicked = st.button(
        "Run",
        type="primary",
        width="stretch",
        disabled=busy,
    )
with col_ex:
    if st.button("Load JSON example", width="stretch", disabled=busy):
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
        try:
            result = start_run_async(
                query,
                use_llm=use_llm,
                baseline_path=baseline_path,
                mode=mode,
            )
            st.session_state["last_run_id"] = result.run_id
            st.rerun()
        except RuntimeError as ex:
            st.error(str(ex))

run_id = (st.session_state.get("last_run_id") or "").strip()
if not run_id:
    st.info("Enter a request and click **Run**.")
    st.stop()

result = get_run(run_id)
if result is None:
    st.warning("Run not found in this server process (restart clears in-memory runs).")
    st.stop()

st.subheader(f"Run `{result.run_id}`")
meta_cols = st.columns(4)
meta_cols[0].metric("Status", result.status)
meta_cols[1].metric("Mode", result.mode)
meta_cols[2].metric("Stage", STAGE_LABELS.get(result.stage, result.stage))
meta_cols[3].write(f"**Started**\n\n{result.started_at}")

# Stage checklist
st.markdown("### Workflow")
stages_order = ["extract", "validate", "r_running" if result.mode == "full" else "demo_artifacts", "sanity_check", "answer"]
stage_idx = {s: i for i, s in enumerate(stages_order)}
cur = stage_idx.get(result.stage, -1)
if result.status in {"completed", "completed_with_issues"}:
    cur = len(stages_order)
cols = st.columns(len(stages_order))
for i, s in enumerate(stages_order):
    label = STAGE_LABELS.get(s, s)
    if i < cur:
        cols[i].success(label)
    elif i == cur and result.status == "running":
        cols[i].warning(label + " …")
    elif result.status in {"failed", "needs_clarification"} and i == cur:
        cols[i].error(label)
    else:
        cols[i].caption(label)

if result.status == "running":
    st.info(
        "Run in progress — auto-refreshing. Full R jobs can take a long time; "
        "watch the execution log below."
    )
    with st.expander("Live execution log", expanded=True):
        for line in result.logs[-80:]:
            st.code(line, language=None)
    time.sleep(2)
    st.rerun()

if result.status == "needs_clarification":
    st.warning(result.message or "Need clarification before running R.")
    if result.clarifying_questions:
        st.markdown("**Please answer:**")
        for q in result.clarifying_questions:
            st.write(f"- {q}")
elif result.status == "failed":
    st.error(f"{result.error}: {result.message}")
elif result.status == "completed_with_issues":
    st.warning(result.message)
elif result.status == "completed":
    st.success(result.message or "Completed.")

if result.interpretation:
    st.markdown("### What we understood from your request")
    c1, c2, c3 = st.columns(3)
    c1.metric("Country", result.interpretation.get("country", "—"))
    c2.metric("Sex", result.interpretation.get("sex", "—"))
    c3.metric("Ages", result.interpretation.get("ages", "—"))
    c4, c5, c6 = st.columns(3)
    c4.metric("Fit period", result.interpretation.get("fit_period", "—"))
    c5.metric("Test period", result.interpretation.get("test_period", "—"))
    c6.metric("Forecast year", result.interpretation.get("forecast_target_year", "—"))
    st.caption(f"Models: **{result.interpretation.get('models', '—')}**")

if result.sanity_check:
    st.markdown("### Sanity check")
    if result.sanity_check.get("ok"):
        st.success("Results look consistent with the request.")
    else:
        st.error("Sanity check reported issues.")
    for issue in result.sanity_check.get("issues") or []:
        st.write(f"- {issue}")
    points = result.sanity_check.get("summary_points") or []
    if points:
        st.markdown("**Summary points**")
        for p in points:
            st.write(f"- {p}")

if result.plain_english_answer:
    st.markdown("### Answer")
    if result.status == "completed_with_issues":
        st.caption("Answer withheld from confident LLM polish because sanity check failed.")
    st.markdown(result.plain_english_answer)

with st.expander("Execution log", expanded=result.status in {"failed", "needs_clarification"}):
    for line in result.logs:
        st.code(line, language=None)

if result.status not in {"completed", "completed_with_issues"}:
    st.stop()

if result.mode == "demo":
    st.warning(
        "Demo mode used cached sample CoMoMo tables. "
        "Switch to **full** on a VM with R/HMD for live numbers."
    )

st.markdown("### Details")
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
    st.json(result.user_request) if result.user_request else st.caption("—")

with tab_cfg:
    st.json(result.effective_config) if result.effective_config else st.caption("—")

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
        requested = {str(m).upper() for m in (result.user_request.get("models") or [])}
        if requested and "model" in df.columns:
            highlight = df[df["model"].astype(str).str.upper().isin(requested)]
            if not highlight.empty:
                st.markdown("**Your named models**")
                st.dataframe(highlight, width="stretch")
        st.dataframe(df, width="stretch")
    else:
        st.caption("No ranking table.")

with tab_fc:
    df = result.tables.get("forecast_rates_preview")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, width="stretch")
    else:
        st.caption("No forecast preview.")

with tab_w:
    df = result.tables.get("weights_table")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, width="stretch")
    else:
        st.caption("No weights table.")

if result.metadata:
    with st.expander("Run metadata"):
        st.json(result.metadata)
