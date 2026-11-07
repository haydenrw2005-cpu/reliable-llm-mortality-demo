# Reliable LLM Build

Mortality chatbot workflow:

**Plain English → LLM validates config → CoMoMo/StMoMo (R) → LLM sanity-check → plain-English answer**

## Modes

| Mode | Behaviour | Where |
|------|-----------|--------|
| `full` (default on VM) | Live R + HMD | Docker / cloud VM |
| `demo` | Cached sample tables | Streamlit Community Cloud / smoke tests |

Public Streamlit Cloud demo: https://reliable-llm-mortality-demo-mjrv4jhlggrnbrwkhnccey.streamlit.app/  
Live R: follow [DEPLOY_VM.md](DEPLOY_VM.md).

## Local demo (no R)

```bash
pip install -r requirements.txt
copy .env.example .env
# set GROQ_API_KEY; PIPELINE_MODE=demo
streamlit run app.py
```

## Local / VM full mode

Needs R, StMoMo, CoMoMo (`r/CoMoMo`), HMD credentials. Easiest path:

```bash
cp .env.example .env   # GROQ + HMD + PIPELINE_MODE=full
docker compose up -d --build
```

See [DEPLOY_VM.md](DEPLOY_VM.md).

## Layout

```
app.py                 Streamlit UI (async runs + stage progress)
src/
  llm_extract.py       NL → JSON
  request_validate.py  Pre-R validation
  config_validate.py   Baseline merge
  pipeline.py          Staged orchestration + R subprocess
  result_check.py      Deterministic + LLM sanity check
  answer_summary.py    Plain-English answer
r/
  rolling_window_check_mse_comomo_api_fixed_forecast_final.R
  CoMoMo/
```

Original assessment submission folder is untouched; R/CoMoMo were copied into `r/`.
