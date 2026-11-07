# Reliable LLM Build

Deployable rebuild of the mortality-forecast demo.

**What changed vs the submission prototype**

| Old | New |
|-----|-----|
| FastAPI + Streamlit + local Llama GGUF (~5 GB) | **One Streamlit app** + **cloud LLM** (Groq free tier by default) |
| Hard to host publicly | **Demo mode** for Streamlit Community Cloud / Hugging Face Spaces |
| Full R run required for every click | Live **NL → JSON → effective config**; sample CoMoMo tables for public demos |
| Optional full R still available | Set `PIPELINE_MODE=full` + `R_PIPELINE_SCRIPT` locally |

Original submission files are untouched.

## Quick start (local)

```bash
cd "reliable llm build"
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and paste a free Groq key: https://console.groq.com/keys
streamlit run app.py
```

Open http://localhost:8501

## Free public hosting (recommended)

1. Push **this folder** to a GitHub repo (or only this folder as the repo root).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set main file path to `app.py`.
4. In **Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_..."
PIPELINE_MODE = "demo"
```

5. Deploy. Visitors get live LLM extraction + cached sample MSE/forecast tables.

**Why demo mode on free hosts:** CoMoMo/StMoMo + HMD can run for hours and needs R. Free Streamlit apps cannot host that stack reliably for many concurrent users.

## Pipeline modes

### `demo` (default)
1. Cloud LLM extracts request JSON from natural language.
2. Validator merges into baseline effective config.
3. UI shows bundled sample outputs under `sample_data/run_demo/`.

### `full` (local / lab machine)
1. Same extract + validate.
2. Calls `Rscript` on `R_PIPELINE_SCRIPT` (point at the original
   `rolling_window_check_mse_comomo_api_fixed_forecast_final.R`).
3. Needs HMD credentials, R, StMoMo, CoMoMo installed beside that script.

## LLM providers

Any OpenAI-compatible chat API works:

```env
LLM_API_KEY=...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
```

Groq’s free tier is the intended default for class demos.

Without an API key you can turn off **Use cloud LLM extraction** and paste JSON in the text box (see `examples/user_request_sample.json`).

## Layout

```
reliable llm build/
  app.py                 # Streamlit entry
  src/
    llm_extract.py       # Cloud LLM NL → JSON
    config_validate.py   # Baseline merge + HMD country checks
    pipeline.py          # Orchestration (demo | full)
    demo_artifacts.py    # Cached sample tables
  config/                # Baseline JSON templates
  prompts/               # Extractor system prompt
  sample_data/run_demo/  # Sample CoMoMo CSVs for public demo
```

## Notes

- Do not commit `.env` or Streamlit `secrets.toml`.
- Old FastAPI/UI/local-GGUF code remains in the submission folder for reference only.
