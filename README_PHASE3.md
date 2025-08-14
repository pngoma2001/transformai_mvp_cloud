# Phase 3 — AI Mode (Optional)

This phase adds **AI-generated plays** via OpenAI or Anthropic, with **schema validation** and **safe fallbacks**.

## How to use AI Mode (Streamlit Cloud)

1. Open your Streamlit app → **⋯ → Settings → Secrets** and add one or both:
```
OPENAI_API_KEY = sk-xxxxxxxx
ANTHROPIC_API_KEY = xxxxxxxxx
```
2. In the app sidebar, toggle **Enable AI-generated plays**.
3. Choose **Provider** and **Model** (defaults provided).
4. Click **Analyze Company**. If AI fails or is missing, the app falls back to the rule-based engine.

## Local use
Set environment variables before launching Streamlit:
```bash
export OPENAI_API_KEY=sk-xxx
export ANTHROPIC_API_KEY=xxx
streamlit run app.py
```

## Notes
- The app validates AI JSON against `schemas/analysis.schema.json` and reverts to rules on any error.
- If you never add a key or toggle AI off, everything works as before.
