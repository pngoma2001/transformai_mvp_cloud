# TransformAI MVP (Cloud)

A Streamlit-based demo for AI-driven transformation plays for PE portfolio companies.

## Quick Deploy (Streamlit Cloud)

1. Create a **public** GitHub repo (e.g., `transformai_mvp_cloud`).
2. Upload the **contents** of this folder to the repo **root** (do not nest inside subfolders).
3. Go to https://share.streamlit.io → **New app**.
4. Select your repo, set **Branch** = `main` and **Main file path** = `app.py` → **Deploy**.
5. Wait for build → your public URL will look like `https://<username>-transformai-mvp-cloud.streamlit.app`.

## Local Run (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo Structure

```
app.py
engine.py
models.py
requirements.txt
data/
```

## Notes

- No external APIs required.
- PDF export is a simple HTML download (you can print to PDF from your browser).
- This build avoids Pydantic for compatibility with Python 3.13 on Streamlit Cloud.
