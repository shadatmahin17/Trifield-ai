# TriField AI Backend

FastAPI backend for the TriField AI research workspace.

## Local development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> for the interactive API docs.

## Deploy on Render

This project pins Python to 3.12.13 with `.python-version` and `runtime.txt`. Do not let Render use its default Python 3.14 runtime because ChromaDB/tokenizers can fail to build on Python 3.14.

### Blueprint deploy

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** from the repository. Render will read `render.yaml`.
3. Add the required secret environment variable:
   - `ANTHROPIC_API_KEY`
4. Deploy the service.

### Manual web service deploy

If creating a Render web service manually instead of using the blueprint, use:

- **Runtime:** Python
- **Python version:** `3.12.13`
- **Build command:** `python -m pip install --upgrade pip && python -m pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

Also add these environment variables:

- `PYTHON_VERSION=3.12.13`
- `APP_ENV=production`
- `CHROMA_PATH=/opt/render/project/src/chroma_db`
- `ANTHROPIC_API_KEY=<your Anthropic API key>`

If the Render logs show `.venv/bin/python3.14`, set `PYTHON_VERSION=3.12.13` in the Render dashboard and redeploy.

The API docs are available at `/docs` after deployment.
