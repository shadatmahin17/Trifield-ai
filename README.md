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

1. Push this repository to GitHub.
2. In Render, create a new **Blueprint** from the repository. Render will read `render.yaml`.
3. Add the required secret environment variable:
   - `ANTHROPIC_API_KEY`
4. Deploy the service.

If creating a Render web service manually instead of using the blueprint, use:

- **Runtime:** Python
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

The API docs are available at `/docs` after deployment.
