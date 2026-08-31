# Live demo API

A thin FastAPI wrapper around the existing, unmodified
`mhrag.agent.loop.run_agentic_retrieval` — see `app.py`'s module docstring
for the full design rationale. This is product functionality for the
portfolio frontend's "Live Agentic RAG Demo" section, not another
benchmark run: it makes real (small) Mantle API calls per question asked.

## Run

```bash
# From the project root, with the same .venv used for everything else:
pip install -e ".[demo]"

# Reads $OPENAI_API_KEY / $MANTLE_BASE_URL from the project root .env —
# start it from the project root, or `source .env` first either way.
uvicorn backend.app:app --reload --port 8000
```

Requires the same live Qdrant collection every other live script in this
project uses (`docker compose up -d qdrant`, already indexed).

## Endpoints

- `GET /healthz` — liveness check.
- `POST /api/ask` — `{"question": string}` → `AskResponse` (see `app.py`).
  Rejects empty/oversized questions (422), rate-limits per client IP (429),
  and times out (504) rather than hanging — see `app.py` for the exact
  limits.

## Test

```bash
cd backend
pytest tests/ -q
```

Offline — a fake pipeline stands in for the real one, so these tests never
load a model, touch Qdrant, or call Mantle.

## Security

- The Mantle API key is read server-side only (`python-dotenv`, same
  environment-variable contract `mhrag.generation.mantle_client` already
  uses) and never appears in any response.
- CORS is restricted to `DEMO_ALLOWED_ORIGINS` (defaults to
  `http://localhost:3000`) — never `*`.
- No database, no session, no auth — this is a stateless demo endpoint.
