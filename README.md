uv run uvicorn main:app --reload --port 8000
uv run uvicorn main:app --port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true