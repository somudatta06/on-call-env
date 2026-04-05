FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency manifest first (layer cache)
COPY pyproject.toml README.md ./

# Stub package layout matching pyproject.toml package-dir mappings:
#   "on_call_env"              = "."          → /app/__init__.py
#   "on_call_env.server"       = "server"     → /app/server/__init__.py
#   "on_call_env.server.tasks" = "server/tasks" → /app/server/tasks/__init__.py
RUN mkdir -p server/tasks \
    && touch __init__.py \
             server/__init__.py \
             server/tasks/__init__.py

# Install all dependencies (uses stubs so editable install resolves correctly)
RUN uv pip install --system -e ".[dev]"

# Copy full source (overwrites stubs with real files)
COPY . .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "on_call_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]
