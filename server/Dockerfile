FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency manifest first (layer cache)
COPY pyproject.toml README.md ./

# Stub package directories so editable install resolves correctly
RUN mkdir -p on_call_env/server/tasks \
    && touch on_call_env/__init__.py \
          on_call_env/server/__init__.py \
          on_call_env/server/tasks/__init__.py

# Install all dependencies
RUN uv pip install --system -e ".[dev]"

# Copy full source (overwrites stubs with real files)
COPY . .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "on_call_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860"]
