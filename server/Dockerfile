FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency specs first (layer cache)
COPY pyproject.toml ./
COPY README.md ./

# Create minimal package stub so pip can install in editable mode
RUN mkdir -p on_call_env/server/tasks \
    && touch on_call_env/__init__.py \
    && touch on_call_env/server/__init__.py \
    && touch on_call_env/server/tasks/__init__.py

# Install dependencies (openenv-core + pytest + openai)
RUN uv pip install --system -e ".[dev]"

# Copy full source
COPY . .

# Non-root user for security
RUN useradd -m -u 1000 agent && chown -R agent /app
USER agent

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "on_call_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
