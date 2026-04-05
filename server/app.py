"""
FastAPI application for the On-Call Environment.

Exposes HTTP and WebSocket endpoints that EnvClient can connect to:
  POST /reset   — start a new incident episode
  POST /step    — take an action
  GET  /state   — current episode state
  GET  /schema  — action/observation/state JSON schemas
  GET  /health  — health check
  GET  /metadata — environment metadata
  WS   /ws      — persistent WebSocket session for multi-step episodes

Usage (development):
    uvicorn on_call_env.server.app:app --reload --host 0.0.0.0 --port 8000

Usage (production):
    uv run server

Or directly:
    python -m on_call_env.server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import OnCallAction, OnCallObservation
    from .on_call_env_environment import OnCallEnvironment
except ModuleNotFoundError:
    from models import OnCallAction, OnCallObservation
    from server.on_call_env_environment import OnCallEnvironment


# Create the FastAPI application.
# OnCallEnvironment is a factory callable (class itself) — each WebSocket session
# gets its own isolated environment instance.
app = create_app(
    OnCallEnvironment,          # factory: called to create a new env per session
    OnCallAction,
    OnCallObservation,
    env_name="on-call-env",
    max_concurrent_envs=10,     # support up to 10 concurrent training workers
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """
    Entry point for direct execution.

    Enables:
        uv run server
        python -m on_call_env.server.app
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="On-Call Environment Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main()  # calls main() with defaults; override via CLI args above if needed
