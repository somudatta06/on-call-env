"""
On-Call Environment Client.

Persistent WebSocket client for multi-step interaction with the server.

Example:
    >>> import asyncio
    >>> from on_call_env import OnCallEnv, OnCallAction
    >>>
    >>> async def main():
    ...     async with OnCallEnv(base_url="http://localhost:8000") as env:
    ...         result = await env.reset(task_id="fix_memory_leak")
    ...         print(result.observation.alert)
    ...
    ...         action = OnCallAction(action_type="list_services")
    ...         result = await env.step(action)
    ...         print(result.observation.last_action_result)
    >>>
    >>> asyncio.run(main())
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import OnCallAction, OnCallObservation
except ImportError:
    from models import OnCallAction, OnCallObservation


class OnCallEnv(EnvClient[OnCallAction, OnCallObservation, State]):
    """
    WebSocket client for the On-Call Production Incident Environment.

    Maintains a persistent connection for efficient multi-step episodes.
    Use as an async context manager or call close() when done.
    """

    def _step_payload(self, action: OnCallAction) -> Dict:
        """Convert OnCallAction to JSON payload for the server step message."""
        payload: Dict = {"action_type": action.action_type}
        if action.service_name is not None:
            payload["service_name"] = action.service_name
        if action.filename is not None:
            payload["filename"] = action.filename
        if action.content is not None:
            payload["content"] = action.content
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[OnCallObservation]:
        """Parse server response into StepResult[OnCallObservation]."""
        obs_data = payload.get("observation", {})
        observation = OnCallObservation(
            alert=obs_data.get("alert", ""),
            task_description=obs_data.get("task_description", ""),
            last_action_result=obs_data.get("last_action_result", ""),
            tests_passing=obs_data.get("tests_passing", 0),
            tests_total=obs_data.get("tests_total", 0),
            services_healthy=obs_data.get("services_healthy", 0),
            services_total=obs_data.get("services_total", 1),
            files_in_workspace=obs_data.get("files_in_workspace", []),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
