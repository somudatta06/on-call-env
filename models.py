"""
Data models for the On-Call Environment.

An agent acts as an on-call software engineer responding to production incidents.
It investigates alerts, reads logs/metrics, debugs source code, applies fixes,
and verifies system health is restored.
"""

from typing import List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class OnCallAction(Action):
    """Action an on-call engineer can take to investigate and fix an incident."""

    action_type: str = Field(
        ...,
        description=(
            "One of: list_services | read_logs | read_metrics | "
            "read_file | write_file | run_tests | check_health"
        ),
    )
    service_name: Optional[str] = Field(
        default=None,
        description="Target service name (for read_logs, read_metrics)",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Source file name (for read_file, write_file)",
    )
    content: Optional[str] = Field(
        default=None,
        description="Full corrected file content (for write_file only)",
    )


class OnCallObservation(Observation):
    """Observation returned after each action in the on-call environment."""

    alert: str = Field(
        default="",
        description="PagerDuty-style incident alert that triggered this episode",
    )
    task_description: str = Field(
        default="",
        description="What the agent must accomplish to resolve the incident",
    )
    last_action_result: str = Field(
        default="",
        description="Output / result from the last action taken",
    )
    tests_passing: int = Field(
        default=0,
        description="Number of tests currently passing in the workspace",
    )
    tests_total: int = Field(
        default=0,
        description="Total number of tests in the task",
    )
    services_healthy: int = Field(
        default=0,
        description="Number of services currently healthy (derived from test state)",
    )
    services_total: int = Field(
        default=1,
        description="Total services in this task",
    )
    files_in_workspace: List[str] = Field(
        default_factory=list,
        description="List of source files available for reading/writing",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far in this episode",
    )
