"""
OnCallEnvironment — Production Incident Response RL Environment.

An agent acts as an on-call software engineer: receives a PagerDuty-style
alert, investigates logs/metrics/source code, applies fixes, and verifies
system health is restored via pytest.

Sessions are fully isolated in per-session /tmp directories.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OnCallAction, OnCallObservation
    from ..server.tasks import TASK_REGISTRY
    from ..server.executor import Executor
except ImportError:
    from models import OnCallAction, OnCallObservation
    from server.tasks import TASK_REGISTRY
    from server.executor import Executor


# ── Reward computation ─────────────────────────────────────────────────────────
# The grader computes a task score from step rewards. Each individual reward
# AND the cumulative sum must stay strictly in (0, 1). With max 40 steps we
# budget:  intermediate = 0.01 each  →  max 39 × 0.01 = 0.39
#          terminal     ≤ 0.59       →  total ≤ 0.98  < 1.0
# All values safe at 2-decimal formatting (never "0.00" or "1.00").

_INTERMEDIATE_REWARD = 0.01          # safe: formats as "0.01"
_MAX_TERMINAL_REWARD = 0.59          # 0.39 + 0.59 = 0.98 < 1.0
_MIN_TERMINAL_REWARD = 0.02          # formats as "0.02"


def _compute_reward(
    prev_passing: int,
    curr_passing: int,
    tests_total: int,
    step_count: int,
    max_steps: int,
    new_file: bool,
    done: bool,
) -> float:
    """
    Reward per step.
    - Non-terminal steps: fixed 0.01
    - Terminal step (done=True): proportional to test pass rate, in [0.02, 0.59]

    Guarantees:
    - Each individual reward ∈ (0, 1)   (actually [0.01, 0.59])
    - Cumulative sum over episode ∈ (0, 1) (max ≈ 0.98)
    - 2-decimal format never produces "0.00" or "1.00"
    """
    if not done:
        return _INTERMEDIATE_REWARD

    # Terminal step — score proportional to how many tests pass
    if tests_total == 0:
        return _MIN_TERMINAL_REWARD

    ratio = curr_passing / tests_total  # 0.0 to 1.0
    # Scale into [_MIN_TERMINAL_REWARD, _MAX_TERMINAL_REWARD]
    reward = _MIN_TERMINAL_REWARD + ratio * (_MAX_TERMINAL_REWARD - _MIN_TERMINAL_REWARD)
    return round(reward, 4)  # e.g. 0.59, 0.31, 0.02


# ── Environment ────────────────────────────────────────────────────────────────

class OnCallEnvironment(Environment):
    """
    Production Incident Response environment.

    Each instance represents one isolated session. Concurrent sessions are
    fully supported — each gets its own /tmp workspace.
    """

    # Required for multi-session WebSocket support
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task = None
        self._workspace: Optional[str] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._tests_passing: int = 0
        self._files_read: set = set()
        self._done: bool = False

    # ── Public properties (used by Executor) ──────────────────────────────────

    @property
    def task(self):
        return self._task

    @property
    def workspace(self) -> str:
        return self._workspace or ""

    @property
    def tests_passing(self) -> int:
        return self._tests_passing

    @tests_passing.setter
    def tests_passing(self, value: int) -> None:
        self._tests_passing = value

    @property
    def files_read(self) -> set:
        return self._files_read

    # ── Environment interface ──────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> OnCallObservation:
        """
        Reset the environment for a new incident episode.

        Args:
            task_id: Which incident to investigate. One of:
                     "fix_memory_leak" | "fix_api_gateway" | "fix_payment_service"
                     Defaults to "fix_memory_leak".
            episode_id: Optional episode identifier.
            seed: Ignored (tasks are deterministic).
        """
        # Clean up previous workspace
        if self._workspace and self._task:
            self._task.cleanup_workspace(self._workspace)

        task_id = kwargs.get("task_id", "fix_memory_leak")
        if task_id not in TASK_REGISTRY:
            task_id = "fix_memory_leak"

        task_class = TASK_REGISTRY[task_id]
        self._task = task_class()
        self._episode_id = episode_id or str(uuid.uuid4())
        self._workspace = self._task.setup_workspace(self._episode_id)
        self._step_count = 0
        self._tests_passing = 0
        self._files_read = set()
        self._done = False

        return OnCallObservation(
            alert=self._task.alert,
            task_description=self._task.description,
            last_action_result=(
                "=== Incident Alert Received ===\n"
                f"{self._task.alert}\n\n"
                "Begin investigation: use list_services to see status, "
                "read_logs/read_metrics for context, read_file to inspect source, "
                "write_file to apply fix, run_tests to verify."
            ),
            tests_passing=0,
            tests_total=self._task.tests_total,
            services_healthy=0,
            services_total=self._task.services_total,
            files_in_workspace=self._task.file_names,
            step_count=0,
            done=False,
            reward=0.01,
        )

    def step(
        self,
        action: OnCallAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OnCallObservation:
        """Execute an action and return the resulting observation."""
        if self._task is None:
            return OnCallObservation(
                last_action_result="[ERROR] Call reset() first to start an episode.",
                done=True,
                reward=0.01,
            )

        if self._done:
            return self._build_obs(
                "Episode already done. Call reset() to start a new episode.",
                reward=0.01,
            )

        prev_passing = self._tests_passing

        # Dispatch action
        result_text, new_file = Executor.execute(action, self)

        # Compute reward
        self._step_count += 1
        is_complete = (self._tests_passing == self._task.tests_total)
        at_limit = (self._step_count >= self._task.max_steps)

        reward = _compute_reward(
            prev_passing=prev_passing,
            curr_passing=self._tests_passing,
            tests_total=self._task.tests_total,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            new_file=new_file,
            done=is_complete,
        )

        if is_complete or at_limit:
            self._done = True

        return self._build_obs(result_text, reward=reward)

    @property
    def state(self) -> State:
        """Return current episode state."""
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
        )

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="on-call-env",
            description=(
                "Production incident response environment: investigate alerts, "
                "read logs/metrics, debug source code, apply hotfixes, "
                "and verify system health is restored."
            ),
            version="0.1.0",
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_obs(self, result_text: str, reward: float) -> OnCallObservation:
        """Build an observation from current state."""
        tests_total = self._task.tests_total if self._task else 0
        services_total = self._task.services_total if self._task else 1
        file_names = self._task.file_names if self._task else []

        services_healthy = 0
        if tests_total > 0:
            ratio = self._tests_passing / tests_total
            services_healthy = round(ratio * services_total)

        return OnCallObservation(
            alert=self._task.alert if self._task else "",
            task_description=self._task.description if self._task else "",
            last_action_result=result_text,
            tests_passing=self._tests_passing,
            tests_total=tests_total,
            services_healthy=services_healthy,
            services_total=services_total,
            files_in_workspace=file_names,
            step_count=self._step_count,
            done=self._done,
            reward=reward,
        )
