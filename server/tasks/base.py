"""Base class for all On-Call Environment tasks."""

import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseTask(ABC):
    """Abstract base for a production incident task."""

    # ── Override these in subclasses ──────────────────────────────────────────
    task_id: str = ""
    alert: str = ""
    description: str = ""
    services_total: int = 1
    max_steps: int = 20

    # Source files the agent can read/write (filename -> buggy content)
    @property
    @abstractmethod
    def source_files(self) -> Dict[str, str]:
        """Return {filename: buggy_source_code}."""

    # Test files (filename -> test code) — agent cannot modify these
    @property
    @abstractmethod
    def test_files(self) -> Dict[str, str]:
        """Return {filename: test_code}."""

    # Pre-authored realistic logs per service
    @property
    @abstractmethod
    def logs(self) -> Dict[str, str]:
        """Return {service_name: log_text}."""

    # Pre-authored metrics per service
    @property
    @abstractmethod
    def metrics(self) -> Dict[str, str]:
        """Return {service_name: metrics_text}."""

    # ── Workspace management ───────────────────────────────────────────────────

    def setup_workspace(self, session_id: str) -> str:
        """
        Copy buggy source + test files to an isolated temp directory.
        Returns the workspace path.
        """
        workspace = os.path.join(tempfile.gettempdir(), f"oncall_{session_id}")
        os.makedirs(workspace, exist_ok=True)

        for filename, content in self.source_files.items():
            with open(os.path.join(workspace, filename), "w") as f:
                f.write(content)

        for filename, content in self.test_files.items():
            with open(os.path.join(workspace, filename), "w") as f:
                f.write(content)

        return workspace

    @staticmethod
    def cleanup_workspace(workspace: str) -> None:
        """Remove the workspace directory."""
        if workspace and os.path.exists(workspace):
            shutil.rmtree(workspace, ignore_errors=True)

    # ── Observation helpers ────────────────────────────────────────────────────

    @property
    def file_names(self) -> List[str]:
        """Source files available to the agent (tests are hidden from listing)."""
        return sorted(self.source_files.keys())

    def list_services_output(self, tests_passing: int) -> str:
        ratio = tests_passing / max(len(self.test_files_flat), 1)
        status = "HEALTHY" if ratio >= 1.0 else "DEGRADED"
        health = self._compute_health(tests_passing)
        lines = [
            f"=== Service Health Dashboard ===",
            f"Incident: {self.alert}",
            "",
            f"Service: {self.task_id.replace('fix_', '')}",
            f"  Status:       {status}",
            f"  Memory:       {health['memory_pct']}%",
            f"  Error rate:   {health['error_rate']}%",
            f"  P99 latency:  {health['p99_latency']}ms",
            "",
            f"Source files:  {', '.join(self.file_names)}",
            f"Tests:         {tests_passing}/{self.tests_total} passing",
        ]
        return "\n".join(lines)

    def get_logs(self, service_name: str) -> str:
        key = (service_name or "").strip()
        # Try exact match, then partial match
        if key in self.logs:
            return self.logs[key]
        for k, v in self.logs.items():
            if key.lower() in k.lower() or k.lower() in key.lower():
                return v
        available = ", ".join(self.logs.keys()) if self.logs else "none"
        return f"[ERROR] Service '{service_name}' not found. Available: {available}"

    def get_metrics(self, service_name: str, tests_passing: int) -> str:
        health = self._compute_health(tests_passing)
        key = (service_name or "").strip()
        base = self.metrics.get(key) or next(iter(self.metrics.values()), "")
        # Overlay live health values
        live = (
            f"\n=== Live Metrics (current) ===\n"
            f"  memory_usage_pct:  {health['memory_pct']}\n"
            f"  error_rate_pct:    {health['error_rate']}\n"
            f"  p99_latency_ms:    {health['p99_latency']}\n"
            f"  status:            {health['status']}\n"
        )
        return base + live

    def health_output(self, tests_passing: int) -> str:
        h = self._compute_health(tests_passing)
        ratio = tests_passing / max(self.tests_total, 1)
        return (
            f"=== System Health ===\n"
            f"  Tests passing:   {tests_passing}/{self.tests_total} ({ratio*100:.0f}%)\n"
            f"  Status:          {h['status']}\n"
            f"  Memory:          {h['memory_pct']}%\n"
            f"  Error rate:      {h['error_rate']}%\n"
            f"  P99 latency:     {h['p99_latency']}ms\n"
        )

    # ── Reward / grading ───────────────────────────────────────────────────────

    @property
    def tests_total(self) -> int:
        """Count actual test functions across all test files."""
        import re
        count = 0
        for content in self.test_files.values():
            count += len(re.findall(r'^def test_', content, re.MULTILINE))
        return max(count, 1)

    @property
    def test_files_flat(self) -> Dict[str, str]:
        """All test files (flat dict, same as test_files for single-file tasks)."""
        return self.test_files

    def grade(self, tests_passing: int) -> float:
        """Final grader: strictly in (0, 1) — evaluator rejects 0.0 and 1.0."""
        raw = tests_passing / max(self.tests_total, 1)
        return max(0.01, min(0.99, round(raw, 4)))

    # ── Internal ───────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def _health_profile(self) -> Dict[str, float]:
        """Return {'mem': base_mem_pct, 'err': base_err_pct, 'lat': base_lat_ms}."""

    def _compute_health(self, tests_passing: int) -> Dict[str, float]:
        ratio = tests_passing / max(self.tests_total, 1)
        p = self._health_profile
        return {
            "memory_pct": round(p["mem"] * (1 - ratio * 0.7), 1),
            "error_rate": round(p["err"] * (1 - ratio), 1),
            "p99_latency": round(p["lat"] * (1 - ratio * 0.85), 0),
            "status": "HEALTHY" if ratio >= 1.0 else "DEGRADED",
        }
