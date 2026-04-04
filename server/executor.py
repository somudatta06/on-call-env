"""
Executor — dispatches OnCallActions and runs pytest in isolated workspaces.

Handles all 7 action types:
  list_services, read_logs, read_metrics, read_file, write_file,
  run_tests, check_health
"""

from __future__ import annotations

import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .on_call_env_environment import OnCallEnvironment
from ..models import OnCallAction


# ── Test result parser ─────────────────────────────────────────────────────────

@dataclass
class TestResult:
    passing: int
    failing: int
    total: int
    output: str

    def format_output(self) -> str:
        status = "✓ ALL TESTS PASS" if self.failing == 0 else f"✗ {self.failing} FAILING"
        return (
            f"=== Test Results ===\n"
            f"  {status}\n"
            f"  Passing: {self.passing}/{self.total}\n"
            f"  Failing: {self.failing}/{self.total}\n\n"
            f"--- pytest output ---\n"
            f"{self.output[:4000]}"
        )

    @classmethod
    def parse(cls, output: str) -> "TestResult":
        """Parse pytest stdout/stderr to extract pass/fail counts."""
        # Try to match summary line: "2 passed, 1 failed" or "5 passed"
        # e.g. "3 failed, 2 passed in 0.12s"
        passing = 0
        failing = 0

        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        error_match  = re.search(r"(\d+) error", output)

        if passed_match:
            passing = int(passed_match.group(1))
        if failed_match:
            failing = int(failed_match.group(1))
        if error_match:
            failing += int(error_match.group(1))

        # If pytest reports "no tests ran" treat as 0/0
        total = passing + failing
        return cls(passing=passing, failing=failing, total=total, output=output)


# ── Path safety ────────────────────────────────────────────────────────────────

def _safe_path(workspace: str, filename: str) -> Path:
    """Resolve path and block traversal outside workspace."""
    resolved = (Path(workspace) / filename).resolve()
    workspace_resolved = Path(workspace).resolve()
    if not str(resolved).startswith(str(workspace_resolved) + "/") and \
       resolved != workspace_resolved:
        raise ValueError(f"Path traversal attempt blocked: {filename!r}")
    return resolved


# ── Executor ───────────────────────────────────────────────────────────────────

class Executor:
    """Stateless action dispatcher for the On-Call environment."""

    @staticmethod
    def execute(action: OnCallAction, env: "OnCallEnvironment") -> Tuple[str, bool]:
        """
        Dispatch action, return (result_text, new_file_explored).

        new_file_explored is True the FIRST time a source file is read —
        used for the exploration reward bonus.
        """
        atype = (action.action_type or "").strip().lower()

        if atype == "list_services":
            return env.task.list_services_output(env.tests_passing), False

        elif atype == "read_logs":
            name = action.service_name or ""
            return env.task.get_logs(name), False

        elif atype == "read_metrics":
            name = action.service_name or ""
            return env.task.get_metrics(name, env.tests_passing), False

        elif atype == "read_file":
            fname = action.filename or ""
            return Executor._read_file(env, fname)

        elif atype == "write_file":
            fname = action.filename or ""
            content = action.content or ""
            return Executor._write_file(env, fname, content)

        elif atype == "run_tests":
            tr = Executor.run_tests(env.workspace)
            env.tests_passing = tr.passing
            return tr.format_output(), False

        elif atype == "check_health":
            return env.task.health_output(env.tests_passing), False

        else:
            valid = "list_services | read_logs | read_metrics | read_file | write_file | run_tests | check_health"
            return f"[ERROR] Unknown action_type: {action.action_type!r}. Valid: {valid}", False

    # ── File operations ────────────────────────────────────────────────────────

    @staticmethod
    def _read_file(env: "OnCallEnvironment", filename: str) -> Tuple[str, bool]:
        if not filename:
            available = ", ".join(env.task.file_names)
            return f"[ERROR] No filename given. Available files: {available}", False

        try:
            path = _safe_path(env.workspace, filename)
        except ValueError as e:
            return f"[ERROR] {e}", False

        if not path.exists():
            available = ", ".join(env.task.file_names)
            return (
                f"[ERROR] File '{filename}' not found in workspace.\n"
                f"Available source files: {available}"
            ), False

        content = path.read_text(encoding="utf-8")
        new_file = filename not in env.files_read
        env.files_read.add(filename)
        return content, new_file

    @staticmethod
    def _write_file(env: "OnCallEnvironment", filename: str, content: str) -> Tuple[str, bool]:
        if not filename:
            return "[ERROR] No filename given for write_file.", False

        # Only allow writing to known source files (not test files)
        if filename not in env.task.source_files:
            writable = ", ".join(env.task.file_names)
            return (
                f"[ERROR] '{filename}' is not a writable source file.\n"
                f"Writable files: {writable}"
            ), False

        if not content:
            return f"[ERROR] No content provided for write_file.", False

        try:
            path = _safe_path(env.workspace, filename)
        except ValueError as e:
            return f"[ERROR] {e}", False

        path.write_text(content, encoding="utf-8")
        return (
            f"[OK] Written '{filename}' ({len(content)} bytes).\n"
            f"Run 'run_tests' to verify your fix."
        ), False

    # ── Test runner ────────────────────────────────────────────────────────────

    @staticmethod
    def run_tests(workspace: str, timeout: int = 30) -> TestResult:
        """Run pytest in the workspace, return parsed results."""
        try:
            result = subprocess.run(
                [
                    "python3", "-m", "pytest", ".",
                    "-q", "--tb=short", "--no-header",
                    "--color=no",
                ],
                capture_output=True,
                text=True,
                cwd=workspace,
                timeout=timeout,
            )
            combined = result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired:
            combined = f"[ERROR] pytest timed out after {timeout}s"
        except Exception as e:
            combined = f"[ERROR] Failed to run pytest: {e}"

        return TestResult.parse(combined)
