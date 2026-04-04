"""
on-call-env: Production Incident Response RL Environment

An agent acts as an on-call software engineer:
  1. Receives a PagerDuty-style P1/P2/P3 alert
  2. Investigates using read_logs, read_metrics, read_file
  3. Identifies bugs in source code
  4. Applies fixes via write_file
  5. Verifies system health with run_tests

Tasks:
  fix_memory_leak      — P3 Easy:   1 bug, 1 file,  5 tests
  fix_api_gateway      — P2 Medium: 2 bugs, 2 files, 11 tests
  fix_payment_service  — P1 Hard:   3 bugs, 3 files, 20 tests
"""

from .models import OnCallAction, OnCallObservation
from .client import OnCallEnv

__all__ = [
    "OnCallAction",
    "OnCallObservation",
    "OnCallEnv",
]
