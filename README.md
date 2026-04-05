---
title: on-call-env
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# on-call-env

**Production Incident Response RL Environment**

An agent acts as an on-call software engineer responding to real production incidents. It must:
1. Read PagerDuty-style alerts, logs, and metrics to understand the incident
2. Investigate source code to identify the bug
3. Apply a hotfix via `write_file`
4. Verify the fix with `run_tests`
5. Confirm system health is restored

## Tasks

| Task | Severity | Bugs | Files | Tests |
|------|----------|------|-------|-------|
| `fix_memory_leak` | P3 Easy | 1 | 1 | 5 |
| `fix_api_gateway` | P2 Medium | 2 | 2 | 11 |
| `fix_payment_service` | P1 Hard | 3 | 3 | 20 |

## Actions

| Action | Description |
|--------|-------------|
| `list_services` | Show all services with live health status and test progress |
| `read_logs` | Get pre-authored production logs for a service |
| `read_metrics` | Get live metrics (memory, error rate, latency) |
| `read_file` | Read source file content from workspace |
| `write_file` | Write fixed source file (full content, not a diff) |
| `run_tests` | Run pytest and see pass/fail breakdown |
| `check_health` | Summarise system health derived from test state |

## Quick Start

```bash
# Install dependencies
uv sync

# Start the server
uv run server

# Run baseline inference agent
API_BASE_URL="https://router.huggingface.co/v1" \
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
HF_TOKEN="your_token_here" \
python inference.py
```

## Reward Design

- **Dense progress**: `(tests_passing_delta) / tests_total` — always informative
- **Exploration bonus**: `+0.03` for first read of each source file — rewards investigation
- **Step penalty**: `-0.008` per step — encourages efficiency
- **Terminal bonus**: `+1.0` when all tests pass, with efficiency multiplier for speed

## Environment API

```python
from on_call_env import OnCallEnv, OnCallAction

async with OnCallEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task_id="fix_memory_leak")
    print(result.observation.alert)

    result = await env.step(OnCallAction(action_type="read_file", filename="cache_manager.py"))
    print(result.observation.last_action_result)
```
