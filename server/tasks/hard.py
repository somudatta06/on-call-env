"""
Task 3 (Hard) — fix_payment_service
P1 Alert: duplicate charges detected, $12k revenue discrepancy.

Three bugs across three files:

Bug 1 — payment_processor.py:
  Idempotency key includes `int(time.time())` (seconds-precision timestamp).
  A client retry within the same second generates the SAME key → deduplication works.
  But a retry 1-2 seconds later (common in real systems) generates a NEW key →
  the payment is processed twice, charging the customer twice.
  Fix: use `request_id` (a client-provided stable identifier) as the key.

Bug 2 — validator.py:
  `validate_amount()` uses Python `float(amount_str)`.  For amounts like "1000.10",
  float introduces a representation error: 1000.0999999999999.
  Downstream rounding then stores "1000.09" instead of "1000.10" — $0.01 per txn.
  Fix: (a) use `Decimal(amount_str)` directly (not float), (b) also catch
  `InvalidOperation` in the except clause (Decimal raises this, not ValueError),
  and (c) return the Decimal directly (not via float→str→Decimal round-trip).

Bug 3 — ledger.py:
  `record_transaction()` silently truncates merchant names to 50 chars with `[:50]`.
  When the caller later calls `get_transactions("Full Long Merchant Name...")` with
  the original (untruncated) name, the lookup returns [] — the entry is effectively
  orphaned.
  Fix: raise ValueError if merchant_name exceeds 50 chars instead of silently truncating.
"""

from typing import Dict
from .base import BaseTask


# ── Buggy source files ─────────────────────────────────────────────────────────

_PAYMENT_PROCESSOR_BUGGY = '''\
"""Payment processor with idempotency protection."""
import time
from typing import Optional


class PaymentProcessor:
    """Processes payments with idempotency to prevent duplicate charges."""

    def __init__(self):
        self._processed: dict = {}   # idempotency_key -> result

    def process_payment(
        self,
        order_id: str,
        amount: str,
        merchant: str,
        request_id: str,
    ) -> dict:
        """
        Process a payment.  Returns existing result if already processed (idempotent).

        Args:
            order_id:   Unique order identifier.
            amount:     Payment amount as string (e.g. "99.99").
            merchant:   Merchant name.
            request_id: Client-provided idempotency token — must be stable across retries.

        Returns:
            dict with keys: order_id, amount, merchant, status, transaction_id, is_duplicate
        """
        # BUG: idempotency key uses int(time.time()) — changes every second.
        # A retry 2 seconds later creates a NEW key and processes the payment AGAIN.
        idempotency_key = f"{order_id}_{int(time.time())}"   # BUG
        # FIX: idempotency_key = f"{order_id}_{request_id}"

        if idempotency_key in self._processed:
            return {**self._processed[idempotency_key], "is_duplicate": True}

        result = {
            "order_id": order_id,
            "amount": amount,
            "merchant": merchant,
            "status": "success",
            "transaction_id": f"txn_{order_id}_{request_id[:8]}",
            "is_duplicate": False,
        }
        self._processed[idempotency_key] = result
        return result
'''

_VALIDATOR_BUGGY = '''\
"""Input validation for payment fields."""
from decimal import Decimal, InvalidOperation
from typing import Union


def validate_amount(amount_str: str) -> Union[Decimal, str]:
    """
    Validate and parse a payment amount string.

    Returns the parsed amount as a Decimal, or the string "invalid_amount" on error.
    """
    try:
        # BUG: float() introduces representation errors.
        # "1000.10" becomes 1000.0999999999999... causing $0.01 discrepancies.
        amount = float(amount_str)   # BUG — should be Decimal(amount_str)
    except (ValueError, TypeError):
        return "invalid_amount"

    if amount <= 0:
        return "invalid_amount"

    # Return as Decimal for downstream processing, but precision is already lost
    return Decimal(str(amount))   # converting float→str→Decimal does NOT recover precision


def validate_merchant_name(name: str) -> bool:
    """Return True if merchant name is valid (non-empty, ≤ 100 chars)."""
    return bool(name) and len(name) <= 100
'''

_LEDGER_BUGGY = '''\
"""Transaction ledger — records and retrieves payment history."""


class Ledger:
    """In-memory transaction ledger keyed by merchant name."""

    def __init__(self):
        self._entries: dict = {}   # merchant_name -> list[dict]

    def record_transaction(
        self,
        merchant_name: str,
        amount: str,
        transaction_id: str,
    ) -> None:
        """
        Record a completed transaction.

        Args:
            merchant_name:  Merchant name (must be ≤ 50 chars — enforced by DB schema).
            amount:         Payment amount as string.
            transaction_id: Unique transaction identifier.
        """
        # BUG: silently truncates merchant_name to 50 chars.
        # If the caller later calls get_transactions() with the full (>50 char) name,
        # the lookup returns [] because the stored key is different.
        stored_name = merchant_name[:50]   # BUG — should raise ValueError if too long
        # FIX:
        # if len(merchant_name) > 50:
        #     raise ValueError(
        #         f"Merchant name too long: {len(merchant_name)} chars (max 50). "
        #         f"Name: {merchant_name!r}"
        #     )
        # stored_name = merchant_name

        if stored_name not in self._entries:
            self._entries[stored_name] = []

        self._entries[stored_name].append({
            "transaction_id": transaction_id,
            "amount": amount,
        })

    def get_transactions(self, merchant_name: str) -> list:
        """Return all transactions for the given merchant (exact name match)."""
        return self._entries.get(merchant_name, [])

    def total_revenue(self, merchant_name: str) -> str:
        """Return total revenue for a merchant as a formatted string."""
        from decimal import Decimal
        txns = self.get_transactions(merchant_name)
        if not txns:
            return "0.00"
        total = sum(Decimal(t["amount"]) for t in txns)
        return str(total)
'''

# ── Tests ──────────────────────────────────────────────────────────────────────

_TEST_PAYMENT = '''\
"""Tests for payment_processor, validator, and ledger."""
import time
import pytest
from decimal import Decimal
from payment_processor import PaymentProcessor
from validator import validate_amount, validate_merchant_name
from ledger import Ledger


# ── PaymentProcessor tests ─────────────────────────────────────────────────────

def test_payment_processes_successfully():
    """A valid payment should return status=success."""
    pp = PaymentProcessor()
    result = pp.process_payment("ORD_001", "99.99", "ShopCo", "req_abc123")
    assert result["status"] == "success"
    assert result["order_id"] == "ORD_001"


def test_idempotency_same_request_id():
    """Same request_id → second call must return duplicate=True without reprocessing."""
    pp = PaymentProcessor()
    r1 = pp.process_payment("ORD_002", "50.00", "ShopCo", "req_stable")
    r2 = pp.process_payment("ORD_002", "50.00", "ShopCo", "req_stable")
    assert r1["is_duplicate"] is False
    assert r2["is_duplicate"] is True


def test_idempotency_prevents_duplicate_charge():
    """Retrying with the same request_id must NOT create a second transaction."""
    pp = PaymentProcessor()
    pp.process_payment("ORD_003", "100.00", "ShopCo", "req_retry")
    # Second attempt (simulating network retry)
    result = pp.process_payment("ORD_003", "100.00", "ShopCo", "req_retry")
    assert result["is_duplicate"] is True, (
        "Retry with same request_id must be detected as duplicate"
    )


def test_idempotency_stable_across_time():
    """Idempotency must hold even if time.time() has advanced by >1 second."""
    pp = PaymentProcessor()
    r1 = pp.process_payment("ORD_004", "75.00", "ShopCo", "req_time_stable")
    # Simulate a 2-second delay between original request and retry
    # by calling process_payment again — the key must still be based on request_id
    r2 = pp.process_payment("ORD_004", "75.00", "ShopCo", "req_time_stable")
    assert r2["is_duplicate"] is True, (
        "Idempotency must not depend on time.time() — must use request_id"
    )


def test_different_request_ids_are_different_payments():
    """Different request_ids for the same order → treated as separate payments."""
    pp = PaymentProcessor()
    r1 = pp.process_payment("ORD_005", "20.00", "ShopCo", "req_001")
    r2 = pp.process_payment("ORD_005", "20.00", "ShopCo", "req_002")
    assert r1["is_duplicate"] is False
    assert r2["is_duplicate"] is False


# ── Validator tests ────────────────────────────────────────────────────────────

def test_validate_simple_amount():
    """Simple amounts must parse correctly."""
    result = validate_amount("99.99")
    assert result == Decimal("99.99"), f"Expected Decimal('99.99'), got {result!r}"


def test_validate_amount_precision_large():
    """Amounts ≥ 1000 with decimal cents must have exact precision."""
    result = validate_amount("1000.10")
    assert result == Decimal("1000.10"), (
        f"Expected Decimal('1000.10'), got {result!r}  "
        f"(hint: float('1000.10') = {float('1000.10')})"
    )


def test_validate_amount_precision_edge():
    """Edge case: 9999.99 must not lose precision."""
    result = validate_amount("9999.99")
    assert result == Decimal("9999.99"), f"Got {result!r}"


def test_validate_amount_returns_decimal_type():
    """validate_amount must return a Decimal, not a float."""
    result = validate_amount("42.50")
    assert isinstance(result, Decimal), f"Expected Decimal, got {type(result).__name__}"


def test_validate_invalid_amount():
    """Non-numeric strings must return 'invalid_amount'."""
    assert validate_amount("not_a_number") == "invalid_amount"
    assert validate_amount("-5.00") == "invalid_amount"
    assert validate_amount("0") == "invalid_amount"


# ── Ledger tests ───────────────────────────────────────────────────────────────

def test_ledger_record_and_retrieve_short_name():
    """Short merchant names (≤50 chars) must be stored and retrieved correctly."""
    ledger = Ledger()
    ledger.record_transaction("ShopCo", "50.00", "txn_001")
    txns = ledger.get_transactions("ShopCo")
    assert len(txns) == 1
    assert txns[0]["transaction_id"] == "txn_001"


def test_ledger_long_name_raises_error():
    """Merchant names > 50 chars must raise ValueError, not silently truncate."""
    ledger = Ledger()
    long_name = "A" * 51   # 51 chars — exceeds 50-char DB schema limit
    with pytest.raises(ValueError, match="too long"):
        ledger.record_transaction(long_name, "25.00", "txn_002")


def test_ledger_lookup_with_exact_name():
    """get_transactions() must find entries when queried with the exact stored name."""
    ledger = Ledger()
    name = "Exactly Fifty Characters Long Name Here!!!!"  # 44 chars — within limit
    ledger.record_transaction(name, "10.00", "txn_003")
    result = ledger.get_transactions(name)
    assert len(result) == 1, "Should find exactly 1 transaction"


def test_ledger_multiple_transactions_same_merchant():
    """Multiple transactions for the same merchant must all be recorded."""
    ledger = Ledger()
    for i in range(3):
        ledger.record_transaction("MerchantX", f"{10 * (i+1)}.00", f"txn_{i}")
    txns = ledger.get_transactions("MerchantX")
    assert len(txns) == 3


def test_full_payment_flow():
    """End-to-end: validate → process → record → retrieve."""
    pp = PaymentProcessor()
    ledger = Ledger()

    amount_str = "1500.50"
    merchant = "Premium Store Ltd"

    # Validate
    amount = validate_amount(amount_str)
    assert isinstance(amount, Decimal)
    assert amount == Decimal("1500.50")

    # Process
    result = pp.process_payment("ORD_999", amount_str, merchant, "req_e2e_001")
    assert result["status"] == "success"

    # Record
    ledger.record_transaction(merchant, str(amount), result["transaction_id"])

    # Retrieve
    txns = ledger.get_transactions(merchant)
    assert len(txns) == 1
    assert txns[0]["transaction_id"] == result["transaction_id"]
    assert Decimal(txns[0]["amount"]) == Decimal("1500.50")
'''

# ── Pre-authored logs / metrics ────────────────────────────────────────────────

_LOGS = """\
=== payment-service logs (last 50 lines) ===
2026-04-04 16:02:00 INFO  payment  - PaymentProcessor started
2026-04-04 16:02:11 ERROR payment  - Duplicate charge detected! order_id=ORD_8821
2026-04-04 16:02:11 ERROR payment  - idempotency_key ORD_8821_1743782530 != ORD_8821_1743782532
2026-04-04 16:02:11 ERROR payment  - Customer charged TWICE for ORD_8821 ($249.99 each)
2026-04-04 16:02:15 WARN  validator - Amount precision warning: float('1000.10')=1000.0999999999999
2026-04-04 16:02:15 WARN  validator - Stored 1000.09 instead of 1000.10 for txn_8822
2026-04-04 16:02:20 ERROR ledger   - FK-like constraint violation: merchant lookup failed
2026-04-04 16:02:20 ERROR ledger   - Stored key: 'Super Long Merchant Name That Exceeds F'
2026-04-04 16:02:20 ERROR ledger   - Query key:  'Super Long Merchant Name That Exceeds Fifty Characters'
2026-04-04 16:02:20 ERROR ledger   - get_transactions() returned [] — orphaned record
2026-04-04 16:02:25 CRIT  payment  - Revenue discrepancy: $12,040.00 in last hour
2026-04-04 16:02:25 CRIT  payment  - PagerDuty P1 alert fired: DUPLICATE_CHARGES
"""

_METRICS = """\
=== payment-service metrics (last 5 min) ===
  transactions_per_sec:  145
  duplicate_charges:     23    [CRITICAL — expected 0]
  amount_discrepancies:  18    [CRITICAL — expected 0]
  orphaned_records:      7     [CRITICAL — expected 0]
  error_rate_pct:        8.0
  p99_latency_ms:        500
  memory_usage_pct:      45.0
  revenue_discrepancy:   $12,040.00
"""


class PaymentServiceTask(BaseTask):
    task_id = "fix_payment_service"
    alert = (
        "[P1] CRITICAL — payment-service: duplicate charges detected, "
        "$12k revenue discrepancy in last hour"
    )
    description = (
        "The payment service has three bugs causing duplicate charges, precision loss, "
        "and orphaned ledger records. Read logs and all three source files to identify "
        "each bug, fix them, and verify all 15 tests pass."
    )
    services_total = 3
    max_steps = 40

    @property
    def source_files(self) -> Dict[str, str]:
        return {
            "payment_processor.py": _PAYMENT_PROCESSOR_BUGGY,
            "validator.py": _VALIDATOR_BUGGY,
            "ledger.py": _LEDGER_BUGGY,
        }

    @property
    def test_files(self) -> Dict[str, str]:
        return {"test_payment.py": _TEST_PAYMENT}

    @property
    def logs(self) -> Dict[str, str]:
        return {
            "payment-service": _LOGS,
            "payment_service": _LOGS,
            "payment": _LOGS,
            "ledger": _LOGS,
            "validator": _LOGS,
        }

    @property
    def metrics(self) -> Dict[str, str]:
        return {
            "payment-service": _METRICS,
            "payment_service": _METRICS,
            "payment": _METRICS,
        }

    @property
    def _health_profile(self) -> Dict[str, float]:
        return {"mem": 45.0, "err": 8.0, "lat": 500.0}
