"""Task registry for the On-Call Environment."""

from .easy import MemoryLeakTask
from .medium import ApiGatewayTask
from .hard import PaymentServiceTask

TASK_REGISTRY = {
    "fix_memory_leak": MemoryLeakTask,
    "fix_api_gateway": ApiGatewayTask,
    "fix_payment_service": PaymentServiceTask,
}

__all__ = ["TASK_REGISTRY", "MemoryLeakTask", "ApiGatewayTask", "PaymentServiceTask"]
