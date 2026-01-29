from __future__ import annotations


class NonRetryableError(Exception):
    """
    Raised for inputs that should go straight to the DLQ (bad payload, unsupported event shape, etc.).
    """

