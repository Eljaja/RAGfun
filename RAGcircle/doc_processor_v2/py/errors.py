from __future__ import annotations

import traceback as _tb


class NonRetryableError(Exception):
    """
    Raised for errors that should go straight to the DLQ — retrying will never help.

    Examples: bad payload, auth failure, schema mismatch, malformed filename.

    Carries the original exception and its traceback so DLQ consumers
    can inspect the root cause without re-running the pipeline.
    """

    def __init__(self, reason: str, *, cause: Exception | None = None):
        super().__init__(reason)
        self.reason = reason
        self.original_error = cause
        if cause is not None:
            self.__cause__ = cause
            self.original_traceback: str | None = "".join(
                _tb.format_exception(type(cause), cause, cause.__traceback__)
            )
        else:
            self.original_traceback = None

