"""Custom exceptions and handlers for the presign API"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# ----------------------------
# Custom Exceptions
# ----------------------------

class S3OperationError(Exception):
    """Base exception for S3 operations"""
    def __init__(self, message: str, operation: str, status_code: int = 500):
        self.message = message
        self.operation = operation
        self.status_code = status_code
        super().__init__(self.message)


class BucketAlreadyExistsError(Exception):
    """Raised when attempting to create an existing bucket"""
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        super().__init__(f"Bucket {bucket_name} already exists")


class BucketNotFoundError(Exception):
    """Raised when bucket doesn't exist"""
    def __init__(self, bucket_name: str, ):
        self.bucket_name = bucket_name
        super().__init__(f"Bucket {bucket_name} not found")


class InvalidKeyError(Exception):
    """Raised for invalid S3 keys"""
    def __init__(self, key: str, reason: str):
        self.key = key
        self.reason = reason
        super().__init__(f"Invalid key '{key}': {reason}")


# ----------------------------
# Exception Handlers
# ----------------------------

async def invalid_key_handler(request: Request, exc: InvalidKeyError):
    logger.warning(f"Invalid key attempted: {exc.key} - {exc.reason}", extra={"key": exc.key})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid object key", "detail": exc.reason}
    )


async def bucket_exists_handler(request: Request, exc: BucketAlreadyExistsError):
    logger.info(f"Bucket creation failed - already exists: {exc.bucket_name}")
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"error": "Bucket already exists", "bucket": exc.bucket_name}
    )


async def bucket_not_found_handler(request: Request, exc: BucketNotFoundError):
    logger.warning(f"Bucket not found: {exc.bucket_name}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Bucket not found", "bucket": exc.bucket_name}
    )


async def s3_operation_handler(request: Request, exc: S3OperationError):
    logger.error(
        f"S3 operation failed: {exc.operation} - {exc.message}",
        extra={"operation": exc.operation, "status_code": exc.status_code}
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "S3 operation failed", "detail": exc.message}
    )


# ----------------------------
# Error Handling Wrapper
# ----------------------------

async def safe_s3_call(operation_name: str, coro):
    """
    Wrapper for S3 operations that extracts proper error codes and wraps exceptions.
    
    Usage:
        await safe_s3_call("create_bucket", s3_cli.create_bucket(Bucket="my-bucket"))
    """
    try:
        return await coro
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500)
        raise S3OperationError(
            f"{operation_name} failed: {error_code}",
            operation_name,
            status_code=http_status
        )
    except Exception as e:
        # Network/connection errors
        raise S3OperationError(str(e), operation_name, status_code=503)

# ----------------------------
# Registration Function
# ----------------------------

def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app"""
    app.add_exception_handler(InvalidKeyError, invalid_key_handler)
    app.add_exception_handler(BucketAlreadyExistsError, bucket_exists_handler)
    app.add_exception_handler(BucketNotFoundError, bucket_not_found_handler)
    app.add_exception_handler(S3OperationError, s3_operation_handler)
