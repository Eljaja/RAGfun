from typing import Any

from botocore.exceptions import ClientError

from exceptions import BucketAlreadyExistsError, BucketNotFoundError, S3OperationError, safe_s3_call
from models import (
    BucketInfo,
    CollectionCreateResponse,
    CollectionDeleteResponse,
    CollectionListResponse,
)


# ----------------------------
# S3 Operations
# ----------------------------

async def bucket_exists(s3_cli, bucket_name: str) -> bool:
    """
    Check if bucket exists.
    Returns True if exists, False if not found.
    Raises S3OperationError for network/permission issues.
    """
    try:
        await s3_cli.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        http_status = e.response.get(
            "ResponseMetadata", {}).get("HTTPStatusCode", 500)

        if error_code in ("404", "NoSuchBucket"):
            return False

        raise S3OperationError(
            f"Failed to check bucket: {error_code}",
            "head_bucket",
            status_code=http_status
        )
    except Exception as e:
        # Catch-all for network/connection errors (network timeouts, DNS issues, etc.)
        raise S3OperationError(str(e), "head_bucket", status_code=503)


async def create_collection(
    s3_cli,
    bucket_name: str,
    aws_region,
    queue_arn,
) -> CollectionCreateResponse:
    """
    Create bucket with event notifications.
    Raises BucketAlreadyExistsError if bucket exists.
    """
    if await bucket_exists(s3_cli, bucket_name):
        return False

    await create_bucket(s3_cli, bucket_name=bucket_name, region=aws_region)
    await subscribe_bucket_to_events(
        s3_cli,
        bucket_name=bucket_name,
        queue_arn=queue_arn,
        events=["s3:ObjectCreated:*", "s3:ObjectRemoved:*"],
        config_id="rabbit-notification",
    )

    return True


async def create_bucket(s3_cli, bucket_name: str, region: str = "us-east-1") -> None:
    """Create bucket only. No side effects beyond creation."""
    create_kwargs: dict[str, Any] = {"Bucket": bucket_name}
    if region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {
            "LocationConstraint": region}

    await safe_s3_call("create_bucket", s3_cli.create_bucket(**create_kwargs))


async def subscribe_bucket_to_events(
    s3_cli,
    bucket_name: str,
    queue_arn: str,
    events: list[str],
    config_id: str,
) -> None:
    """
    Attach notifications to a freshly created bucket.
    This MUST NOT be called for existing buckets.
    """
    notification_config = {
        "QueueConfigurations": [
            {
                "Id": config_id,
                "QueueArn": queue_arn,
                "Events": events,
            }
        ]
    }

    await safe_s3_call(
        "put_bucket_notification",
        s3_cli.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration=notification_config,
        )
    )



