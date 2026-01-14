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
    settings,
) -> CollectionCreateResponse:
    """
    Create bucket with event notifications.
    Raises BucketAlreadyExistsError if bucket exists.
    """
    if await bucket_exists(s3_cli, bucket_name):
        raise BucketAlreadyExistsError(bucket_name)

    await create_bucket(s3_cli, bucket_name=bucket_name, region=settings.aws_region)
    await subscribe_bucket_to_events(
        s3_cli,
        bucket_name=bucket_name,
        queue_arn=settings.queue_arn,
        events=["s3:ObjectCreated:*", "s3:ObjectRemoved:*"],
        config_id="rabbit-notification",
    )

    return CollectionCreateResponse(
        bucket=bucket_name,
        created=True,
        notifications_set=True,
    )


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


async def delete_collection(
    s3_cli,
    bucket_name: str
) -> CollectionDeleteResponse:
    """
    Delete all objects in a bucket and then delete the bucket itself.
    This is a destructive operation that cannot be undone.
    """
    if not await bucket_exists(s3_cli, bucket_name):
        raise BucketNotFoundError(bucket_name)

    # List and delete all objects in the bucket
    objects_deleted = 0
    continuation_token = None

    while True:
        # List objects
        list_params: dict[str, Any] = {"Bucket": bucket_name, "MaxKeys": 1000}
        if continuation_token:
            list_params["ContinuationToken"] = continuation_token

        response = await safe_s3_call(
            "list_objects_v2",
            s3_cli.list_objects_v2(**list_params)
        )

        # Delete objects if any
        contents = response.get("Contents", [])
        if contents:
            objects_to_delete = [{"Key": obj["Key"]} for obj in contents]
            await safe_s3_call(
                "delete_objects",
                s3_cli.delete_objects(
                    Bucket=bucket_name,
                    Delete={"Objects": objects_to_delete}
                )
            )
            objects_deleted += len(objects_to_delete)

        # Check if there are more objects
        if not response.get("IsTruncated", False):
            break
        continuation_token = response.get("NextContinuationToken")

    # Delete the bucket
    await safe_s3_call(
        "delete_bucket",
        s3_cli.delete_bucket(Bucket=bucket_name)
    )

    return CollectionDeleteResponse(
        bucket=bucket_name,
        deleted=True,
        objects_deleted=objects_deleted,
    )


async def list_collections(s3_cli) -> CollectionListResponse:
    """List all buckets/collections"""
    response = await safe_s3_call(
        "list_buckets",
        s3_cli.list_buckets()
    )

    buckets = []
    for bucket in response.get("Buckets", []):
        buckets.append(BucketInfo(
            name=bucket["Name"],
            creation_date=bucket["CreationDate"].isoformat(),
        ))

    return CollectionListResponse(
        buckets=buckets,
        count=len(buckets),
    )