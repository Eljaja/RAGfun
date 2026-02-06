from typing import Any

from exceptions import safe_s3_call
from models import (
    ObjectDeleteRequest,
    ObjectDeleteResponse,
    ObjectInfo,
    ObjectListRequest,
    ObjectListResponse,
    PresignDownloadRequest,
    PresignDownloadResponse,
    PresignPostResponse,
    PresignPutResponse,
    PresignUploadRequest,
)


async def generate_presigned_post(
    s3_cli,
    request: PresignUploadRequest
) -> PresignPostResponse:
    """
    Generate presigned POST with conditions (more powerful than PUT).
    Supports file size limits, content-type enforcement, etc.
    """
    conditions = []
    fields = {}

    # Enforce content type if provided
    if request.content_type:
        conditions.append(["eq", "$Content-Type", request.content_type])
        fields["Content-Type"] = request.content_type

    # Enforce max file size if provided
    if request.max_size_bytes:
        conditions.append(["content-length-range", 1, request.max_size_bytes])

    response = await safe_s3_call(
        "generate_presigned_post",
        s3_cli.generate_presigned_post(
            Bucket=request.bucket,
            Key=request.key,
            Fields=fields if fields else None,
            Conditions=conditions if conditions else None,
            ExpiresIn=request.expires_seconds,
        )
    )

    return PresignPostResponse(
        url=response["url"],
        fields=response["fields"],
        expires=request.expires_seconds,
    )


async def generate_presigned_put_url(
    s3_cli,
    request: PresignUploadRequest
) -> PresignPutResponse:
    """Generate presigned PUT URL (legacy - prefer POST for better validation)"""
    params: dict[str, Any] = {
        "Bucket": request.bucket,
        "Key": request.key
    }

    # If you include ContentType in the signature, the uploader MUST send the same header.
    if request.content_type:
        params["ContentType"] = request.content_type

    url = await safe_s3_call(
        "generate_presigned_url",
        s3_cli.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=request.expires_seconds,
        )
    )

    return PresignPutResponse(
        method="PUT",
        url=url,
        expires=request.expires_seconds,
    )



async def generate_presigned_download_url(
    s3_cli,
    request: PresignDownloadRequest
) -> PresignDownloadResponse:
    """Generate presigned GET URL for object download"""
    params: dict[str, Any] = {
        "Bucket": request.bucket,
        "Key": request.key
    }

    url = await safe_s3_call(
        "generate_presigned_url",
        s3_cli.generate_presigned_url(
            ClientMethod="get_object",
            Params=params,
            ExpiresIn=request.expires_seconds,
        )
    )

    return PresignDownloadResponse(
        method="GET",
        url=url,
        expires=request.expires_seconds,
    )


async def delete_object(
    s3_cli,
    request: ObjectDeleteRequest
) -> ObjectDeleteResponse:
    """Delete an object from S3"""
    await safe_s3_call(
        "delete_object",
        s3_cli.delete_object(
            Bucket=request.bucket,
            Key=request.key
        )
    )

    return ObjectDeleteResponse(
        bucket=request.bucket,
        key=request.key,
        deleted=True,
    )


async def list_objects(
    s3_cli,
    request: ObjectListRequest
) -> ObjectListResponse:
    """List objects in a bucket with optional prefix filter"""
    params: dict[str, Any] = {
        "Bucket": request.bucket,
        "MaxKeys": request.max_keys,
    }

    if request.prefix:
        params["Prefix"] = request.prefix

    if request.continuation_token:
        params["ContinuationToken"] = request.continuation_token

    response = await safe_s3_call(
        "list_objects_v2",
        s3_cli.list_objects_v2(**params)
    )

    objects = []
    for obj in response.get("Contents", []):
        objects.append(ObjectInfo(
            key=obj["Key"],
            size=obj["Size"],
            last_modified=obj["LastModified"].isoformat(),
            etag=obj["ETag"].strip('"'),
        ))

    return ObjectListResponse(
        bucket=request.bucket,
        objects=objects,
        is_truncated=response.get("IsTruncated", False),
        continuation_token=response.get("NextContinuationToken"),
    )



