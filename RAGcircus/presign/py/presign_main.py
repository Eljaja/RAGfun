from __future__ import annotations

import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from aiobotocore.session import get_session
from fastapi import Depends, FastAPI, HTTPException, Request


from settings import Settings

# from aiobotocore.config import Config



# Routers with auth/no-auth
# Exception handlers that map infra/buisness errors to http codes and messages for the user
# and also create logs/traces and other things for us to investigate 



# ----------------------------
# Models / validation helpers
# ----------------------------

_KEY_BAD_PATTERNS = [
    r"^\s*$",  # empty / whitespace
    r"\.\.",  # path traversal
    r"\\",  # backslashes
    r"[\x00-\x1f\x7f]",  # control chars
]
_KEY_BAD_RE = re.compile("|".join(_KEY_BAD_PATTERNS))


def validate_s3_key(key: str) -> None:
    """
    Basic "don't be weird" validation.
    You can tighten this (allowed chars, max length, prefix rules, etc.).
    """
    if _KEY_BAD_RE.search(key):
        raise HTTPException(status_code=400, detail="Invalid object key/filename.")


@dataclass(frozen=True)
class ObjectMetadata:
    bucket: str
    key: str
    expires_seconds: int = 60
    content_type: Optional[str] = None


# ----------------------------
# FastAPI lifespan (create client once)
# ----------------------------


@asynccontextmanager
async def lifecycle(app: FastAPI):
    settings = Settings()

    session = get_session()

    async with AsyncExitStack() as stack:
        s3 = await stack.enter_async_context(
            session.create_client(
                "s3",
                endpoint_url=settings.s3_endpoint_url,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
            )
        )
        app.state.s3 = s3
        app.state.settings = settings  # store for later use
        yield

    # stack closes the client


app = FastAPI(lifespan=lifecycle)


def get_s3(request: Request):
    """
    Cheap dependency: fetch the singleton client from app.state.
    (You *can* also put it on request.state in middleware, but app.state is simpler.)
    """
    s3 = getattr(request.app.state, "s3", None)
    if s3 is None:
        raise RuntimeError("S3 client not initialized")
    return s3


def get_settings(request: Request):
    return request.app.state.settings


# ----------------------------
# Core ops you asked for
# ----------------------------


async def create_collection(s3_cli, bucket_name: str, region: str) -> dict[str, Any]:
    """
    Orchestrator.
    If bucket already exists, short-circuit and do NOT mutate it.
    """
    try:
        await s3_cli.head_bucket(Bucket=bucket_name)
        raise HTTPException(
            status_code=400,
            detail={
                "bucket": bucket_name,
                "created": False,
                "reason": "already-exists",
                "notifications_set": False,
            },
        )
    except Exception:
        # we assume the happy path where we do not
        pass

    await create_bucket(s3_cli, bucket_name=bucket_name, region=region)
    await subscribe_bucket_to_events(
        s3_cli,
        bucket_name=bucket_name,
        queue_arn="arn:rustfs:sqs:us-east-1:webhook:webhook",
        events=["s3:ObjectCreated:*", "s3:ObjectRemoved:*"],
        config_id="webhook-notification",
    )

    return {
        "bucket": bucket_name,
        "created": True,
        "notifications_set": True,
    }


async def create_bucket(s3_cli, bucket_name: str, region: str) -> None:
    """
    Create bucket only. No side effects beyond creation.
    """
    create_kwargs: dict[str, Any] = {"Bucket": bucket_name}
    if region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}

    try:
        await s3_cli.create_bucket(**create_kwargs)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create bucket: {e}",
        )


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

    try:
        await s3_cli.put_bucket_notification_configuration(
            Bucket=bucket_name,
            NotificationConfiguration=notification_config,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set bucket notification configuration: {e}",
        )


async def add_object(s3_cli, obj_metadata: ObjectMetadata) -> dict[str, Any]:
    validate_s3_key(obj_metadata.key)

    params: dict[str, Any] = {"Bucket": obj_metadata.bucket, "Key": obj_metadata.key}

    # If you include ContentType in the signature, the uploader MUST send the same header.
    if obj_metadata.content_type:
        params["ContentType"] = obj_metadata.content_type

    try:
        # In aiobotocore, generate_presigned_url is available on the client.
        url = await s3_cli.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=int(obj_metadata.expires_seconds),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to presign: {e}")

    return {
        "method": "PUT",
        "url": url,
        "expires": int(obj_metadata.expires_seconds),
    }


# ----------------------------
# Stubs (as requested)
# ----------------------------


async def remove_collection(s3_cli, bucket_name: str):
    # TODO: delete all objects + delete bucket
    raise HTTPException(status_code=501, detail="Not implemented yet")


async def list_collection(s3_cli):
    # TODO: list buckets
    raise HTTPException(status_code=501, detail="Not implemented yet")


async def remove_object(request: Request):
    raise HTTPException(status_code=501, detail="Not implemented yet")


async def download_object(request: Request):
    raise HTTPException(status_code=501, detail="Not implemented yet")


async def list_object(request: Request):
    raise HTTPException(status_code=501, detail="Not implemented yet")


def secure_download_object(request: Request):
    raise HTTPException(status_code=501, detail="Not implemented yet")


# ----------------------------
# Tiny demo endpoints
# ----------------------------


@app.post("/collections/{bucket_name}")
async def api_create_bucket(
    bucket_name: str,
    s3=Depends(get_s3),
    settings=Depends(get_settings),
):
    region = settings.aws_region
    return await create_collection(s3, bucket_name=bucket_name, region=region)


@app.post("/objects/presign/put")
async def api_presign_put(meta: dict, s3=Depends(get_s3)):
    # quick-and-dirty parsing; replace with Pydantic if you want
    obj = ObjectMetadata(
        bucket=meta["bucket"],
        key=meta["key"],
        expires_seconds=int(meta.get("expires_seconds", 60)),
        content_type=meta.get("content_type"),
    )
    return await add_object(s3, obj)
