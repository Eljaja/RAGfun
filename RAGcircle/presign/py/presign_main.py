from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from aiobotocore.session import get_session
from fastapi import APIRouter, Depends, FastAPI, Request

from ops_bucket import create_collection, delete_collection, list_collections
from exceptions import register_exception_handlers
from models import (
    CollectionCreateResponse,
    CollectionDeleteResponse,
    CollectionListResponse,
    ObjectDeleteRequest,
    ObjectDeleteResponse,
    ObjectListRequest,
    ObjectListResponse,
    PresignDownloadRequest,
    PresignDownloadResponse,
    PresignPostResponse,
    PresignPutResponse,
    PresignUploadRequest,
)
from ops_object import (
    delete_object,
    generate_presigned_download_url,
    generate_presigned_post,
    generate_presigned_put_url,
    list_objects,
)
from settings import Settings

logger = logging.getLogger(__name__)


# ----------------------------
# Dependencies
# ----------------------------

def get_s3(request: Request):
    """Fetch the S3 client from app state"""
    s3 = getattr(request.app.state, "s3", None)
    if s3 is None:
        raise RuntimeError("S3 client not initialized")
    return s3


def get_settings(request: Request) -> Settings:
    """Fetch settings from app state"""
    return request.app.state.settings


# ----------------------------
# Lifespan
# ----------------------------

@asynccontextmanager
async def lifecycle(app: FastAPI):
    """Initialize and cleanup S3 client"""
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
        app.state.settings = settings
        yield

# ----------------------------
# Routers
# ----------------------------

public_router = APIRouter(prefix="/public", tags=["public"])
protected_router = APIRouter(prefix="/api", tags=["protected"])


# ----------------------------
# Public Endpoints (no auth)
# ----------------------------

@public_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


# ----------------------------
# Protected Endpoints (require auth)
# ----------------------------


# TOTHINK: might have more than one bucket creation op
# OR: well bucket can have subfolders
@protected_router.post("/collections/{bucket_name}", response_model=CollectionCreateResponse)
async def api_create_collection(
    bucket_name: str,
    s3=Depends(get_s3),
    settings: Settings = Depends(get_settings),
) -> CollectionCreateResponse:
    """Create a new S3 bucket/collection with event notifications"""
    return await create_collection(s3, bucket_name=bucket_name, settings=settings)


@protected_router.delete("/collections/{bucket_name}", response_model=CollectionDeleteResponse)
async def api_delete_collection(
    bucket_name: str,
    s3=Depends(get_s3),
) -> CollectionDeleteResponse:
    """Delete a bucket/collection and all objects within it (destructive operation)"""
    return await delete_collection(s3, bucket_name=bucket_name)


@protected_router.get("/collections", response_model=CollectionListResponse)
async def api_list_collections(
    s3=Depends(get_s3),
) -> CollectionListResponse:
    """List all buckets/collections"""
    return await list_collections(s3)


@protected_router.post("/objects/presign/upload", response_model=PresignPostResponse)
async def api_presign_upload(
    request: PresignUploadRequest,
    s3=Depends(get_s3),
) -> PresignPostResponse:
    """
    Generate presigned POST for object upload (recommended).
    Supports file size limits and content-type validation.
    """
    return await generate_presigned_post(s3, request)


@protected_router.post("/objects/presign/put", response_model=PresignPutResponse, tags=["legacy"])
async def api_presign_put(
    request: PresignUploadRequest,
    s3=Depends(get_s3),
) -> PresignPutResponse:
    """Generate presigned PUT URL for object upload (legacy - prefer /upload)"""
    return await generate_presigned_put_url(s3, request)


@protected_router.post("/objects/presign/download", response_model=PresignDownloadResponse)
async def api_presign_download(
    request: PresignDownloadRequest,
    s3=Depends(get_s3),
) -> PresignDownloadResponse:
    """Generate presigned GET URL for object download"""
    return await generate_presigned_download_url(s3, request)


@protected_router.post("/objects/delete", response_model=ObjectDeleteResponse)
async def api_delete_object(
    request: ObjectDeleteRequest,
    s3=Depends(get_s3),
) -> ObjectDeleteResponse:
    """Delete an object from S3"""
    return await delete_object(s3, request)


@protected_router.post("/objects/list", response_model=ObjectListResponse)
async def api_list_objects(
    request: ObjectListRequest,
    s3=Depends(get_s3),
) -> ObjectListResponse:
    """List objects in a bucket with optional prefix filter"""
    return await list_objects(s3, request)


# ----------------------------
# App Initialization
# ----------------------------

app = FastAPI(lifespan=lifecycle, title="S3 Presign API")
register_exception_handlers(app)
app.include_router(public_router)
app.include_router(protected_router)
