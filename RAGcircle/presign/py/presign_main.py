from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from aiobotocore.session import get_session
from fastapi import APIRouter, Depends, FastAPI, Request

from ops_bucket import create_collection, delete_collection, list_collections
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
from exceptions import register_exception_handlers



from fastapi import UploadFile, File, Form, HTTPException
from pydantic import BaseModel


from database_ops import create_db, ProjectDB, DocumentDB
from auth import UserCreds, authenticated
from projects import authorize_project

logger = logging.getLogger(__name__)


# ----------------------------
# Request/Response Models
# ----------------------------

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


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


def get_project_db(request: Request) -> ProjectDB:
    return request.app.state.project_db

def get_document_db(request: Request) -> DocumentDB:
    return request.app.state.document_db



 


# ----------------------------
# Lifespan
# ----------------------------

from collectors import QdrantStore, BM25Store
qdrant = QdrantStore("http://localhost:8903", 768)
opensearch = BM25Store("http://localhost:8905")


@asynccontextmanager
async def lifecycle(app: FastAPI):
    """Initialize and cleanup S3 client"""
    settings = Settings()
    session = get_session()

    # if state.settings.redis_url:
    # try:
    #     state.redis = Redis.from_url(state.settings.redis_url, decode_responses=True)
    #     state.project_store = ProjectStore(
    #         redis=state.redis,
    #         max_projects_per_user=state.settings.max_projects_per_user
    #     )
    #     logger.info("redis_initialized", extra={"extra": {"url": state.settings.redis_url}})
    # except Exception as e:
    #     logger.error("redis_init_failed", extra={"extra": {"error": str(e)}})

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
        project_db, document_db = await stack.enter_async_context(
            create_db(
                settings.database_url,
                max_projects_per_user=settings.max_projects_per_user,
            )
        )
        app.state.s3 = s3
        app.state.project_db = project_db
        app.state.document_db = document_db
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
# Project Endpoints
# ----------------------------

@protected_router.post("/v1/projects")
async def create_project(
    payload: ProjectCreate,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
):
    """Create a new project (enforces max limit per user)."""
    project = await project_db.create(
        user_id=user.user_id,
        name=payload.name,
        description=payload.description,
    )


    await qdrant.ensure_collection(project.project_id, dimension=768)
    await opensearch.ensure_index(project.project_id)
    return {"ok": True, "project": project.to_dict()}


@protected_router.get("/v1/projects")
async def list_projects(
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
):
    """List all projects owned by the authenticated user."""
    projects = await project_db.list_for_user(user.user_id)
    return {"ok": True, "projects": [p.to_dict() for p in projects]}


@protected_router.get("/v1/projects/{project_id}")
async def get_project(
    project_id: str,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
):
    """Get a single project (with ownership check)."""
    project = await authorize_project(user, project_id, project_db)
    return {"ok": True, "project": project.to_dict()}


@protected_router.patch("/v1/projects/{project_id}")
async def update_project(
    project_id: str,
    payload: ProjectUpdate,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
):
    """Update project metadata."""
    # Verify ownership first
    await authorize_project(user, project_id, project_db)

    project = await project_db.update(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
    )
    return {"ok": True, "project": project.to_dict() if project else None}


@protected_router.delete("/v1/projects/{project_id}")
async def delete_project(
    project_id: str,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    s3 = Depends(get_s3)
):
    """
    Delete a project (soft delete).
    Also deletes all documents associated with the project.
    """
    # Verify ownership first
    await authorize_project(user, project_id, project_db)

    await qdrant.delete_collection(project_id)
    await opensearch.delete_index(project_id)

    documents = await document_db.list_by_project(project_id)
    for doc in documents: 
        await s3.delete_object(Bucket="ragfun", Key=project_id + "_" + doc.get("doc_id"))

    # Delete all documents for this project first
    await document_db.delete_by_project(project_id)

    # Soft-delete the project
    deleted = await project_db.delete(project_id, user.user_id)


    return {"ok": deleted, "project_id": project_id}



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



from storage import UploadMeta, upload_with_content_addressing, detect_content_type
import uuid



# what could be improved 
# user permissions/max_file_size
# 
@protected_router.post("/v1/file/upload")
async def upload(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    title: str = Form(...),
    description: str | None = Form(None),

    user: UserCreds = Depends(authenticated),
    s3_client = Depends(get_s3),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    settings: Settings = Depends(get_settings),
):
    # Verify user owns the project
    await authorize_project(user, project_id, project_db)

    content_type = await detect_content_type(file)
    meta = UploadMeta(title=title, description=description)
    # doc_id = uuid.uuid4().hex
    
    async def file_stream():
        while chunk := await file.read(64 * 1024):
            yield chunk

    # Use project_id as storage prefix for S3 organization
    storage_prefix = f"{project_id}_"
    
    upload_result = await upload_with_content_addressing(
        s3=s3_client,
        bucket=settings.bucket_name,
        request_stream=file_stream(),
        content_type=content_type,
        max_bytes=1 * 1024 * 1024,
        storage_prefix=storage_prefix,
        #doc_id=doc_id,
        #project_id=project_id,
    )
    # print(upload_result)

    await document_db.persist_document(upload_result.storage_id, project_id, upload_result, meta)
    
    return {
        "doc_id": upload_result.storage_id,
        "project_id": project_id,
        # "storage_id": ,
        "size": upload_result.size,
        "duplicate": upload_result.duplicate,
    }


@protected_router.get("/v1/projects/{project_id}/documents")
async def list_project_documents(
    project_id: str,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
):
    """List all documents in a project."""
    # Verify user owns the project
    await authorize_project(user, project_id, project_db)

    documents = await document_db.list_by_project(project_id)
    return {"ok": True, "documents": documents}


@protected_router.delete("/v1/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    s3 = Depends(get_s3),
    settings = Depends(get_settings)
):
    """Delete a document (with ownership check via project)."""
    # Get document to find its project
    doc = await document_db.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document_not_found")

    # Verify user owns the project this document belongs to
    await authorize_project(user, doc["project_id"], project_db)

    deleted = await document_db.delete(doc_id)

    await s3.delete_object(Bucket= settings.bucket_name, Key=doc['project_id'] + "_" + doc_id)
    return {"ok": deleted, "doc_id": doc_id}


# preinit bucket if possible + plus pass as a setting
# user passes project id and we actually add normal project_id functionality with dbs
# we can do auth from now on
# add proper crud for ops 
# if we remove a file we also have to remove it from db 
# also think about connecting the tables in the db (cascade delete and stuff)

# TODOs I want 
# maybe move to sql finally to avoid string slop 

# ----------------------------
# App Initialization
# ----------------------------

app = FastAPI(lifespan=lifecycle, title="S3 Presign API")
register_exception_handlers(app)
app.include_router(public_router)
app.include_router(protected_router)


import uvicorn

uvicorn.run(app=app, port=8912, host="localhost")