from __future__ import annotations
import uvicorn
import uuid
import re
import httpx
from storage import UploadMeta, upload_with_content_addressing, detect_content_type, download_from_s3
from collectors import QdrantStore, BM25Store
from collectors import BM25Store, QdrantStore, create_bm25_store, create_qdrant_store

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from aiobotocore.session import get_session
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from settings import Settings
from exceptions import register_exception_handlers


from fastapi import UploadFile, File, Form, HTTPException, Query
from inspect import Parameter, Signature
from pydantic import BaseModel


from database_ops import DocumentEventDB, create_db, ProjectDB, DocumentDB
from auth import UserCreds, authenticated
from projects import authorize_project
from settings import Constants


from ops_bucket import create_collection

logger = logging.getLogger(__name__)


# ----------------------------
# Request/Response Models
# ----------------------------

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    # Immutable after creation — changing these requires re-indexing all docs
    embedding_model: str = "intfloat/multilingual-e5-base"
    chunk_size: int = 512
    chunk_overlap: int = 64
    language: str = "ru"
    # Mutable — can be changed freely
    llm_model: str = "gemma-3-12b"


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    llm_model: str | None = None


EMBEDDING_DIMENSIONS: dict[str, int] = {
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-large-v2": 1024,
    "BAAI/bge-m3": 1024,
}


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


def get_qdrant(request: Request):
    return request.app.state.qdrant


def get_opensearch(request: Request):
    return request.app.state.opensearch


def get_event_db(request: Request) -> DocumentEventDB:
    return request.app.state.event_db


def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


def _effective_project_id(requested_project_id: str, settings: Settings) -> str:
    """Force a single project in stub compatibility mode."""
    if settings.stub_auth_enabled:
        return settings.stub_project_id
    return requested_project_id


async def _resolve_project(
    *,
    user: UserCreds,
    requested_project_id: str,
    settings: Settings,
    project_db: ProjectDB,
) -> object:
    """Return the effective project object, enforcing stub mode when enabled."""
    effective_project_id = _effective_project_id(requested_project_id, settings)
    if settings.stub_auth_enabled:
        return await project_db.ensure_project(
            project_id=effective_project_id,
            user_id=settings.stub_user_id,
            name=settings.stub_project_name,
            embedding_model=settings.stub_project_embedding_model,
            chunk_size=settings.stub_project_chunk_size,
            chunk_overlap=settings.stub_project_chunk_overlap,
            language=settings.stub_project_language,
            llm_model=settings.stub_project_llm_model,
        )
    return await authorize_project(user, effective_project_id, project_db)


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
        # create bucket in case it does not exist
        await create_collection(s3, settings.bucket_name, settings.aws_region, settings.queue_arn,)

        project_db, document_db, event_db = await stack.enter_async_context(
            create_db(
                settings.database_url,
                max_projects_per_user=settings.max_projects_per_user,
            )
        )

        qdrant: QdrantStore = await stack.enter_async_context(create_qdrant_store(settings.qdrant_url))
        opensearch: BM25Store = await stack.enter_async_context(create_bm25_store(settings.opensearch_url))

        if settings.stub_auth_enabled:
            stub_dim = EMBEDDING_DIMENSIONS.get(settings.stub_project_embedding_model, 768)
            await project_db.ensure_project(
                project_id=settings.stub_project_id,
                user_id=settings.stub_user_id,
                name=settings.stub_project_name,
                embedding_model=settings.stub_project_embedding_model,
                chunk_size=settings.stub_project_chunk_size,
                chunk_overlap=settings.stub_project_chunk_overlap,
                language=settings.stub_project_language,
                llm_model=settings.stub_project_llm_model,
            )
            await qdrant.ensure_collection(settings.stub_project_id, dimension=stub_dim)
            await opensearch.ensure_index(settings.stub_project_id)

        http_client = httpx.AsyncClient(timeout=120.0)

        app.state.s3 = s3
        app.state.project_db = project_db
        app.state.document_db = document_db

        app.state.event_db = event_db
        app.state.qdrant = qdrant
        app.state.opensearch = opensearch
        app.state.settings = settings
        app.state.http_client = http_client
        yield

        await http_client.aclose()


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
    settings: Settings = Depends(get_settings),
    project_db: ProjectDB = Depends(get_project_db),
    qdrant: QdrantStore = Depends(get_qdrant),
    opensearch: BM25Store = Depends(get_opensearch)
):
    """Create a new project (enforces max limit per user)."""
    if settings.stub_auth_enabled:
        project = await project_db.ensure_project(
            project_id=settings.stub_project_id,
            user_id=settings.stub_user_id,
            name=settings.stub_project_name,
            embedding_model=settings.stub_project_embedding_model,
            chunk_size=settings.stub_project_chunk_size,
            chunk_overlap=settings.stub_project_chunk_overlap,
            language=settings.stub_project_language,
            llm_model=settings.stub_project_llm_model,
        )
        return {"project": project.to_dict()}

    project = await project_db.create(
        user_id=user.user_id,
        name=payload.name,
        description=payload.description,
        embedding_model=payload.embedding_model,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        language=payload.language,
        llm_model=payload.llm_model,
    )

    dimension = EMBEDDING_DIMENSIONS.get(payload.embedding_model, 768)
    await qdrant.ensure_collection(project.project_id, dimension=dimension)
    await opensearch.ensure_index(project.project_id)
    return {"project": project.to_dict()}


@protected_router.get("/v1/projects")
async def list_projects(
    user: UserCreds = Depends(authenticated),
    settings: Settings = Depends(get_settings),
    project_db: ProjectDB = Depends(get_project_db),
):
    """List all projects owned by the authenticated user."""
    if settings.stub_auth_enabled:
        project = await project_db.ensure_project(
            project_id=settings.stub_project_id,
            user_id=settings.stub_user_id,
            name=settings.stub_project_name,
            embedding_model=settings.stub_project_embedding_model,
            chunk_size=settings.stub_project_chunk_size,
            chunk_overlap=settings.stub_project_chunk_overlap,
            language=settings.stub_project_language,
            llm_model=settings.stub_project_llm_model,
        )
        return {"projects": [project.to_dict()]}

    projects = await project_db.list_for_user(user.user_id)
    return {"projects": [p.to_dict() for p in projects]}


@protected_router.get("/v1/projects/{project_id}")
async def get_project(
    project_id: str,
    user: UserCreds = Depends(authenticated),
    settings: Settings = Depends(get_settings),
    project_db: ProjectDB = Depends(get_project_db),
):
    """Get a single project (with ownership check)."""
    project = await _resolve_project(
        user=user,
        requested_project_id=project_id,
        settings=settings,
        project_db=project_db,
    )
    return {"project": project.to_dict()}


@protected_router.delete("/v1/projects/{project_id}")
async def delete_project(
    project_id: str,
    user: UserCreds = Depends(authenticated),
    settings: Settings = Depends(get_settings),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    s3=Depends(get_s3),
    qdrant: QdrantStore = Depends(get_qdrant),
    opensearch: BM25Store = Depends(get_opensearch)
):
    """
    Delete a project (soft delete).
    Also deletes all documents associated with the project.
    """
    if settings.stub_auth_enabled:
        raise HTTPException(status_code=409, detail="stub_project_is_fixed")

    effective_project_id = _effective_project_id(project_id, settings)
    # Verify ownership first
    await authorize_project(user, effective_project_id, project_db)

    await qdrant.delete_collection(effective_project_id)
    await opensearch.delete_index(effective_project_id)

    documents, _ = await document_db.list_by_project(effective_project_id, limit=10_000)
    for doc in documents:
        # probably is fine
        await s3.delete_object(Bucket="ragfun", Key=effective_project_id + "_" + doc.get("storage_id"))

    # Delete all documents for this project first
    await document_db.delete_by_project(effective_project_id)

    # Soft-delete the project
    deleted = await project_db.delete(effective_project_id, user.user_id)

    return {"project_id": effective_project_id}


@protected_router.get("/v1/projects/{project_id}/documents")
async def list_project_documents(
    project_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: UserCreds = Depends(authenticated),
    settings: Settings = Depends(get_settings),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
):
    """List documents in a project with pagination."""
    effective_project_id = _effective_project_id(project_id, settings)
    await _resolve_project(
        user=user,
        requested_project_id=effective_project_id,
        settings=settings,
        project_db=project_db,
    )

    documents, total = await document_db.list_by_project(
        effective_project_id, limit=limit, offset=offset
    )
    return {
        "documents": documents,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ----------------------------
# Protected Endpoints (require auth)
# ----------------------------


# what could be improved
# user permissions/max_file_size
#

# another topic to look into
# well well well
class DocAttributes(BaseModel):
    title: str
    description: str | None = None
    uri: str | None = None
    source: str | None = None
    lang: str | None = None
    tags: str | None = None  # comma-separated
    acl: str | None = None  # comma-separated
    refresh: bool = False

    @classmethod
    def as_form(cls):
        params = []
        for field_name, field_info in cls.model_fields.items():
            if field_info.is_required():
                default = Form(...)
            else:
                default = Form(field_info.default)
            params.append(
                Parameter(
                    field_name,
                    Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field_info.annotation,
                )
            )

        async def form_body(**data):
            return cls(**data)

        form_body.__signature__ = Signature(params)
        return form_body


@protected_router.post("/v1/projects/{project_id}/upload")
async def upload(
    project_id: str,
    file: UploadFile = File(...),

    # Dependencies 
    attrs: DocAttributes = Depends(DocAttributes.as_form()),
    user: UserCreds = Depends(authenticated),
    s3_client=Depends(get_s3),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    settings: Settings = Depends(get_settings),
):
    doc_id = uuid.uuid4().hex
    effective_project_id = _effective_project_id(project_id, settings)
    # Verify user owns the project
    project = await _resolve_project(
        user=user,
        requested_project_id=effective_project_id,
        settings=settings,
        project_db=project_db,
    )

    content_type = await detect_content_type(file)
    meta = UploadMeta(title=attrs.title, description=attrs.description)

    # Validate content type
    constants = Constants()
    is_allowed = (
        content_type in constants.allowed_content_types or
        re.match(constants.allowed_text_pattern, content_type)
    )
    if not is_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Content type '{content_type}' is not allowed"
        )

    async def file_stream():
        while chunk := await file.read(64 * 1024):
            yield chunk

    # Use project_id as storage prefix for S3 organization
    storage_prefix = f"{effective_project_id}_"


    all_attrs = {
        **{f"doc__{k}": str(v) for k, v in attrs.model_dump().items() if v is not None},
        **{f"project__{k}": str(v) for k, v in project.to_dict().items() if v is not None},
    }
    upload_result = await upload_with_content_addressing(
        s3=s3_client,
        bucket=settings.bucket_name,
        request_stream=file_stream(),
        content_type=content_type,
        max_bytes=1 * 1024 * 1024,
        storage_prefix=storage_prefix,
        doc_id=doc_id,
        # TODO: fix mixed up pydantic and dataclasses
        doc_attrs = all_attrs, 
    )
    if upload_result.duplicate:
        raise HTTPException(status_code=409)

    # Do I like that option?
    # Not really but we can merge results and hide info about doc contents in a way
    # well shi
    # shi x2
    # shi x3
    # whatever, we do not handle millions of docs, maybe we will scale later
    
    await document_db.persist_document(doc_id, effective_project_id, upload_result, meta)

    return {
        "doc_id": doc_id,
        "project_id": effective_project_id,
        "size": upload_result.size,
    }



@protected_router.get("/v1/documents/{doc_id}/download")
async def download_document(
    doc_id: str,
    user: UserCreds = Depends(authenticated),
    s3_client=Depends(get_s3),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    settings: Settings = Depends(get_settings),
):
    """Download a document by its ID (with ownership check via project)."""
    # Get document to find its project and storage info
    doc = await document_db.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document_not_found")

    # Verify user owns the project this document belongs to
    await authorize_project(user, doc["project_id"], project_db)

    # Construct the S3 key using the same pattern as upload
    storage_key = f"{doc['project_id']}_{doc['storage_id']}"

    # Get content type from doc metadata or default to binary
    content_type = doc.get("content_type", "application/octet-stream")

    # Stream the file from S3
    file_stream = download_from_s3(
        s3=s3_client,
        bucket=settings.bucket_name,
        key=storage_key,
    )

    # Create a filename from the document title
    filename = f"{doc.get('title', doc_id)}.bin"

    return StreamingResponse(
        file_stream,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(doc.get("size", 0)),
        },
    )


# TOTHINK
# should I somehow combine info about a project into uuid + filesha256
# NONONONONO
# as Opus said
# requiring project_id is
# good practice even if doc_id is
# globally unique. It acts as a scoping guard
# (prevents accidentally deleting a doc
# from the wrong project) and
# makes authorization
# checks straightforward.
# I need


@protected_router.delete("/v1/documents/{doc_id}")
async def delete_document(
    # project_id: str,
    doc_id: str,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    s3=Depends(get_s3),
    settings=Depends(get_settings)
):
    """Delete a document (with ownership check via project)."""
    # Get document to find its project
    doc = await document_db.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document_not_found")

    # Verify user owns the project this document belongs to
    await authorize_project(user, doc["project_id"], project_db)

    deleted = await document_db.delete(doc_id)
    
    await s3.delete_object(Bucket=settings.bucket_name, Key=doc['project_id'] + "_" + doc.get("storage_id"))
    return {"doc_id": doc_id}


@protected_router.get("/v1/documents/{doc_id}")
async def get_document_info(
    # project_id: str,
    doc_id: str,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    # s3=Depends(get_s3),
    # settings=Depends(get_settings),
    event_db=Depends(get_event_db),
):
    """Delete a document (with ownership check via project)."""
    # Get document to find its project
    doc = await document_db.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document_not_found")

    # Verify user owns the project this document belongs to
    await authorize_project(user, doc["project_id"], project_db)

    doc_obj = await document_db.get(doc_id)
    storage_id = doc_obj.get("storage_id")

    return doc_obj


@protected_router.get("/v1/documents/{doc_id}/status")
async def get_document_info(
    # project_id: str,
    doc_id: str,
    user: UserCreds = Depends(authenticated),
    project_db: ProjectDB = Depends(get_project_db),
    document_db: DocumentDB = Depends(get_document_db),
    # s3=Depends(get_s3),
    # settings=Depends(get_settings),
    event_db=Depends(get_event_db),
):
    """Delete a document (with ownership check via project)."""
    # Get document to find its project
    doc = await document_db.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document_not_found")

    # Verify user owns the project this document belongs to
    await authorize_project(user, doc["project_id"], project_db)

    doc_obj = await document_db.get(doc_id)
    storage_id = doc_obj.get("storage_id")

    ev = await event_db.get_latest_event(doc_id=storage_id, project_id=doc_obj.get("project_id"))
    
    # this is truly meh on large scale 
    # blaaaaaaaaat 
    # blaaaaaaaaat
    # this is really bad 
    # nonononoo
    # nonononononono
    # shit 
    # no nono nonononono
    # what is the point? 
    # this is bad you cannot ship this 
    # no no no no no no no
    # 

    # well here is another idea -> we track prev event for all of the methods except ingestion
    # is this shit? 
    # maybe
    if getattr(ev, "doc_id"): 
        ev.doc_id = doc_id
    return ev


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True
    strategy: str = "hybrid"


class ChatRequestBody(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True
    strategy: str = "hybrid"
    max_retries: int = 1
    reflection_enabled: bool = True


@protected_router.post("/v1/projects/{project_id}/retrieve")
async def retrieve_documents(
    project_id: str,
    body: RetrieveRequest,
    user: UserCreds = Depends(authenticated),
    settings: Settings = Depends(get_settings),
    project_db: ProjectDB = Depends(get_project_db),
    client: httpx.AsyncClient = Depends(get_http_client),
):
    effective_project_id = _effective_project_id(project_id, settings)
    await _resolve_project(
        user=user,
        requested_project_id=effective_project_id,
        settings=settings,
        project_db=project_db,
    )

    resp = await client.post(
        f"{settings.retrieval_url}/retrieve",
        json={
            "project_id": effective_project_id,
            "query": body.query,
            "top_k": body.top_k,
            "rerank": body.rerank,
            "strategy": body.strategy,
        },
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Retrieval service error")

    return resp.json()


@protected_router.post("/v1/projects/{project_id}/chat")
async def chat_with_documents(
    project_id: str,
    body: ChatRequestBody,
    user: UserCreds = Depends(authenticated),
    settings: Settings = Depends(get_settings),
    project_db: ProjectDB = Depends(get_project_db),
    client: httpx.AsyncClient = Depends(get_http_client),
):
    effective_project_id = _effective_project_id(project_id, settings)
    await _resolve_project(
        user=user,
        requested_project_id=effective_project_id,
        settings=settings,
        project_db=project_db,
    )

    resp = await client.post(
        f"{settings.generator_url}/chat",
        json={
            "project_id": effective_project_id,
            "query": body.query,
            "top_k": body.top_k,
            "rerank": body.rerank,
            "strategy": body.strategy,
            "max_retries": body.max_retries,
            "reflection_enabled": body.reflection_enabled,
        },
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Generator service error")

    return resp.json()




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
# register_exception_handlers(app)
app.include_router(public_router)
app.include_router(protected_router)


if __name__ == "__main__":
    uvicorn.run(app=app, port=8912, host="0.0.0.0")
