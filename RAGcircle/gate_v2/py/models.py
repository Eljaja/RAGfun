
from pydantic import BaseModel, Field, field_validator
import re

from settings import Constants


constants = Constants()
# ----------------------------
# Models
# ----------------------------

_KEY_BAD_PATTERNS = [
    r"^\s*$",  # empty / whitespace
    r"\.\.",  # path traversal
    r"\\",  # backslashes
    r"[\x00-\x1f\x7f]",  # control chars
]
_KEY_BAD_RE = re.compile("|".join(_KEY_BAD_PATTERNS))


class PresignUploadRequest(BaseModel):
    """Request model for presigned upload URL generation"""
    bucket: str
    key: str
    expires_seconds: int = Field(default=60, ge=1, le=constants.max_link_ttl)
    content_type: str | None = None
    max_size_bytes: int | None = Field(default=None, ge=1, le=constants.max_file_size)  # Max 100MB

    model_config = {"frozen": True}

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        if _KEY_BAD_RE.search(v):
            raise ValueError("Contains invalid characters or patterns")
        return v


class PresignPostResponse(BaseModel):
    """Response model for presigned POST upload"""
    url: str
    fields: dict[str, str]
    expires: int


class PresignPutResponse(BaseModel):
    """Response model for presigned PUT URL (legacy)"""
    method: str
    url: str
    expires: int


class CollectionCreateResponse(BaseModel):
    """Response model for collection creation"""
    bucket: str
    created: bool
    notifications_set: bool


class CollectionDeleteResponse(BaseModel):
    """Response model for collection deletion"""
    bucket: str
    deleted: bool
    objects_deleted: int


class BucketInfo(BaseModel):
    """Individual bucket information"""
    name: str
    creation_date: str


class CollectionListResponse(BaseModel):
    """Response model for listing collections/buckets"""
    buckets: list[BucketInfo]
    count: int


class PresignDownloadRequest(BaseModel):
    """Request model for presigned download URL generation"""
    bucket: str
    key: str
    expires_seconds: int = Field(default=60, ge=1, le=constants.max_link_ttl)
    
    model_config = {"frozen": True}


class PresignDownloadResponse(BaseModel):
    """Response model for presigned GET URL"""
    method: str
    url: str
    expires: int


class ObjectDeleteRequest(BaseModel):
    """Request model for object deletion"""
    bucket: str
    key: str
    
    model_config = {"frozen": True}


class ObjectDeleteResponse(BaseModel):
    """Response model for object deletion"""
    bucket: str
    key: str
    deleted: bool


class ObjectListRequest(BaseModel):
    """Request model for listing objects"""
    bucket: str
    prefix: str | None = None
    max_keys: int = Field(default=1000, ge=1, le=1000)
    continuation_token: str | None = None
    
    model_config = {"frozen": True}


class ObjectInfo(BaseModel):
    """Individual object information"""
    key: str
    size: int
    last_modified: str
    etag: str


class ObjectListResponse(BaseModel):
    """Response model for listing objects"""
    bucket: str
    objects: list[ObjectInfo]
    is_truncated: bool
    continuation_token: str | None = None