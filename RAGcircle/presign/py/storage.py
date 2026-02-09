import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi import HTTPException
from fastapi import UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from dataclasses import dataclass
import uuid


@dataclass
class UploadPart:
    etag: str
    number: int


@dataclass
class StreamStats:
    size: int = 0
    sha256: str = ""


class PartBuffer:
    """Accumulates chunks and yields full parts."""
    
    def __init__(self, part_size: int = 8 * 1024 * 1024):
        self.part_size = part_size
        self._buffer = bytearray()
    
    def add(self, chunk: bytes) -> bytes | None:
        """Add chunk, return part data if buffer is full."""
        self._buffer.extend(chunk)
        if len(self._buffer) >= self.part_size:
            part = bytes(self._buffer[:self.part_size])
            self._buffer = bytearray(self._buffer[self.part_size:])
            return part
        return None
    
    def flush(self) -> bytes | None:
        """Return remaining data."""
        if self._buffer:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data
        return None


@asynccontextmanager
async def multipart_upload(s3, bucket: str, key: str, content_type: str):
    """
    Context manager for S3 multipart upload.
    Automatically aborts on exception.
    """
    resp = await s3.create_multipart_upload(
        Bucket=bucket, Key=key, ContentType=content_type
    )
    upload_id = resp["UploadId"]
    parts: list[UploadPart] = []
    
    async def upload_part(data: bytes) -> None:
        part_num = len(parts) + 1
        resp = await s3.upload_part(
            Bucket=bucket,
            Key=key,
            PartNumber=part_num,
            UploadId=upload_id,
            Body=data,
        )
        parts.append(UploadPart(etag=resp["ETag"], number=part_num))
    
    async def complete() -> None:
        if parts:
            await s3.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={
                    "Parts": [{"ETag": p.etag, "PartNumber": p.number} for p in parts]
                },
            )
        else:
            # Empty file fallback
            await s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            await s3.put_object(Bucket=bucket, Key=key, Body=b"", ContentType=content_type)
    
    try:
        yield upload_part, complete
    except Exception:
        await s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise


async def validated_stream(
    stream: AsyncIterator[bytes],
    max_bytes: int,
) -> AsyncIterator[tuple[bytes, StreamStats]]:
    """
    Wraps stream with size validation and hash computation.
    Yields (chunk, running_stats).
    """
    hasher = hashlib.sha256()
    total = 0
    
    async for chunk in stream:
        total += len(chunk)
        if total > max_bytes:
            # TODO: switch to domain exceptions + exception handlers 
            raise HTTPException(413, f"File exceeds {max_bytes // (1024*1024)}MB")
        hasher.update(chunk)
        yield chunk, StreamStats(size=total, sha256=hasher.hexdigest())


# --- Clean main function ---

async def upload_via_multipart(
    s3,
    bucket: str,
    storage_id: str,
    request_stream: AsyncIterator[bytes],
    content_type: str,
    max_bytes: int,
    part_size: int = 8 * 1024 * 1024,
    # TODO: add metadata
) -> tuple[int, str]:
    """Stream upload via S3 multipart. Returns (size, sha256)."""
    
    buffer = PartBuffer(part_size)
    stats = StreamStats()
    
    async with multipart_upload(s3, bucket, storage_id, content_type) as (upload_part, complete):
        async for chunk, stats in validated_stream(request_stream, max_bytes):
            if part_data := buffer.add(chunk):
                await upload_part(part_data)
        
        if remaining := buffer.flush():
            await upload_part(remaining)
        
        await complete()
    
    return stats.size, stats.sha256



"""
Layer 1: Immediate retries with backoff
Layer 2: Dead letter queue (DLQ) for failed deletes
Layer 3: S3 lifecycle rule (auto-delete _temp/* after 24h)
Layer 4: Background cleanup worker
"""


class UploadMeta(BaseModel):
    title: str
    description: str | None = None
    extra: dict = {}


@dataclass
class UploadResult:
    storage_id: str
    size: int
    sha256: str
    duplicate: bool




def sha256_hex_to_uuid_v8(sha256_hex: str) -> uuid.UUID:
    """
    Convert 64-char SHA-256 hex → UUIDv8.
    e.g. sha256_hex = "b94d27b9934d3e08a52e52d7da7dabfac1c3012b756d6f0..."
    """
    if len(sha256_hex) != 64 or not all(c in '0123456789abcdefABCDEF' for c in sha256_hex):
        raise ValueError("Must be exactly 64 hex chars")
    
    digest = bytes.fromhex(sha256_hex)  # 32 raw bytes from hex
    bytes_arr = bytearray(digest[:16])  # Only take first 16 bytes for UUID
    
    # Set version to 8 (bits 0b1000 in the high nibble of byte 6)
    bytes_arr[6] = (bytes_arr[6] & 0x0F) | 0x80
    
    # Set variant to RFC 4122 (bits 0b10 in the high 2 bits of byte 8)
    bytes_arr[8] = (bytes_arr[8] & 0x3F) | 0x80
    
    return str(uuid.UUID(bytes=bytes(bytes_arr)))



async def upload_with_content_addressing(
    s3,
    bucket: str,
    request_stream,
    content_type: str,
    max_bytes: int,
    storage_prefix: str, 
) -> UploadResult:
    temp_key = f"_temp_{uuid.uuid4().hex}"
    size, sha256 = await upload_via_multipart(
        s3=s3,
        bucket=bucket,
        storage_id=temp_key,
        request_stream=request_stream,
        content_type=content_type,
        max_bytes=max_bytes,
    )
    real_doc_id = sha256_hex_to_uuid_v8(sha256)
    storage_id = storage_prefix + real_doc_id
    
    # Check if content already exists
    try:
        await s3.head_object(Bucket=bucket, Key=storage_id)
        exists = True
    except s3.exceptions.ClientError:
        exists = False
    
    if exists:
        await s3.delete_object(Bucket=bucket, Key=temp_key)
    else:
        await s3.copy_object(
            Bucket=bucket,
            CopySource=f"{bucket}/{temp_key}",
            Key=storage_id,
            ContentType=content_type,
            MetadataDirective="REPLACE",
            Metadata={
                 "sha256": sha256,
            },
        )
        await s3.delete_object(Bucket=bucket, Key=temp_key)
    
    # await db.create_document(
    #     doc_id=doc_id,
    #     storage_id=storage_id,
    #     title=meta.title,
    #     description=meta.description,
    #     size=size,
    #     content_type=content_type,
    # )
    
    return UploadResult(storage_id=real_doc_id, size=size, sha256=sha256, duplicate=exists)


# @app.post("/upload")
import magic
import mimetypes

async def detect_content_type(file: UploadFile) -> str:
    # Try magic bytes first
    header = await file.read(2048)
    await file.seek(0)
    
    mime = magic.from_buffer(header, mime=True)
    
    # Fall back to extension if magic gives generic result
    if mime in ("application/octet-stream", "text/plain") and file.filename:
        ext_mime, _ = mimetypes.guess_type(file.filename)
        if ext_mime:
            return ext_mime
    
    return mime or "application/octet-stream"



# upload
# can attach 256sha so we could check this stuff in advance 
# delete 
# simple 




