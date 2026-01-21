from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("storage.files")


class StorageBackend:
    """Abstract storage backend interface."""

    def store(
        self,
        doc_id: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
        max_size_bytes: int | None = None,
    ) -> tuple[str, int, str]:
        """Store file stream and return (storage_id, size_bytes, content_hash)."""
        raise NotImplementedError

    def retrieve(self, storage_id: str) -> bytes | None:
        """Retrieve file by storage_id."""
        raise NotImplementedError

    def delete(self, storage_id: str) -> bool:
        """Delete file by storage_id."""
        raise NotImplementedError

    def health(self) -> bool:
        """Check storage health."""
        raise NotImplementedError


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info("local_storage_initialized", extra={"path": str(self.storage_path)})

    def _get_path(self, storage_id: str) -> Path:
        """Get file path from storage_id."""
        # Use subdirectories to avoid too many files in one directory
        # storage_id format: {hash_prefix}/{hash}
        if len(storage_id) >= 4:
            subdir = storage_id[:2]
            return self.storage_path / subdir / storage_id
        return self.storage_path / storage_id

    @staticmethod
    def _storage_id_for_doc_id(doc_id: str) -> str:
        # Deterministic, safe key derived from doc_id (not content hash).
        return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()

    def store(
        self,
        doc_id: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
        max_size_bytes: int | None = None,
    ) -> tuple[str, int, str]:
        """Store file locally from stream and return (storage_id, size_bytes, content_hash)."""
        tmp_dir = self.storage_path / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{uuid.uuid4().hex}.tmp"

        hasher = hashlib.sha256()
        size = 0
        try:
            with open(tmp_path, "wb") as out:
                while True:
                    chunk = fileobj.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if max_size_bytes is not None and size > max_size_bytes:
                        raise ValueError("file_too_large")
                    hasher.update(chunk)
                    out.write(chunk)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        content_hash = hasher.hexdigest()
        storage_id = self._storage_id_for_doc_id(doc_id)
        file_path = self._get_path(storage_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, file_path)

        logger.info("local_storage_stored", extra={"doc_id": doc_id, "storage_id": storage_id, "size": size})
        return storage_id, size, content_hash

    def retrieve(self, storage_id: str) -> bytes | None:
        """Retrieve file from local storage."""
        file_path = self._get_path(storage_id)
        if not file_path.exists():
            logger.warning("local_storage_not_found", extra={"storage_id": storage_id})
            return None

        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error("local_storage_retrieve_error", extra={"storage_id": storage_id, "error": str(e)})
            return None

    def delete(self, storage_id: str) -> bool:
        """Delete file from local storage."""
        file_path = self._get_path(storage_id)
        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            # Try to remove parent directory if empty
            try:
                file_path.parent.rmdir()
            except OSError:
                pass  # Directory not empty or doesn't exist
            logger.info("local_storage_deleted", extra={"storage_id": storage_id})
            return True
        except Exception as e:
            logger.error("local_storage_delete_error", extra={"storage_id": storage_id, "error": str(e)})
            return False

    def health(self) -> bool:
        """Check local storage health."""
        try:
            test_file = self.storage_path / ".healthcheck"
            test_file.write_text("ok")
            test_file.unlink()
            return True
        except Exception:
            return False


class S3StorageBackend(StorageBackend):
    """S3-compatible storage backend (MinIO)."""

    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ):
        self.bucket = bucket
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self._ensure_bucket()
        logger.info("s3_storage_initialized", extra={"endpoint": endpoint_url, "bucket": bucket})

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
                logger.info("s3_bucket_created", extra={"bucket": self.bucket})
            except Exception as e:
                logger.error("s3_bucket_create_error", extra={"bucket": self.bucket, "error": str(e)})
                raise

    def store(
        self,
        doc_id: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
        max_size_bytes: int | None = None,
    ) -> tuple[str, int, str]:
        """Store file in S3 from stream and return (storage_id, size_bytes, content_hash)."""
        storage_id = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        hasher = hashlib.sha256()
        size = 0
        upload_id = None
        parts: list[dict[str, str | int]] = []
        part_number = 1
        part_size = 8 * 1024 * 1024

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            resp = self.s3_client.create_multipart_upload(Bucket=self.bucket, Key=storage_id, **extra_args)
            upload_id = resp["UploadId"]

            while True:
                chunk = fileobj.read(part_size)
                if not chunk:
                    break
                size += len(chunk)
                if max_size_bytes is not None and size > max_size_bytes:
                    raise ValueError("file_too_large")
                hasher.update(chunk)
                up = self.s3_client.upload_part(
                    Bucket=self.bucket,
                    Key=storage_id,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    Body=chunk,
                )
                parts.append({"ETag": up["ETag"], "PartNumber": part_number})
                part_number += 1

            if parts:
                self.s3_client.complete_multipart_upload(
                    Bucket=self.bucket,
                    Key=storage_id,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
            else:
                # Empty file fallback
                if upload_id:
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=storage_id,
                        UploadId=upload_id,
                    )
                    upload_id = None
                self.s3_client.put_object(Bucket=self.bucket, Key=storage_id, Body=b"", **extra_args)

            content_hash = hasher.hexdigest()
            logger.info("s3_storage_stored", extra={"doc_id": doc_id, "storage_id": storage_id, "size": size})
            return storage_id, size, content_hash
        except Exception as e:
            logger.error("s3_storage_store_error", extra={"doc_id": doc_id, "error": str(e)})
            if upload_id:
                try:
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=storage_id,
                        UploadId=upload_id,
                    )
                except Exception:
                    pass
            # best-effort cleanup of partial object
            try:
                self.s3_client.delete_object(Bucket=self.bucket, Key=storage_id)
            except Exception:
                pass
            raise

    def retrieve(self, storage_id: str) -> bytes | None:
        """Retrieve file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=storage_id)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning("s3_storage_not_found", extra={"storage_id": storage_id})
                return None
            logger.error("s3_storage_retrieve_error", extra={"storage_id": storage_id, "error": str(e)})
            return None
        except Exception as e:
            logger.error("s3_storage_retrieve_error", extra={"storage_id": storage_id, "error": str(e)})
            return None

    def delete(self, storage_id: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=storage_id)
            logger.info("s3_storage_deleted", extra={"storage_id": storage_id})
            return True
        except Exception as e:
            logger.error("s3_storage_delete_error", extra={"storage_id": storage_id, "error": str(e)})
            return False

    def health(self) -> bool:
        """Check S3 storage health."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


def create_storage_backend(settings) -> StorageBackend:
    """Factory function to create storage backend."""
    if settings.storage_backend == "s3":
        if not settings.s3_endpoint:
            raise ValueError("S3 endpoint is required for S3 backend")
        if not settings.s3_access_key or not settings.s3_secret_key:
            raise ValueError("S3 credentials are required for S3 backend")

        return S3StorageBackend(
            endpoint_url=str(settings.s3_endpoint),
            bucket=settings.s3_bucket,
            access_key=settings.s3_access_key.get_secret_value(),
            secret_key=settings.s3_secret_key.get_secret_value(),
            region=settings.s3_region,
        )
    else:
        return LocalStorageBackend(storage_path=settings.storage_path)

















