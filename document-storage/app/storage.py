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

    def store(self, doc_id: str, content: bytes, content_type: str | None = None) -> str:
        """Store file and return storage_id."""
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

    def store(self, doc_id: str, content: bytes, content_type: str | None = None) -> str:
        """Store file locally and return storage_id."""
        # Generate deterministic storage_id from content hash
        content_hash = hashlib.sha256(content).hexdigest()
        storage_id = f"{content_hash[:2]}/{content_hash}"

        file_path = self._get_path(storage_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info("local_storage_stored", extra={"doc_id": doc_id, "storage_id": storage_id, "size": len(content)})
        return storage_id

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

    def store(self, doc_id: str, content: bytes, content_type: str | None = None) -> str:
        """Store file in S3 and return storage_id."""
        # Generate deterministic storage_id from content hash
        content_hash = hashlib.sha256(content).hexdigest()
        storage_id = f"{content_hash[:2]}/{content_hash}"

        try:
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type

            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=storage_id,
                Body=content,
                **extra_args,
            )
            logger.info("s3_storage_stored", extra={"doc_id": doc_id, "storage_id": storage_id, "size": len(content)})
            return storage_id
        except Exception as e:
            logger.error("s3_storage_store_error", extra={"doc_id": doc_id, "error": str(e)})
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



