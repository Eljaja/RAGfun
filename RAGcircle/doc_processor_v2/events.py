from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote

from errors import NonRetryableError


# Supported file extensions for processing
SUPPORTED_EXTENSIONS = {
    ".pdf", ".doc", ".docx",  # Documents
    ".txt", ".md",            # Plain text
    ".html", ".htm", ".xhtml",  # HTML
    ".xml",                    # XML
    ".xlsx",                   # Spreadsheets
}


def is_supported_file(filename: str) -> bool:
    name = (filename or "").lower()
    return any(name.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def decode_s3_key(key: str) -> str:
    # RustFS/S3 events sometimes URL-encode keys.
    return unquote(key or "")


@dataclass(frozen=True)
class S3EventInfo:
    bucket: str
    key: str
    event_name: str

    @property
    def decoded_key(self) -> str:
        return decode_s3_key(self.key)


def extract_s3_event_info(event: dict) -> S3EventInfo:
    """
    Extract bucket, key, and event type from S3 event format.
    Supports both AWS S3 format and RustFS format.

    Raises NonRetryableError for payloads that don't match expected shapes.
    """
    try:
        # RustFS/AWS-style: Records[0]
        records = event.get("Records", [])
        if records:
            record = records[0]

            # RustFS format: bucket_name, object_name, event_name at record level
            bucket = record.get("bucket_name")
            key = record.get("object_name")
            event_name = record.get("event_name") or record.get("eventName") or ""
            if bucket and key:
                return S3EventInfo(bucket=bucket, key=key, event_name=str(event_name))

            # AWS S3 format: nested in s3 object
            s3_info = record.get("s3", {})
            if s3_info:
                event_name = record.get("eventName", "") or ""
                bucket = s3_info.get("bucket", {}).get("name")
                key = s3_info.get("object", {}).get("key")
                if bucket and key:
                    return S3EventInfo(bucket=bucket, key=key, event_name=str(event_name))

        # Some RustFS variants: top-level EventName + Key="bucket/object"
        event_name = event.get("EventName") or event.get("eventName") or ""
        key_path = event.get("Key")
        if key_path and "/" in key_path:
            bucket, key = key_path.split("/", 1)
            if bucket and key:
                return S3EventInfo(bucket=bucket, key=key, event_name=str(event_name))

        raise NonRetryableError("could_not_extract_bucket_or_key")
    except NonRetryableError:
        raise
    except Exception as e:
        raise NonRetryableError(f"failed_to_parse_event_structure:{type(e).__name__}") from e


def is_object_created(event_name: str) -> bool:
    return "ObjectCreated" in (event_name or "")


def is_object_removed(event_name: str) -> bool:
    return "ObjectRemoved" in (event_name or "")

