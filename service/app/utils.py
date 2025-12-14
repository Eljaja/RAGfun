from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def redact_uri(uri: str | None, mode: str) -> str | None:
    if uri is None:
        return None
    if mode == "none":
        return uri
    parts = urlsplit(uri)
    if mode == "strip_query":
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    if mode == "strip_all":
        # keep only scheme + netloc
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    return uri


