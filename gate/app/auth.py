from fastapi import HTTPException, Request
from returns.maybe import Maybe, Some, Nothing
from returns.result import Result, Success, Failure
from returns.pipeline import flow
from typing import Optional, Callable, Awaitable, TypeVar
from dataclasses import dataclass, asdict
import hashlib
import json
import contextlib
import os

import httpx

from app.config import load_settings#Settings

T = TypeVar('T')

s = load_settings()
AUTH_SERVER = s.auth_server   
ADMIN_SECRET_TOKEN = s.admin_secret_token 


@dataclass
class UserCreds:
    user_id: str
    limits: dict


# ═══════════════════════════════════════════════════════════════════════════════
# Redis cache for auth tokens
# ═══════════════════════════════════════════════════════════════════════════════

# TODO: too coupled with the data, need some decoupling 
class RedisTokenCache:
    """Simple Redis-backed token cache for UserCreds."""

    def __init__(self, redis_client, ttl_seconds: int = 300):
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    def _key(self, token: str) -> str:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return f"auth:token:{token_hash}"

    async def get(self, token: str) -> Optional[UserCreds]:
        if not self.redis_client or not token:
            return None
        
        cached = await self.redis_client.get(self._key(token))
        if cached:
            return UserCreds(**json.loads(cached))
        return None

    async def set(self, token: str, value: UserCreds) -> None:
        if not self.redis_client or not token:
            return
        
        await self.redis_client.set(
            self._key(token),
            json.dumps(asdict(value)),
            ex=self.ttl_seconds,
        )


# the result of reading some stuff about monads 
# very exotic, def not mainstream in my code right now
def parse_token_functional(header: Optional[str]) -> Result[str, str]:
    """
    Secure token validation with functional approach.
    
    Validates:
    - Bearer scheme present
    - sk- prefix
    - Length 30-50 chars
    - ASCII printable only (no unicode tricks)
    - Alphanumeric + hyphen/underscore only
    - No null bytes or control chars
    """
    
    def extract_bearer(h: str) -> Result[str, str]:
        parts = h.split(" ", 1)
        if len(parts) != 2:
            return Failure("invalid_schema")
        scheme, token = parts
        if scheme.lower() != "bearer":
            return Failure("invalid_schema_no_bearer")
        return Success(token)
    
    def check_prefix(t: str) -> Result[str, str]:
        return Success(t) if t.startswith("sk-") else Failure("no_prefix")
    
    def check_length(t: str) -> Result[str, str]:
        return Success(t) if 30 <= len(t) <= 50 else Failure("bad_length")
    
    def check_ascii_only(t: str) -> Result[str, str]:
        """Reject non-ASCII (prevents unicode homoglyph attacks)"""
        return Success(t) if t.isascii() else Failure("non_ascii_chars")
    
    def check_no_control_chars(t: str) -> Result[str, str]:
        """Reject null bytes, tabs, newlines, etc."""
        has_control = any(ord(c) < 32 or ord(c) == 127 for c in t)
        return Failure("control_chars_found") if has_control else Success(t)
    
    def check_allowed_chars(t: str) -> Result[str, str]:
        """Only alphanumeric + hyphen + underscore allowed"""
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        invalid = set(t) - allowed
        return Failure(f"invalid_chars: {invalid}") if invalid else Success(t)
    
    def check_no_sequences(t: str) -> Result[str, str]:
        """Block suspicious patterns (path traversal, null injection)"""
        suspicious = ["../", "..\\", "\x00", "%00", "<", ">", "'", '"']
        for seq in suspicious:
            if seq in t:
                return Failure("suspicious_sequence")
        return Success(t)
    
    return (
        Maybe.from_optional(header)          # Optional[str] -> Maybe[str]
        .map(str.strip)                      # Maybe[str] -> Maybe[str]
        .map(Success)                        # Maybe[str] -> Maybe[Result[str, str]]
        .value_or(Failure("missing_header")) # Maybe[Result] -> Result[str, str]
        .bind(extract_bearer)
        .bind(check_ascii_only)              # security: no unicode
        .bind(check_no_control_chars)        # security: no \x00, \n, etc
        .bind(check_allowed_chars)           # security: whitelist chars
        .bind(check_no_sequences)            # security: no injection patterns
        .bind(check_prefix)                  # format: sk- prefix
        .bind(check_length)                  # format: 30-50 chars
    )







# ═══════════════════════════════════════════════════════════════════════════════
# Async cache-aside combinator
# ═══════════════════════════════════════════════════════════════════════════════

async def cache_aside(
    key: str,
    cache_get: Callable[[str], Awaitable[T | None]],
    fetcher: Callable[[], Awaitable[Result[T, str]]],
    cache_set: Callable[[str, T], Awaitable[None]]
) -> Result[T, str]:
    """
    Check cache → on miss: fetch + store → return Result
    
    Pure functional: no side effects leak, all dependencies injected
    """
    cached = await cache_get(key)
    if cached is not None:
        return Success(cached)
    
    result = await fetcher()
    match result:
        case Success(value):
            await cache_set(key, value)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoint caller closure (async!)
# ═══════════════════════════════════════════════════════════════════════════════

def call_endpoint(url: str, method: str = "GET", headers: dict = None):
    """
    Closure that captures URL/method, returns async caller
    
    Usage:
        fetch_user = call_endpoint("https://api/user", "GET")
        response = await fetch_user(http_client)
    """
    async def call(client) -> Result[dict, str]:
        try:
            response = await client.request(method, url, headers=headers)
            response.raise_for_status()
            return Success(response.json())
        except Exception as e:
            return Failure(f"request_failed: {e}")
    return call


# ═══════════════════════════════════════════════════════════════════════════════
# Main auth dependency
# ═══════════════════════════════════════════════════════════════════════════════

async def authenticated(request: Request) -> UserCreds:
    """FastAPI dependency: validates token, returns user creds (cached)"""
    
    # cache = request.app.state.cache
    #http_client = request.app.state.http_client
    # http_client = httpx.HttpClient()
    
    # 1. Parse & validate token (functional)
    auth_header = request.headers.get("authorization")
    token_result = parse_token_functional(auth_header)
    
    match token_result:
        case Failure(error):
            raise HTTPException(status_code=401, detail=error)
        case Success(token):
            pass  # continue with valid token
    
    token = token_result.unwrap()
    
    # 2. Build the fetcher closure (captures token + client)
    async def fetch_from_auth() -> Result[UserCreds, str]:
        async with httpx.AsyncClient(timeout=30) as http_client:
            fetch = call_endpoint(
                url=f"{AUTH_SERVER}/verify-token?token={token}",
                method="GET", 
                headers = {"X-Cudo-Admin": f"Bearer {ADMIN_SECRET_TOKEN}"},
            )
            result = await fetch(http_client)
            return result.map(lambda data: UserCreds(
                user_id=data.get("user_id", ""),
                limits=data.get("limits", {})
            ))
    
    # # 3. Cache-aside: check cache, fallback to fetch + cache
    # creds_result = await cache_aside(
    #     key=token,
    #     cache_get=cache.get,
    #     fetcher=fetch_from_auth,
    #     cache_set=cache.set
    # )
    creds_result = await fetch_from_auth()
    
    match creds_result:
        case Success(creds):
            return creds
        case Failure(error):
            raise HTTPException(status_code=503, detail=error)




# ═══════════════════════════════════════════════════════════════════════════════
# Notes on closures:
# ═══════════════════════════════════════════════════════════════════════════════
# Closures DO shine for tasks like "call some endpoint" — nice syntax!
# They're like lightweight anonymous classes that capture context.
# Very pythonic when used for dependency injection & partial application.