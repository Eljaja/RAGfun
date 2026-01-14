#!/usr/bin/env python3
"""
gate.py — учебный CLI для работы с gate (локально)

Цель: чтобы код было удобно читать “блоками по методам”.
Поэтому здесь:
- один класс GateClient;
- методы сгруппированы по HTTP-методам и путям;
- минимум “магии” в CLI.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


# =========================
# Helpers (общие утилиты)
# =========================

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _read_all(resp) -> bytes:
    return resp.read() or b""


def _bool_to_str(v: bool) -> str:
    return "true" if v else "false"


def _parse_csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    xs = [x.strip() for x in s.split(",") if x.strip()]
    return xs or None


# =========================
# HTTP Client (GateClient)
# =========================

@dataclass(frozen=True)
class _MultipartPart:
    name: str
    value: bytes
    filename: str | None = None
    content_type: str | None = None


class GateClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    # ---- низкоуровневые HTTP-хелперы ----
    def _request_json(self, method: str, path: str, *, body: Any | None = None, timeout_s: int = 60) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        req = Request(url, method=method, headers=headers, data=data)
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                raw = _read_all(resp)
                return json.loads(raw.decode("utf-8", errors="replace")) if raw else None
        except HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            text = raw.decode("utf-8", errors="replace") if raw else str(e)
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {text}") from e
        except URLError as e:
            raise RuntimeError(f"Network error: {e}") from e

    def _request_text(self, method: str, path: str, *, timeout_s: int = 60) -> str:
        url = f"{self.base_url}{path}"
        req = Request(url, method=method, headers={"Accept": "*/*"})
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                return _read_all(resp).decode("utf-8", errors="replace")
        except HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            text = raw.decode("utf-8", errors="replace") if raw else str(e)
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {text}") from e
        except URLError as e:
            raise RuntimeError(f"Network error: {e}") from e

    def _build_multipart(self, parts: list[_MultipartPart]) -> tuple[bytes, str]:
        boundary = f"----ragfun-{uuid.uuid4().hex}"
        crlf = b"\r\n"
        out: list[bytes] = []

        for p in parts:
            out.append(f"--{boundary}".encode("utf-8"))
            disp = f'Content-Disposition: form-data; name="{p.name}"'
            if p.filename is not None:
                disp += f'; filename="{p.filename}"'
            out.append(disp.encode("utf-8"))
            if p.filename is not None:
                out.append(f"Content-Type: {p.content_type or 'application/octet-stream'}".encode("utf-8"))
            out.append(b"")
            out.append(p.value)

        out.append(f"--{boundary}--".encode("utf-8"))
        out.append(b"")
        return crlf.join(out), boundary

    # =========================
    # GET /v1/*
    # =========================

    def healthz(self) -> Any:
        # GET /v1/healthz
        return self._request_json("GET", "/v1/healthz")

    def readyz(self) -> Any:
        # GET /v1/readyz
        return self._request_json("GET", "/v1/readyz")

    def version(self) -> Any:
        # GET /v1/version
        return self._request_json("GET", "/v1/version")

    def metrics(self) -> str:
        # GET /v1/metrics (text)
        return self._request_text("GET", "/v1/metrics")

    # =========================
    # POST /v1/chat*
    # =========================

    def chat(
        self,
        *,
        query: str,
        include_sources: bool = True,
        mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        # POST /v1/chat
        payload: dict[str, Any] = {"query": query, "include_sources": include_sources}
        if mode:
            payload["retrieval_mode"] = mode
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if rerank is not None:
            payload["rerank"] = bool(rerank)
        if filters:
            payload["filters"] = filters
        return self._request_json("POST", "/v1/chat", body=payload)

    def chat_stream(
        self,
        *,
        query: str,
        include_sources: bool = True,
        mode: str | None = None,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> Iterator[str]:
        # POST /v1/chat/stream (SSE)
        payload: dict[str, Any] = {"query": query, "include_sources": include_sources}
        if mode:
            payload["retrieval_mode"] = mode
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if rerank is not None:
            payload["rerank"] = bool(rerank)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            f"{self.base_url}/v1/chat/stream",
            method="POST",
            data=data,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )

        try:
            with urlopen(req, timeout=300) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    yield line.decode("utf-8", errors="replace")
        except HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            text = raw.decode("utf-8", errors="replace") if raw else str(e)
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {text}") from e
        except URLError as e:
            raise RuntimeError(f"Network error: {e}") from e

    # =========================
    # Documents (upload/list/status/delete)
    # =========================

    def upload(
        self,
        *,
        file_path: str,
        doc_id: str,
        title: str | None = None,
        uri: str | None = None,
        source: str | None = None,
        lang: str | None = None,
        tags: str | None = None,  # gate ожидает comma-separated string
        acl: str | None = None,  # gate ожидает comma-separated string
        tenant_id: str | None = None,
        project_id: str | None = None,
        refresh: bool = False,
    ) -> Any:
        # POST /v1/documents/upload (multipart/form-data)
        with open(file_path, "rb") as f:
            content = f.read()

        filename = os.path.basename(file_path) or "file"
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        def b(s: str) -> bytes:
            return s.encode("utf-8")

        parts: list[_MultipartPart] = [
            _MultipartPart("file", content, filename=filename, content_type=ctype),
            _MultipartPart("doc_id", b(doc_id)),
            _MultipartPart("refresh", b(_bool_to_str(refresh))),
        ]

        def add_opt(name: str, val: str | None) -> None:
            if val is not None and str(val).strip() != "":
                parts.append(_MultipartPart(name, b(str(val))))

        add_opt("title", title)
        add_opt("uri", uri)
        add_opt("source", source)
        add_opt("lang", lang)
        add_opt("tags", tags)
        add_opt("acl", acl)
        add_opt("tenant_id", tenant_id)
        add_opt("project_id", project_id)

        body, boundary = self._build_multipart(parts)
        req = Request(
            f"{self.base_url}/v1/documents/upload",
            method="POST",
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        try:
            with urlopen(req, timeout=120) as resp:
                raw = _read_all(resp)
                return json.loads(raw.decode("utf-8", errors="replace")) if raw else None
        except HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            text = raw.decode("utf-8", errors="replace") if raw else str(e)
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {text}") from e
        except URLError as e:
            raise RuntimeError(f"Network error: {e}") from e

    def list_docs(
        self,
        *,
        source: str | None = None,
        tags: str | None = None,
        lang: str | None = None,
        collections: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        # GET /v1/documents
        q: dict[str, str] = {"limit": str(limit), "offset": str(offset)}
        if source:
            q["source"] = source
        if tags:
            q["tags"] = tags
        if lang:
            q["lang"] = lang
        if collections:
            q["collections"] = collections
        return self._request_json("GET", "/v1/documents?" + urlencode(q))

    def doc_status(self, *, doc_id: str) -> Any:
        # GET /v1/documents/{doc_id}/status
        enc = quote(doc_id, safe="")
        return self._request_json("GET", f"/v1/documents/{enc}/status")

    def stats(
        self,
        *,
        source: str | None = None,
        tags: str | None = None,
        lang: str | None = None,
        collections: str | None = None,
        page_size: int = 500,
        max_docs: int = 200_000,
    ) -> Any:
        # GET /v1/documents/stats
        q: dict[str, str] = {"page_size": str(page_size), "max_docs": str(max_docs)}
        if source:
            q["source"] = source
        if tags:
            q["tags"] = tags
        if lang:
            q["lang"] = lang
        if collections:
            q["collections"] = collections
        return self._request_json("GET", "/v1/documents/stats?" + urlencode(q))

    def collections(self, *, tenant_id: str | None = None, limit: int = 1000) -> Any:
        # GET /v1/collections
        q: dict[str, str] = {"limit": str(limit)}
        if tenant_id:
            q["tenant_id"] = tenant_id
        return self._request_json("GET", "/v1/collections?" + urlencode(q))

    def delete_doc(self, *, doc_id: str) -> Any:
        # DELETE /v1/documents/{doc_id}
        enc = quote(doc_id, safe="")
        return self._request_json("DELETE", f"/v1/documents/{enc}")

    def delete_all(self, *, batch_size: int = 200, concurrency: int = 10, max_batches: int = 10_000) -> Any:
        # DELETE /v1/documents?confirm=true
        q = {
            "confirm": "true",
            "batch_size": str(batch_size),
            "concurrency": str(concurrency),
            "max_batches": str(max_batches),
        }
        return self._request_json("DELETE", "/v1/documents?" + urlencode(q))

    # =========================
    # Smoke scenario
    # =========================

    def smoke(self, *, file_path: str, doc_id: str | None = None, title: str | None = None) -> int:
        did = doc_id or f"doc-smoke-{int(time.time())}"
        eprint(f"GATE_URL={self.base_url}")
        eprint(f"file_path={file_path}")
        eprint(f"doc_id={did}")

        eprint("\n== readyz ==")
        self.readyz()
        eprint("ready: ok")

        eprint("\n== upload ==")
        try:
            r = self.upload(file_path=file_path, doc_id=did, title=title or "Smoke test doc")
            print(jdump(r))
        except Exception as e:
            eprint(f"upload: warning: {e}")

        eprint("\n== status ==")
        try:
            print(jdump(self.doc_status(doc_id=did)))
        except Exception as e:
            eprint(f"status: warning: {e}")

        eprint("\n== chat ==")
        ch = self.chat(
            query=f"Кратко объясни, что это за документ doc_id={did} (если он загружен) и что такое gate.",
            include_sources=True,
        )
        print(jdump(ch))

        eprint("\n== delete ==")
        try:
            print(jdump(self.delete_doc(doc_id=did)))
        except Exception as e:
            eprint(f"delete: warning: {e}")

        eprint("\nsmoke: done")
        return 0


# =========================
# CLI (минимальная обвязка)
# =========================

def _env_gate_url() -> str:
    return (os.environ.get("GATE_URL") or "http://localhost:8090").strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gate.py", description="Gate helper (локально)")
    p.add_argument("--gate-url", default=None, help="Default: env GATE_URL or http://localhost:8090")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("healthz", help="GET /v1/healthz")
    sub.add_parser("readyz", help="GET /v1/readyz")
    sub.add_parser("version", help="GET /v1/version")
    sub.add_parser("metrics", help="GET /v1/metrics (text)")

    sp = sub.add_parser("upload", help="POST /v1/documents/upload")
    sp.add_argument("file")
    sp.add_argument("doc_id")
    sp.add_argument("--title")
    sp.add_argument("--uri")
    sp.add_argument("--source")
    sp.add_argument("--lang")
    sp.add_argument("--tags", help="Comma-separated string (как ожидает gate)")
    sp.add_argument("--acl", help="Comma-separated string (как ожидает gate)")
    sp.add_argument("--tenant-id")
    sp.add_argument("--project-id")
    sp.add_argument("--refresh", action="store_true")

    sp = sub.add_parser("list-docs", help="GET /v1/documents")
    sp.add_argument("--source")
    sp.add_argument("--tags", help="Comma-separated")
    sp.add_argument("--lang")
    sp.add_argument("--collections", help="Comma-separated project_ids")
    sp.add_argument("--limit", type=int, default=100)
    sp.add_argument("--offset", type=int, default=0)

    sp = sub.add_parser("doc-status", help="GET /v1/documents/{doc_id}/status")
    sp.add_argument("doc_id")

    sp = sub.add_parser("stats", help="GET /v1/documents/stats")
    sp.add_argument("--source")
    sp.add_argument("--tags")
    sp.add_argument("--lang")
    sp.add_argument("--collections")
    sp.add_argument("--page-size", type=int, default=500)
    sp.add_argument("--max-docs", type=int, default=200_000)

    sp = sub.add_parser("collections", help="GET /v1/collections")
    sp.add_argument("--tenant-id")
    sp.add_argument("--limit", type=int, default=1000)

    sp = sub.add_parser("chat", help="POST /v1/chat")
    sp.add_argument("query")
    sp.add_argument("--no-sources", action="store_true")
    sp.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    sp.add_argument("--top-k", type=int)
    sp.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    sp.add_argument("--filter-source")
    sp.add_argument("--filter-lang")
    sp.add_argument("--filter-tags", help="Comma-separated")
    sp.add_argument("--filter-doc-ids", help="Comma-separated")
    sp.add_argument("--filter-tenant-id")
    sp.add_argument("--filter-project-id")
    sp.add_argument("--filter-project-ids", help="Comma-separated")

    sp = sub.add_parser("chat-stream", help="POST /v1/chat/stream (SSE)")
    sp.add_argument("query")
    sp.add_argument("--no-sources", action="store_true")
    sp.add_argument("--mode", choices=["bm25", "vector", "hybrid"])
    sp.add_argument("--top-k", type=int)
    sp.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)

    sp = sub.add_parser("delete-doc", help="DELETE /v1/documents/{doc_id}")
    sp.add_argument("doc_id")

    sp = sub.add_parser("delete-all", help="DELETE /v1/documents?confirm=true")
    sp.add_argument("--confirm", action="store_true", help="Safety flag (обязательно)")
    sp.add_argument("--batch-size", type=int, default=200)
    sp.add_argument("--concurrency", type=int, default=10)
    sp.add_argument("--max-batches", type=int, default=10_000)

    sp = sub.add_parser("smoke", help="readyz → upload → status → chat → delete")
    sp.add_argument("--file", default="./README.md")
    sp.add_argument("--doc-id", default=None)
    sp.add_argument("--title", default=None)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    base = (args.gate_url or _env_gate_url()).rstrip("/")
    c = GateClient(base)

    try:
        if args.cmd == "healthz":
            print(jdump(c.healthz()))
            return 0
        if args.cmd == "readyz":
            print(jdump(c.readyz()))
            return 0
        if args.cmd == "version":
            print(jdump(c.version()))
            return 0
        if args.cmd == "metrics":
            sys.stdout.write(c.metrics())
            return 0

        if args.cmd == "upload":
            r = c.upload(
                file_path=args.file,
                doc_id=args.doc_id,
                title=args.title,
                uri=args.uri,
                source=args.source,
                lang=args.lang,
                tags=args.tags,
                acl=args.acl,
                tenant_id=args.tenant_id,
                project_id=args.project_id,
                refresh=bool(args.refresh),
            )
            print(jdump(r))
            return 0

        if args.cmd == "list-docs":
            print(
                jdump(
                    c.list_docs(
                        source=args.source,
                        tags=args.tags,
                        lang=args.lang,
                        collections=args.collections,
                        limit=args.limit,
                        offset=args.offset,
                    )
                )
            )
            return 0

        if args.cmd == "doc-status":
            print(jdump(c.doc_status(doc_id=args.doc_id)))
            return 0

        if args.cmd == "stats":
            print(
                jdump(
                    c.stats(
                        source=args.source,
                        tags=args.tags,
                        lang=args.lang,
                        collections=args.collections,
                        page_size=args.page_size,
                        max_docs=args.max_docs,
                    )
                )
            )
            return 0

        if args.cmd == "collections":
            print(jdump(c.collections(tenant_id=args.tenant_id, limit=args.limit)))
            return 0

        if args.cmd == "chat":
            filters: dict[str, Any] = {}
            if args.filter_source:
                filters["source"] = args.filter_source
            if args.filter_lang:
                filters["lang"] = args.filter_lang
            if args.filter_tags:
                t = _parse_csv(args.filter_tags)
                if t:
                    filters["tags"] = t
            if args.filter_doc_ids:
                d = _parse_csv(args.filter_doc_ids)
                if d:
                    filters["doc_ids"] = d
            if args.filter_tenant_id:
                filters["tenant_id"] = args.filter_tenant_id
            if args.filter_project_id:
                filters["project_id"] = args.filter_project_id
            if args.filter_project_ids:
                pids = _parse_csv(args.filter_project_ids)
                if pids:
                    filters["project_ids"] = pids
            r = c.chat(
                query=args.query,
                include_sources=not bool(args.no_sources),
                mode=args.mode,
                top_k=args.top_k,
                rerank=args.rerank,
                filters=filters or None,
            )
            print(jdump(r))
            return 0

        if args.cmd == "chat-stream":
            for line in c.chat_stream(
                query=args.query,
                include_sources=not bool(args.no_sources),
                mode=args.mode,
                top_k=args.top_k,
                rerank=args.rerank,
            ):
                sys.stdout.write(line)
                sys.stdout.flush()
            return 0

        if args.cmd == "delete-doc":
            print(jdump(c.delete_doc(doc_id=args.doc_id)))
            return 0

        if args.cmd == "delete-all":
            if not args.confirm:
                raise RuntimeError("Refusing to delete all documents without --confirm")
            print(jdump(c.delete_all(batch_size=args.batch_size, concurrency=args.concurrency, max_batches=args.max_batches)))
            return 0

        if args.cmd == "smoke":
            return c.smoke(file_path=args.file, doc_id=args.doc_id, title=args.title)

        raise RuntimeError(f"Unknown command: {args.cmd}")
    except Exception as e:
        eprint(f"error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

