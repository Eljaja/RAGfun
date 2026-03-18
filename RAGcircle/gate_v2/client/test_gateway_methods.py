from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx

try:
    from .sdk import (
        APIError,
        AgentStreamRequest,
        ChatRequest,
        ChatStreamRequest,
        ClientAuth,
        ProjectCreateRequest,
        RagGatewayClient,
    )
except ImportError:  # pragma: no cover - direct script execution
    from sdk import (
        APIError,
        AgentStreamRequest,
        ChatRequest,
        ChatStreamRequest,
        ClientAuth,
        ProjectCreateRequest,
        RagGatewayClient,
    )


SECRET_FILE = Path(__file__).with_name("pineapple_secret_message.txt")
SECRET_TEXT = 'Here is a secret message about pineapples "they are Sentient"\n'


def ensure_secret_file() -> Path:
    SECRET_FILE.write_text(SECRET_TEXT, encoding="utf-8")
    return SECRET_FILE


def bearer_headers(token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        t = token.strip()
        headers["Authorization"] = t if t.lower().startswith("bearer ") else f"Bearer {t}"
    return headers


def ensure_project(client: RagGatewayClient, requested_project_id: str) -> str:
    try:
        p = client.get_project(requested_project_id).project
        return str(p.get("project_id") or requested_project_id)
    except APIError as exc:
        if exc.status_code != 404:
            raise
    created = client.create_project(
        ProjectCreateRequest(
            name=requested_project_id,
            description="gateway methods test project",
        )
    ).project
    return str(created.get("project_id") or requested_project_id)


def wait_for_status_event(client: RagGatewayClient, doc_id: str, timeout_s: int = 45, poll_s: int = 3) -> dict[str, Any] | None:
    deadline = time.time() + timeout_s
    last_status: dict[str, Any] | None = None
    while time.time() < deadline:
        try:
            status = client.get_document_status(doc_id)
            last_status = status
            et = str(status.get("event_type") or "").lower()
            if et in {"indexed", "processed", "embeddings_created"}:
                return status
        except Exception:
            pass
        time.sleep(poll_s)
    return last_status


def run_step(name: str, fn: Callable[[], Any], failures: list[str]) -> Any | None:
    print(f"\n=== {name} ===")
    try:
        result = fn()
        if result is not None:
            print(result)
        print("[PASS]")
        return result
    except Exception as exc:
        print(f"[FAIL] {exc}")
        failures.append(name)
        return None


def run_stream(name: str, stream_iter, failures: list[str], max_events: int = 40) -> None:
    print(f"\n=== {name} ===")
    try:
        count = 0
        for event in stream_iter:
            count += 1
            et = event.get("type")
            if et in {"token"}:
                continue
            print(event)
            if et in {"done", "error"}:
                break
            if count >= max_events:
                print("Reached max_events; stopping stream check.")
                break
        print("[PASS]")
    except Exception as exc:
        print(f"[FAIL] {exc}")
        failures.append(name)


def main() -> int:
    gateway_url = os.environ.get("RAG_GATEWAY_URL", "http://localhost:8916").rstrip("/")
    bearer_token = os.environ.get("RAG_BEARER_TOKEN")
    requested_project_id = os.environ.get("RAG_PROJECT_ID", "default")
    run_stream_checks = os.environ.get("RAG_RUN_STREAM_CHECKS", "false").lower() in {"1", "true", "yes"}
    cleanup_doc = os.environ.get("RAG_CLEANUP_DOC", "false").lower() in {"1", "true", "yes"}

    print(f"Gateway URL: {gateway_url}")
    print(f"Project ID: {requested_project_id}")
    print(f"Run stream checks: {run_stream_checks}")
    print(f"Cleanup doc: {cleanup_doc}")

    failures: list[str] = []

    # Raw health/readiness checks (not all are present in SDK).
    with httpx.Client(timeout=20.0, headers=bearer_headers(bearer_token)) as raw:
        run_step(
            "GET /public/health",
            lambda: raw.get(f"{gateway_url}/public/health").json(),
            failures,
        )
        run_step(
            "GET /storage-api/public/health",
            lambda: raw.get(f"{gateway_url}/storage-api/public/health").json(),
            failures,
        )
        run_step(
            "GET /api/v1/readyz",
            lambda: raw.get(f"{gateway_url}/api/v1/readyz").json(),
            failures,
        )
        run_step(
            "GET /agent-api/v1/readyz",
            lambda: raw.get(f"{gateway_url}/agent-api/v1/readyz").json(),
            failures,
        )

    secret_file = ensure_secret_file()
    uploaded_doc_id: str | None = None

    client = RagGatewayClient(
        base_url=gateway_url,
        auth=ClientAuth(bearer_token=bearer_token),
    )
    try:
        run_step("list_projects", lambda: client.list_projects().model_dump(), failures)

        project_id = run_step(
            "ensure_project",
            lambda: ensure_project(client, requested_project_id),
            failures,
        ) or requested_project_id

        run_step("get_project", lambda: client.get_project(project_id).model_dump(), failures)
        run_step(
            "list_project_documents",
            lambda: client.list_project_documents(project_id, limit=5, offset=0).model_dump(),
            failures,
        )

        upload = run_step(
            "upload_document",
            lambda: client.upload_document(
                project_id,
                secret_file,
                title="pineapple secret note",
                source="gateway-methods-test",
                lang="en",
            ).model_dump(),
            failures,
        )
        if isinstance(upload, dict):
            uploaded_doc_id = str(upload.get("doc_id") or "")

        if uploaded_doc_id:
            run_step(
                "wait_for_status_event",
                lambda: wait_for_status_event(client, uploaded_doc_id) or {"status": "no_index_event_yet"},
                failures,
            )
            run_step("get_document", lambda: client.get_document(uploaded_doc_id), failures)
            run_step("get_document_status", lambda: client.get_document_status(uploaded_doc_id), failures)

        run_step(
            "chat (/api/v1/chat)",
            lambda: client.chat(
                ChatRequest(
                    query='What is the secret message about pineapples? Return the exact quote only.',
                    include_sources=True,
                    filters={"project_ids": [project_id]},
                ),
            ).model_dump(),
            failures,
        )

        if run_stream_checks:
            run_stream(
                "chat_stream (/api/v1/chat/stream)",
                client.chat_stream(
                    ChatStreamRequest(
                        query='Repeat the quote about sentient pineapples exactly.',
                        include_sources=True,
                        filters={"project_ids": [project_id]},
                    )
                ),
                failures,
            )
            run_stream(
                "agent_stream (/agent-api/v1/agent/stream)",
                client.agent_stream(
                    AgentStreamRequest(
                        query='Find the pineapple secret quote and return it verbatim.',
                        include_sources=True,
                        filters={"project_ids": [project_id]},
                    )
                ),
                failures,
            )

        if cleanup_doc and uploaded_doc_id:
            run_step("delete_document", lambda: client.delete_document(uploaded_doc_id), failures)
    finally:
        client.close()

    print("\n=== SUMMARY ===")
    if failures:
        print(f"Failed steps: {len(failures)}")
        for name in failures:
            print(f"- {name}")
        return 1
    print("All steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

