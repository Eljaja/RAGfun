"""Upload a file, poll its processing status, then ask a question about it.

Usage:
    RAG_GATEWAY_URL=http://localhost:8918 \
    RAG_BEARER_TOKEN=sk-... \
    RAG_PROJECT_ID=default \
    python examples/upload_and_document_status.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sdk import (  # noqa: E402
    AgentRequest,
    APIError,
    ClientAuth,
    RagGatewayClient,
)

SECRET_FILE = ROOT_DIR / "pineapple_secret_message.txt"
SECRET_TEXT = 'Here is a secret message about pineapples "they are Sentient"\n'


def main() -> int:
    gateway_url = os.environ.get("RAG_GATEWAY_URL", "http://localhost:8918").rstrip("/")
    bearer_token = os.environ.get("RAG_BEARER_TOKEN")
    project_id = os.environ.get("RAG_PROJECT_ID", "default")

    SECRET_FILE.write_text(SECRET_TEXT, encoding="utf-8")

    client = RagGatewayClient(
        base_url=gateway_url,
        auth=ClientAuth(bearer_token=bearer_token),
    )
    try:
        print(f"Project: {project_id}")
        print("Details:", client.get_project(project_id).model_dump())

        doc_id: str | None = None
        try:
            resp = client.upload_document(
                project_id,
                SECRET_FILE,
                title="pineapple secret note",
                lang="en",
            )
            doc_id = resp.doc_id
            print(f"Uploaded: doc_id={doc_id}")
        except APIError as exc:
            if exc.status_code == 409:
                print("Duplicate upload (409) — continuing.")
            else:
                raise

        if not doc_id:
            docs = client.list_project_documents(project_id, limit=5)
            if docs.documents:
                doc_id = docs.documents[0].get("doc_id")

        if doc_id:
            deadline = time.time() + 60
            while time.time() < deadline:
                status = client.get_document_status(doc_id)
                event = status.get("event_type", "unknown")
                print(f"  status: {event}")
                if event in ("indexed", "processed", "embeddings_created", "error_processing"):
                    break
                time.sleep(3)

        answer = client.agent_chat(
            AgentRequest(
                project_id=project_id,
                query="What is the secret message about pineapples? Return exact quote.",
            )
        )
        print(f"Answer: {answer.answer}")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
