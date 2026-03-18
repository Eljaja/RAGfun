from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sdk import APIError, ClientAuth, RagGatewayClient  # noqa: E402

SECRET_FILE = ROOT_DIR / "pineapple_secret_message.txt"
SECRET_TEXT = 'Here is a secret message about pineapples "they are Sentient"\n'


def ensure_secret_file() -> Path:
    SECRET_FILE.write_text(SECRET_TEXT, encoding="utf-8")
    return SECRET_FILE


def wait_for_status_event(client: RagGatewayClient, doc_id: str, timeout_s: int = 45, poll_s: int = 3) -> dict | None:
    deadline = time.time() + timeout_s
    last_status: dict | None = None
    while time.time() < deadline:
        try:
            status = client.get_document_status(doc_id)
            last_status = status
            event_type = str(status.get("event_type") or "").lower()
            if event_type in {"indexed", "processed", "embeddings_created"}:
                return status
        except Exception:
            pass
        time.sleep(poll_s)
    return last_status


def main() -> int:
    gateway_url = os.environ.get("RAG_GATEWAY_URL", "http://202.181.159.221:8916").rstrip("/")
    bearer_token = os.environ.get("RAG_BEARER_TOKEN")

    print(f"Gateway URL: {gateway_url}")
    client = RagGatewayClient(
        base_url=gateway_url,
        auth=ClientAuth(bearer_token=bearer_token),
    )
    try:
        project_id = client.default_project_id
        print(f"Project: {project_id}")
        print("Project details:", client.get_project(project_id).model_dump())

        file_path = ensure_secret_file()
        uploaded_doc_id: str | None = None

        try:
            upload = client.upload_document(
                project_id,
                file_path,
                title="pineapple secret note",
                source="sdk-example-upload",
                lang="en",
            ).model_dump()
            uploaded_doc_id = str(upload.get("doc_id") or "")
            print("Upload:", upload)
        except APIError as exc:
            if exc.status_code == 409:
                print("Upload returned 409 conflict (duplicate). Continuing.")
            else:
                raise

        docs = client.list_project_documents(project_id, limit=10, offset=0).model_dump()
        print("Documents:", docs)

        if not uploaded_doc_id and docs.get("documents"):
            uploaded_doc_id = str(docs["documents"][0].get("doc_id") or "")

        if uploaded_doc_id:
            print(f"Checking status for doc_id={uploaded_doc_id}")
            status_event = wait_for_status_event(client, uploaded_doc_id)
            print("Polled status event:", status_event)
            print("Latest status:", client.get_document_status(uploaded_doc_id))
        else:
            print("No document id available to check status.")
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
