"""Shared fixtures for integration tests.

Only one env var matters:
    RAG_BEARER_TOKEN  – sk-… token (defaults to "stub" for stub-auth mode)

Optional overrides:
    RAG_GATEWAY_URL   – gate base URL (default http://localhost:8918)

Everything else (project, documents) is auto-managed by fixtures.
"""
from __future__ import annotations

import os
import sys
import time
import textwrap
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).resolve().parent.parent
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk import APIError, ClientAuth, ProjectCreateRequest, RagGatewayClient

TERMINAL_OK = {"indexed", "processed", "embeddings_created"}
TERMINAL_ERR = {"error_processing", "deleted"}

PYTEST_PROJECT_NAME = "pytest-auto"


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: minimal end-to-end pipeline check")
    config.addinivalue_line("markers", "upload: document upload and indexing")
    config.addinivalue_line("markers", "chat: chat endpoint tests")
    config.addinivalue_line("markers", "crud: project CRUD lifecycle")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gateway_url():
    return os.environ.get("RAG_GATEWAY_URL", "http://localhost:8918")


@pytest.fixture(scope="session")
def bearer_token():
    return os.environ.get("RAG_BEARER_TOKEN", "stub")


@pytest.fixture(scope="session")
def client(gateway_url, bearer_token):
    c = RagGatewayClient(
        base_url=gateway_url,
        auth=ClientAuth(bearer_token=bearer_token),
        timeout_s=120.0,
    )
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Project — find existing or create one
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_id(client):
    """Return a usable project_id.

    Strategy:
    1. List projects — if any exist, use the first one.
    2. Otherwise create a temporary project.
    """
    resp = client.list_projects()
    if resp.projects:
        pid = resp.projects[0].get("project_id") or resp.projects[0].get("name")
        return pid

    created = client.create_project(ProjectCreateRequest(
        name=PYTEST_PROJECT_NAME,
        description="Auto-created by test suite",
    ))
    return created.project["project_id"]


# ---------------------------------------------------------------------------
# Sample files
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_txt(tmp_path_factory):
    p = tmp_path_factory.mktemp("fixtures") / "sample.txt"
    p.write_text(
        textwrap.dedent("""\
            Pineapples are tropical fruits native to South America.
            They are rich in vitamins C and B6, manganese, and bromelain enzymes.
            A pineapple plant produces only one fruit per growth cycle.
            The word "pineapple" comes from its resemblance to a pine cone.
        """),
        encoding="utf-8",
    )
    return p


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    p = tmp_path_factory.mktemp("fixtures") / "sample.pdf"
    content = (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Test document) Tj ET\n"
        b"endstream endobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000210 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\n"
        b"startxref\n306\n%%EOF\n"
    )
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# Uploaded document — upload once, poll until indexed, share across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def uploaded_doc(client, project_id, sample_txt):
    """Upload a document, wait for indexing, yield it, then delete on teardown.

    Session-scoped: runs once, result shared by all tests that need it.
    On teardown the uploaded document is deleted so the project stays clean.
    """
    we_uploaded = False
    doc_id = None

    try:
        resp = client.upload_document(
            project_id, sample_txt,
            title="pytest-session-doc",
            lang="en",
        )
        we_uploaded = True
        doc_id = resp.doc_id
    except APIError as exc:
        if exc.status_code == 409:
            docs = client.list_project_documents(project_id, limit=1)
            if docs.documents:
                doc_id = docs.documents[0].get("doc_id")
                resp = type("ReusedUpload", (), {
                    "doc_id": doc_id,
                    "project_id": project_id,
                    "size": None,
                })()
            else:
                pytest.fail("409 duplicate but no documents found to reuse")
        else:
            raise

    if we_uploaded:
        poll_until_done(client, resp.doc_id, timeout=180)

    yield resp

    if doc_id:
        try:
            client.delete_document(doc_id)
        except APIError:
            pass


# ---------------------------------------------------------------------------
# Helpers (importable by test files)
# ---------------------------------------------------------------------------

def poll_until_done(
    client: RagGatewayClient,
    doc_id: str,
    *,
    timeout: float = 120,
    interval: float = 3,
) -> dict:
    """Poll document status until a terminal event or timeout."""
    deadline = time.monotonic() + timeout
    last_event = "unknown"
    while time.monotonic() < deadline:
        status = client.get_document_status(doc_id)
        last_event = status.get("event_type", "unknown")
        if last_event in TERMINAL_OK:
            return status
        if last_event in TERMINAL_ERR:
            pytest.fail(
                f"Document {doc_id} processing failed: {last_event}\n{status}"
            )
        time.sleep(interval)
    pytest.fail(
        f"Document {doc_id} timed out after {timeout}s (last event: {last_event})"
    )
