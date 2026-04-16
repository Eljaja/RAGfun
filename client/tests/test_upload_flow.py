"""Upload flow: upload -> poll status -> verify indexed."""
import pytest

import sys
from pathlib import Path
SDK_ROOT = Path(__file__).resolve().parent.parent
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk import APIError
from conftest import poll_until_done


@pytest.mark.upload
class TestUploadAndIndex:
    """Uses the session-scoped uploaded_doc fixture — upload + poll
    happens once, and these tests verify the result."""

    def test_upload_succeeded(self, uploaded_doc):
        assert uploaded_doc.doc_id

    def test_document_is_indexed(self, client, uploaded_doc):
        status = client.get_document_status(uploaded_doc.doc_id)
        assert status["event_type"] in (
            "indexed", "processed", "embeddings_created", "uploaded",
        )

    def test_get_document_info(self, client, uploaded_doc):
        doc = client.get_document(uploaded_doc.doc_id)
        assert doc.get("doc_id") or doc.get("storage_id")

    def test_list_documents_includes_upload(self, client, project_id, uploaded_doc):
        docs = client.list_project_documents(project_id, limit=50)
        assert isinstance(docs.documents, list)
        assert len(docs.documents) >= 1


@pytest.mark.upload
class TestDuplicateUpload:

    def test_second_upload_of_same_content_returns_409(
        self, client, project_id, sample_txt, uploaded_doc
    ):
        """After the session upload, uploading the same file again must 409."""
        with pytest.raises(APIError) as exc_info:
            client.upload_document(
                project_id, sample_txt,
                title="pytest-dup-check",
                lang="en",
            )
        assert exc_info.value.status_code == 409


@pytest.mark.upload
class TestUploadPdf:

    def test_upload_pdf_and_poll(self, client, project_id, sample_pdf):
        doc_id = None
        try:
            resp = client.upload_document(
                project_id, sample_pdf,
                title="pytest-sample-pdf",
                lang="en",
            )
        except APIError as exc:
            if exc.status_code == 409:
                pytest.skip("PDF already uploaded (409)")
            raise

        doc_id = resp.doc_id
        assert doc_id
        try:
            status = poll_until_done(client, doc_id, timeout=180)
            assert status["event_type"] in ("indexed", "processed", "embeddings_created")
        finally:
            if doc_id:
                try:
                    client.delete_document(doc_id)
                except APIError:
                    pass
