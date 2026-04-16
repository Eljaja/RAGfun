"""Smoke test — replicates the manual Swagger flow with visible output.

Run with:
    pytest tests/test_smoke.py -s -v
    pytest -m smoke -s -v
"""
import pytest
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parent.parent
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk import AgentRequest, SimpleChatRequest

SEP = "=" * 60


@pytest.mark.smoke
def test_full_pipeline(client, project_id, uploaded_doc):
    """health → project → upload → status → agent_chat → simple_chat"""

    print(f"\n{SEP}")
    print("SMOKE TEST — full pipeline")
    print(SEP)

    # -- health --
    health = client.health()
    print(f"\n[health]  {health}")

    # -- project --
    project = client.get_project(project_id)
    proj = project.project
    print(f"\n[project] id={proj.get('project_id')}  name={proj.get('name')}")

    # -- uploaded doc status --
    status = client.get_document_status(uploaded_doc.doc_id)
    print(f"\n[upload]  doc_id={uploaded_doc.doc_id}")
    print(f"[status]  event_type={status.get('event_type')}")

    # -- agent chat --
    query_1 = "What is this document about?"
    resp = client.agent_chat(AgentRequest(
        project_id=project_id,
        query=query_1,
    ))
    print(f"\n[agent_chat]")
    print(f"  query:   {query_1}")
    print(f"  answer:  {resp.answer}")
    print(f"  mode:    {resp.mode}")
    print(f"  sources: {len(resp.sources)} chunks")
    assert resp.answer

    # -- simple chat --
    query_2 = "Summarize the document in one sentence."
    resp2 = client.simple_chat(SimpleChatRequest(
        project_id=project_id,
        query=query_2,
    ))
    print(f"\n[simple_chat]")
    print(f"  query:   {query_2}")
    print(f"  answer:  {resp2.answer}")
    print(f"  sources: {resp2.sources}")
    assert resp2.answer

    print(f"\n{SEP}")
    print("SMOKE OK")
    print(SEP)
