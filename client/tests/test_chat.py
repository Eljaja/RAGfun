"""Chat tests — run after upload so the project has indexed documents.

Every test here takes ``uploaded_doc`` as a fixture parameter, which
guarantees pytest won't run them until the document is uploaded and indexed.
"""
import pytest
import sys
from pathlib import Path
SDK_ROOT = Path(__file__).resolve().parent.parent
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk import AgentRequest, SimpleChatRequest


@pytest.mark.chat
class TestAgentChat:

    def test_returns_answer(self, client, project_id, uploaded_doc):
        resp = client.agent_chat(AgentRequest(
            project_id=project_id,
            query="What is this document about?",
        ))
        assert resp.answer, f"Empty answer: {resp}"
        assert len(resp.answer) > 5

    def test_with_mode(self, client, project_id, uploaded_doc):
        resp = client.agent_chat(AgentRequest(
            project_id=project_id,
            query="Summarize the document in one sentence.",
            mode="minimal",
        ))
        assert resp.answer

    def test_with_history(self, client, project_id, uploaded_doc):
        resp = client.agent_chat(AgentRequest(
            project_id=project_id,
            query="Tell me more about that.",
            history=[
                {"role": "user", "content": "What is this document about?"},
                {"role": "assistant", "content": "It is about tropical fruits."},
            ],
        ))
        assert resp.answer


@pytest.mark.chat
class TestAgentChatStream:

    def test_stream_produces_events(self, client, project_id, uploaded_doc):
        events = list(client.agent_chat_stream(AgentRequest(
            project_id=project_id,
            query="What is this document about?",
        )))
        assert len(events) > 0


@pytest.mark.chat
class TestSimpleChat:

    def test_returns_answer(self, client, project_id, uploaded_doc):
        resp = client.simple_chat(SimpleChatRequest(
            project_id=project_id,
            query="What is this document about?",
        ))
        assert resp.answer, f"Empty answer: {resp}"

    def test_fast_preset(self, client, project_id, uploaded_doc):
        resp = client.simple_chat(SimpleChatRequest(
            project_id=project_id,
            query="Summarize the document briefly.",
            preset="fast",
            rerank=False,
        ))
        assert resp.answer


@pytest.mark.chat
class TestSimpleChatStream:

    def test_stream_produces_events(self, client, project_id, uploaded_doc):
        events = list(client.simple_chat_stream(SimpleChatRequest(
            project_id=project_id,
            query="What is this document about?",
        )))
        assert len(events) > 0
