from __future__ import annotations

import os

try:
    from .sdk import ChatRequest, ClientAuth, RagGatewayClient
except ImportError:  # pragma: no cover - direct script execution
    from sdk import ChatRequest, ClientAuth, RagGatewayClient


def main() -> int:
    gateway_url = os.environ.get("RAG_GATEWAY_URL", "http://localhost:8916").rstrip("/")
    bearer_token = os.environ.get("RAG_BEARER_TOKEN")

    print(f"Gateway URL: {gateway_url}")
    client = RagGatewayClient(
        base_url=gateway_url,
        auth=ClientAuth(bearer_token=bearer_token),
    )

    try:
        print("\n=== list_projects (filtered to default) ===")
        print(client.list_projects().model_dump())

        print("\n=== get_project(default) ===")
        project = client.get_project().model_dump()
        print(project)

        print("\n=== list_project_documents(default) ===")
        docs = client.list_project_documents(limit=10, offset=0).model_dump()
        print(docs)

        print("\n=== chat (/api/v1/chat) ===")
        answer = client.chat(
            ChatRequest(
                query='What is the pineapple secret message? Return quote only.',
                include_sources=True,
            )
        ).model_dump()
        print(answer)
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
