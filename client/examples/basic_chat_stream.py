from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sdk import ChatStreamRequest, ClientAuth, RagGatewayClient  # noqa: E402


def main() -> int:
    gateway_url = os.environ.get("RAG_GATEWAY_URL", "http://202.181.159.221:8916").rstrip("/")
    bearer_token = os.environ.get("RAG_BEARER_TOKEN")

    print(f"Gateway URL: {gateway_url}")
    client = RagGatewayClient(
        base_url=gateway_url,
        auth=ClientAuth(bearer_token=bearer_token),
    )
    try:
        payload = ChatStreamRequest(
            query="What is the pineapple secret message? Return quote only.",
            include_sources=True,
        )
        for event in client.chat_stream(payload):
            print(event)
            if event.get("type") in {"done", "error"}:
                break
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
