from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sdk import AgentStreamRequest, ChatStreamRequest, ClientAuth, RagGatewayClient  # noqa: E402


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


def assert_stream_completes(
    stream_iter: Iterator[dict[str, Any]],
    *,
    max_events: int,
    print_limit: int = 10,
) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    token_events = 0
    last_done: dict[str, Any] | None = None

    for idx, event in enumerate(stream_iter, start=1):
        et = str(event.get("type") or "unknown")
        if et == "token":
            token_events += 1
        else:
            event_counts[et] += 1
            if sum(event_counts.values()) <= print_limit:
                print(event)

        if et == "error":
            raise RuntimeError(str(event.get("error") or "stream returned error event"))
        if et == "done":
            last_done = event
            break
        if idx >= max_events:
            raise RuntimeError(f"stream did not reach 'done' within {max_events} events")

    if last_done is None:
        raise RuntimeError("stream ended without a 'done' event")

    answer = str(last_done.get("answer") or "").strip()
    if not answer:
        raise RuntimeError("done event has empty answer")

    return {
        "done_answer_preview": answer[:120],
        "token_events": token_events,
        "non_token_event_counts": dict(event_counts),
    }


def main() -> int:
    gateway_url = os.environ.get("RAG_GATEWAY_URL", "http://202.181.159.221:8916").rstrip("/")
    bearer_token = os.environ.get("RAG_BEARER_TOKEN")
    max_events = int(os.environ.get("RAG_STREAM_MAX_EVENTS", "500"))

    print(f"Gateway URL: {gateway_url}")
    print(f"Max stream events: {max_events}")

    failures: list[str] = []

    client = RagGatewayClient(base_url=gateway_url, auth=ClientAuth(bearer_token=bearer_token))
    try:
        run_step("get_project(default)", lambda: client.get_project().model_dump(), failures)

        run_step(
            "chat_stream (/api/v1/chat/stream)",
            lambda: assert_stream_completes(
                client.chat_stream(
                    ChatStreamRequest(
                        query='What is the pineapple secret message? Return quote only.',
                        include_sources=True,
                        filters={"project_ids": [client.default_project_id]},
                    )
                ),
                max_events=max_events,
            ),
            failures,
        )

        run_step(
            "agent_stream (/agent-api/v1/agent/stream)",
            lambda: assert_stream_completes(
                client.agent_stream(
                    AgentStreamRequest(
                        query='What is the pineapple secret message? Return quote only.',
                        include_sources=True,
                        filters={"project_ids": [client.default_project_id]},
                    )
                ),
                max_events=max_events,
            ),
            failures,
        )
    finally:
        client.close()

    print("\n=== SUMMARY ===")
    if failures:
        print(f"Failed steps: {len(failures)}")
        for name in failures:
            print(f"- {name}")
        return 1

    print("Streaming endpoint checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
