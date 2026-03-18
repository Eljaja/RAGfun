from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sdk import ChatRequest, ClientAuth, ProjectCreateRequest, RagGatewayClient, SDKError  # noqa: E402


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


def expect_sdk_error(name: str, fn: Callable[[], Any], failures: list[str]) -> None:
    print(f"\n=== {name} ===")
    try:
        fn()
    except SDKError as exc:
        print(f"[PASS] blocked: {exc}")
        return
    except Exception as exc:
        print(f"[FAIL] unexpected exception type: {type(exc).__name__}: {exc}")
        failures.append(name)
        return

    print("[FAIL] expected SDKError but call succeeded")
    failures.append(name)


def main() -> int:
    gateway_url = os.environ.get("RAG_GATEWAY_URL", "http://202.181.159.221:8916").rstrip("/")
    bearer_token = os.environ.get("RAG_BEARER_TOKEN")

    print(f"Gateway URL: {gateway_url}")
    failures: list[str] = []

    client = RagGatewayClient(base_url=gateway_url, auth=ClientAuth(bearer_token=bearer_token))
    try:
        run_step("get_project(default)", lambda: client.get_project().model_dump(), failures)
        run_step("list_project_documents(default)", lambda: client.list_project_documents(limit=5, offset=0).model_dump(), failures)
        run_step(
            "chat(default)",
            lambda: client.chat(
                ChatRequest(
                    query='What is the pineapple secret message? Return quote only.',
                    include_sources=True,
                )
            ).model_dump(),
            failures,
        )

        expect_sdk_error("create_project blocked", lambda: client.create_project(ProjectCreateRequest(name="other-project")), failures)
        expect_sdk_error("delete_project blocked", lambda: client.delete_project("default"), failures)
        expect_sdk_error("get_project(non-default) blocked", lambda: client.get_project("other-project"), failures)
        expect_sdk_error(
            "list_project_documents(non-default) blocked",
            lambda: client.list_project_documents("other-project", limit=1, offset=0),
            failures,
        )
        expect_sdk_error(
            "upload_document(non-default) blocked",
            lambda: client.upload_document("other-project", Path(__file__)),
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

    print("Default-project policy checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
