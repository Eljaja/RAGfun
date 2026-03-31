from __future__ import annotations

import os

from client import ClientAuth, RAGOpenAIClient


def main() -> None:
    base_url = os.getenv("RAG_GATEWAY_URL", "http://localhost:8917")
    api_key = os.environ["RAG_API_KEY"]

    with RAGOpenAIClient(base_url=base_url, auth=ClientAuth(api_key=api_key)) as client:
        project = client.projects.ensure(
            name="sdk-demo-stream",
            description="Streaming demo project",
        )
        project_id = project["project_id"]
        print("project_id:", project_id)

        events = client.chat.completions.create(
            project_id=project_id,
            messages=[{"role": "user", "content": "Сформулируй 3 ключевых тезиса по проекту"}],
            stream=True,
        )

        print("answer: ", end="", flush=True)
        for chunk in events:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
