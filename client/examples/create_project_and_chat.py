from __future__ import annotations

import os

from client import ClientAuth, RAGOpenAIClient


def main() -> None:
    base_url = os.getenv("RAG_GATEWAY_URL", "http://localhost:8917")
    api_key = os.environ["RAG_API_KEY"]

    with RAGOpenAIClient(base_url=base_url, auth=ClientAuth(api_key=api_key)) as client:
        project = client.projects.ensure(
            name="sdk-demo-project",
            description="Project created from SDK",
        )
        project_id = project["project_id"]
        print("project_id:", project_id)

        completion = client.chat.completions.create(
            project_id=project_id,
            messages=[{"role": "user", "content": "Что загружено в проект?"}],
            stream=False,
        )
        print("answer:", completion["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
