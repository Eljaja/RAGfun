from __future__ import annotations

import os

from client import ClientAuth, RAGOpenAIClient


def main() -> None:
    base_url = os.getenv("RAG_GATEWAY_URL", "http://localhost:8917")
    api_key = os.environ["RAG_API_KEY"]

    with RAGOpenAIClient(base_url=base_url, auth=ClientAuth(api_key=api_key)) as client:
        project = client.projects.ensure(
            name="sdk-demo-multiturn",
            description="Multi-turn + filters demo project",
        )
        project_id = project["project_id"]
        print("project_id:", project_id)

        messages = [
            {"role": "system", "content": "Отвечай лаконично и по делу."},
            {"role": "user", "content": "Какие документы есть в проекте?"},
            {"role": "assistant", "content": "Я могу посмотреть доступный контекст и источники."},
            {"role": "user", "content": "Сделай короткую сводку с фокусом на архитектуру."},
        ]

        completion = client.chat.completions.create(
            project_id=project_id,
            messages=messages,
            mode="hybrid",
            top_k=5,
            use_hyde=True,
            use_fact_queries=True,
            include_sources=True,
            filters={"tags": ["architecture"]},
        )

        print("answer:", completion["choices"][0]["message"]["content"])
        rag = completion.get("rag", {})
        print("partial:", rag.get("partial"))
        print("sources:", len(rag.get("sources", [])))


if __name__ == "__main__":
    main()
