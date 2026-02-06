#!/usr/bin/env python3
"""Добавляет тестовые документы в retrieval для проверки agent-search и deep-research."""
import argparse
import sys
import httpx

# Документы для проверки поиска (RAG, Python, embeddings, fine-tuning)
TEST_DOCS = [
    {
        "doc_id": "doc_rag_intro",
        "title": "Что такое RAG",
        "text": """
RAG (Retrieval-Augmented Generation) — это архитектура для улучшения ответов языковых моделей.
RAG объединяет поиск по документам с генерацией текста. Сначала система находит релевантные фрагменты
в базе знаний, затем передаёт их в LLM как контекст для ответа. Это снижает галлюцинации и позволяет
работать с актуальными данными без переобучения модели.
""",
    },
    {
        "doc_id": "doc_python_basics",
        "title": "Python для начинающих",
        "text": """
Python — интерпретируемый язык программирования высокого уровня. Создан Гвидо ван Россумом в 1991 году.
Python поддерживает несколько парадигм: ООП, функциональное и императивное программирование.
Популярен в data science, веб-разработке (Django, FastAPI), автоматизации и машинном обучении.
""",
    },
    {
        "doc_id": "doc_rag_vs_finetuning",
        "title": "RAG против fine-tuning",
        "text": """
RAG и fine-tuning решают разные задачи. Fine-tuning обучает модель на новых данных, меняя веса.
RAG не меняет модель — он подтягивает контекст из внешнего хранилища при каждом запросе.
RAG проще обновлять (добавлять документы), не требует GPU для переобучения. Fine-tuning даёт
более «встроенные» знания, но дороже в поддержке. Часто используют оба подхода вместе.
""",
    },
    {
        "doc_id": "doc_embeddings",
        "title": "Эмбеддинги в RAG",
        "text": """
Эмбеддинги — векторные представления текста. В RAG документы и запросы преобразуются в эмбеддинги,
затем ищутся ближайшие векторы (cosine similarity, dot product). Популярные модели: sentence-transformers,
OpenAI text-embedding-ada-002, Cohere embed. Размерность обычно 384–1536. Для русского текста
важно использовать многоязычные или русскоязычные модели эмбеддингов.
""",
    },
    {
        "doc_id": "doc_hybrid_search",
        "title": "Гибридный поиск",
        "text": """
Гибридный поиск сочетает BM25 (лексический) и векторный поиск. BM25 хорошо находит точные совпадения
и сущности, векторный — семантическое сходство. Результаты объединяют через RRF (Reciprocal Rank Fusion).
Это повышает recall и качество retrieval в RAG-системах.
""",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Add test documents to retrieval")
    ap.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Retrieval base URL (8080=ragfun, 8085=rugfunsota; 8085 may hang if rag_fun is also running)",
    )
    ap.add_argument(
        "--project-id",
        default="test",
        help="project_id for documents (default: test)",
    )
    ap.add_argument("--refresh", action="store_true", help="Refresh index after each doc")
    args = ap.parse_args()
    url = args.url.rstrip("/")

    print(f"Adding {len(TEST_DOCS)} documents to {url}...")
    with httpx.Client(timeout=60.0) as c:
        # Health check
        try:
            r = c.get(f"{url}/v1/healthz", timeout=5)
            r.raise_for_status()
        except Exception as e:
            print(f"Retrieval not reachable: {e}")
            sys.exit(1)

        for doc in TEST_DOCS:
            payload = {
                "mode": "document",
                "document": {
                    "doc_id": doc["doc_id"],
                    "project_id": args.project_id,
                    "title": doc["title"],
                    "source": "test_scripts",
                },
                "text": doc["text"].strip(),
                "refresh": args.refresh,
            }
            try:
                r = c.post(f"{url}/v1/index/upsert", json=payload)
                r.raise_for_status()
                print(f"  OK: {doc['doc_id']}")
            except Exception as e:
                print(f"  FAIL: {doc['doc_id']}: {e}")
                sys.exit(1)

    print("\nDone. Try queries via agent-search (8091) or gate /v1/chat (8090):")
    print("  - 'Что такое RAG?'")
    print("  - 'Python для data science'")
    print("  - 'RAG vs fine-tuning'")
    print("  - 'Эмбеддинги и векторный поиск'")
    print("  - 'Гибридный поиск BM25'")


if __name__ == "__main__":
    main()
