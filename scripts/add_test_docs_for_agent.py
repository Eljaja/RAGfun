#!/usr/bin/env python3
"""Добавляет тестовые документы для проверки agent-search и deep-research с разными параметрами."""
import argparse
import sys
import httpx

# Документы: (doc_id, project_id, source, lang, title, text)
# Разные project_id, source, lang для тестов фильтров
TEST_DOCS = [
    # project_id=agent_test, source=manual, lang=ru
    (
        "agent_rag_intro",
        "agent_test",
        "manual",
        "ru",
        "Что такое RAG",
        """
RAG (Retrieval-Augmented Generation) — архитектура для улучшения ответов языковых моделей.
RAG объединяет поиск по документам с генерацией текста. Сначала система находит релевантные фрагменты
в базе знаний, затем передаёт их в LLM как контекст. Это снижает галлюцинации и позволяет
работать с актуальными данными без переобучения модели.
""",
    ),
    (
        "agent_python_basics",
        "agent_test",
        "manual",
        "ru",
        "Python для начинающих",
        """
Python — интерпретируемый язык программирования высокого уровня. Создан Гвидо ван Россумом в 1991 году.
Python поддерживает несколько парадигм: ООП, функциональное и императивное программирование.
Популярен в data science, веб-разработке (Django, FastAPI), автоматизации и машинном обучении.
""",
    ),
    (
        "agent_hybrid_search",
        "agent_test",
        "manual",
        "ru",
        "Гибридный поиск",
        """
Гибридный поиск сочетает BM25 (лексический) и векторный поиск. BM25 хорошо находит точные совпадения,
векторный — семантическое сходство. Результаты объединяют через RRF. Это повышает recall в RAG.
""",
    ),
    # project_id=tech_docs, source=api, lang=en
    (
        "tech_fastapi",
        "tech_docs",
        "api",
        "en",
        "FastAPI framework",
        """
FastAPI is a modern, fast web framework for building APIs with Python 3.8+.
It is based on standard Python type hints. Key features: automatic OpenAPI docs,
async support, data validation with Pydantic. Used for microservices and ML APIs.
""",
    ),
    (
        "tech_embeddings",
        "tech_docs",
        "api",
        "en",
        "Embeddings in RAG",
        """
Embeddings are vector representations of text. In RAG, documents and queries are
converted to embeddings, then nearest vectors are searched. Popular models:
sentence-transformers, OpenAI text-embedding-ada-002. Dimension typically 384-1536.
""",
    ),
    # project_id=demo (уже есть в системе)
    (
        "demo_rag_vs_finetuning",
        "demo",
        "cli",
        "ru",
        "RAG против fine-tuning",
        """
RAG и fine-tuning решают разные задачи. Fine-tuning обучает модель на новых данных, меняя веса.
RAG не меняет модель — он подтягивает контекст из внешнего хранилища при каждом запросе.
RAG проще обновлять, не требует GPU для переобучения.
""",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-url", default="http://localhost:8085", help="Retrieval service URL")
    args = ap.parse_args()
    url = args.retrieval_url.rstrip("/")

    print(f"Adding {len(TEST_DOCS)} documents to retrieval {url}...")
    with httpx.Client(timeout=60.0) as c:
        try:
            r = c.get(f"{url}/v1/healthz", timeout=5)
            if r.status_code != 200:
                print(f"Retrieval not reachable: {r.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"Retrieval not reachable: {e}")
            sys.exit(1)

        for doc_id, project_id, source, lang, title, text in TEST_DOCS:
            payload = {
                "mode": "document",
                "document": {
                    "doc_id": doc_id,
                    "project_id": project_id,
                    "source": source,
                    "title": title,
                    "lang": lang,
                },
                "text": text.strip(),
                "refresh": True,
            }
            try:
                r = c.post(f"{url}/v1/index/upsert", json=payload)
                r.raise_for_status()
                j = r.json()
                print(f"  OK: {doc_id} (project={project_id}, source={source}, lang={lang})")
            except Exception as e:
                print(f"  FAIL {doc_id}: {e}")
                sys.exit(1)

    print("\nDone. Test queries:")
    print("  agent-search: curl -X POST http://localhost:8093/v1/agent/stream -H 'Content-Type: application/json' -d '{\"query\":\"Что такое RAG?\",\"filters\":{\"project_id\":\"agent_test\"}}'")
    print("  gate chat:   curl -X POST http://localhost:8092/v1/chat -H 'Content-Type: application/json' -d '{\"query\":\"What is FastAPI?\",\"filters\":{\"project_id\":\"tech_docs\"}}'")


if __name__ == "__main__":
    main()
