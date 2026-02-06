#!/usr/bin/env python3
"""Добавляет тестовые документы через gate (upload → document-storage → async indexing)."""
import argparse
import sys
import time
import httpx

# Документы для проверки поиска
TEST_DOCS = [
    ("doc_rag_intro", "Что такое RAG", """
RAG (Retrieval-Augmented Generation) — архитектура для улучшения ответов языковых моделей.
RAG объединяет поиск по документам с генерацией текста. Сначала система находит релевантные фрагменты
в базе знаний, затем передаёт их в LLM как контекст. Это снижает галлюцинации и позволяет
работать с актуальными данными без переобучения модели.
"""),
    ("doc_python_basics", "Python для начинающих", """
Python — интерпретируемый язык программирования высокого уровня. Создан Гвидо ван Россумом в 1991 году.
Python поддерживает несколько парадигм: ООП, функциональное и императивное программирование.
Популярен в data science, веб-разработке (Django, FastAPI), автоматизации и машинном обучении.
"""),
    ("doc_rag_vs_finetuning", "RAG против fine-tuning", """
RAG и fine-tuning решают разные задачи. Fine-tuning обучает модель на новых данных, меняя веса.
RAG не меняет модель — он подтягивает контекст из внешнего хранилища при каждом запросе.
RAG проще обновлять, не требует GPU для переобучения. Fine-tuning даёт более встроенные знания.
"""),
    ("doc_embeddings", "Эмбеддинги в RAG", """
Эмбеддинги — векторные представления текста. В RAG документы и запросы преобразуются в эмбеддинги,
затем ищутся ближайшие векторы. Популярные модели: sentence-transformers, OpenAI text-embedding-ada-002.
Размерность обычно 384–1536. Для русского текста важны многоязычные модели эмбеддингов.
"""),
    ("doc_hybrid_search", "Гибридный поиск", """
Гибридный поиск сочетает BM25 (лексический) и векторный поиск. BM25 хорошо находит точные совпадения,
векторный — семантическое сходство. Результаты объединяют через RRF. Это повышает recall в RAG.
"""),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-url", default="http://localhost:8092", help="Gate base URL (rugfunsota: 8092)")
    ap.add_argument("--wait-indexed", type=int, default=60, help="Max seconds to wait for indexing")
    args = ap.parse_args()
    url = args.gate_url.rstrip("/")

    print(f"Uploading {len(TEST_DOCS)} documents via gate {url}...")
    with httpx.Client(timeout=120.0) as c:
        # Gate reachable
        try:
            r = c.get(f"{url}/v1/healthz", timeout=10)
            if r.status_code != 200:
                print(f"Gate not reachable: {r.status_code} {r.text[:200]}")
                sys.exit(1)
        except Exception as e:
            print(f"Gate not reachable: {e}")
            sys.exit(1)

        doc_ids = []
        for doc_id, title, text in TEST_DOCS:
            files = {"file": ("doc.txt", text.strip().encode("utf-8"), "text/plain")}
            data = {"doc_id": doc_id, "title": title, "source": "test_scripts", "lang": "ru", "refresh": "true"}
            try:
                r = c.post(f"{url}/v1/documents/upload", files=files, data=data)
                r.raise_for_status()
                j = r.json()
                if j.get("ok") and j.get("storage", {}).get("ok"):
                    doc_ids.append(doc_id)
                    print(f"  Uploaded: {doc_id}")
                else:
                    print(f"  Upload failed {doc_id}: {j}")
            except Exception as e:
                print(f"  FAIL {doc_id}: {e}")
                sys.exit(1)

        if not doc_ids:
            print("No documents uploaded.")
            sys.exit(1)

        # Wait for indexing
        print(f"\nWaiting up to {args.wait_indexed}s for indexing...")
        deadline = time.monotonic() + args.wait_indexed
        indexed = set()
        while time.monotonic() < deadline and len(indexed) < len(doc_ids):
            for did in doc_ids:
                if did in indexed:
                    continue
                try:
                    r = c.get(f"{url}/v1/documents/{did}/status", timeout=5)
                    if r.status_code == 200:
                        j = r.json()
                        if j.get("indexed"):
                            indexed.add(did)
                            print(f"  Indexed: {did}")
                except Exception:
                    pass
            if len(indexed) < len(doc_ids):
                time.sleep(2)

        if len(indexed) < len(doc_ids):
            print(f"  Warning: only {len(indexed)}/{len(doc_ids)} indexed within timeout")

    print("\nDone. Try queries:")
    print("  - 'Что такое RAG?'")
    print("  - 'Python для data science'")
    print("  - 'RAG vs fine-tuning'")
    print("  - 'Эмбеддинги и векторный поиск'")


if __name__ == "__main__":
    main()
