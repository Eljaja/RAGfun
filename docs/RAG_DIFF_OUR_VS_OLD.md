# Разница между нашим RAG (rugfunsota / szavodnov/RAGfun) и старым (ifedotov/rag_fun)

Оба стека на BEIR SciFact дают почти одинаковые метрики (MRR ~0.66, Recall@10 ~0.81). Ниже — чем отличаются реализации.

---

## 1. Чанкинг (разбиение текста на чанки)

| Аспект | Наш RAG (rugfunsota) | Старый RAG (rag_fun) |
|--------|----------------------|----------------------|
| **Retrieval (mode=document)** | `RAG_CHUNK_STRATEGY`: **semantic** \| token. **Semantic** — по секциям/заголовкам (Markdown, нумерованные разделы), минимальный overlap. **Token** — параграфы + скользящее окно по токенам (tiktoken). | Только **token-based**: `chunk_text()` — параграфы + скользящее окно по токенам. Нет выбора стратегии. |
| **Doc-processor** (при индексации через пайплайн) | `PROCESSOR_CHUNK_STRATEGY`: **semantic** \| fixed. **Semantic** — секционная разрезка (заголовки, разделы). **Fixed** — по символам с overlap, с сохранением таблиц и code blocks. | Только **char-based**: `chunk_text_chars()` — по символам с учётом Markdown-блоков (таблицы, код). Нет секционной стратегии. |

**Итог:** у нас есть семантический/секционный чанкинг (в retrieval и в doc-processor); в старом RAG везде только токенный/символьный.

---

## 2. Реранкер

| Аспект | Наш RAG | Старый RAG |
|--------|---------|------------|
| **Реализация** | Протокол `Reranker` + две реализации: **RerankClient** (HTTP, внешний сервис) и **LocalReranker** (in-process, sentence-transformers, CPU). | Только **RerankClient** (HTTP). |
| **Конфиг** | `RAG_RERANK_PROVIDER`: **http** \| **local**. При `local` не нужен внешний сервис. | Всегда HTTP (infinity-rerank). |
| **Поведение** | В docker оба используют один и тот же Infinity rerank (BAAI/bge-reranker-v2-m3), режим auto. | То же. |

**Итог:** у нас добавлен опциональный локальный реранкер (CPU); при использовании того же Infinity разница только в возможности выбора.

---

## 3. Retrieval: кандидаты и пул перед реранком

| Параметр | Наш RAG | Старый RAG |
|----------|---------|------------|
| **retrieval_candidates** | Явный параметр: сколько кандидатов собираем до реранка (default 20). | Нет такого параметра — пул задаётся только через bm25_top_k / vector_top_k и fusion. |
| **bm25_top_k / vector_top_k** | В коде default 25; в docker переопределено на 200. | В коде default 50; в docker 200. |
| **rerank_max_candidates** | Default 20 (в коде), в docker 200. | Default 50 (в коде), в docker 200. |

В docker у обоих по сути одинаковый сценарий: топ-200 от гибрида → реранк по 200. Разница в дефолтах кода и в явной концепции «retrieval_candidates» у нас.

---

## 4. Конфиг и код (service)

| Параметр / возможность | Наш RAG | Старый RAG |
|------------------------|---------|------------|
| **chunk_strategy** (retrieval) | Есть: semantic \| token. | Нет — один способ чанкинга. |
| **RAG_RERANK_PROVIDER** | Есть: http \| local. | Нет — только http. |
| **embedding_contextual_headers** | Есть (включено в docker). | Есть (включено в docker). |
| **Логирование этапов поиска** | Подробные логи: rag_stage_bm25, rag_stage_vector, rag_stage_fusion, rag_stage_rerank, rag_search_done. | Без такой разбивки по этапам. |

---

## 5. Doc-processor

| Возможность | Наш RAG | Старый RAG |
|-------------|---------|------------|
| Стратегия чанкинга | semantic (по секциям) или fixed (по символам). | Только char-based (chunk_text_chars). |
| Конфиг | `PROCESSOR_CHUNK_STRATEGY`, `chunk_size_chars`, `chunk_overlap_chars`. | `chunk_size_chars`, `chunk_overlap_chars`. |

---

## 6. Почему метрики почти совпадают на BEIR SciFact

- Один и тот же корпус, одни и те же запросы и qrels.
- Одинаковые эмбеддинги (BAAI/bge-m3), реранкер (BAAI/bge-reranker-v2-m3), гибрид BM25+vector, в docker одинаковые лимиты (200/200).
- В **нашем** прогоне индекс строился через `index_beir_corpus.py` с **mode=document** — чанкинг делал **retrieval** (у нас semantic, у старого — token). Тексты SciFact короткие; секционная разрезка vs токенная даёт близкое число и размер чанков, поэтому итоговые метрики почти одинаковые.
- Различия сильнее проявятся на своих документах с явной структурой (много заголовков/разделов) или при переключении на локальный реранкер / другие лимиты кандидатов.

---

## Кратко

- **Наш RAG:** семантический/секционный чанкинг (retrieval + doc-processor), выбор реранкера (http или local), явный параметр retrieval_candidates, подробные логи поиска.
- **Старый RAG:** только токенный/символьный чанкинг, только HTTP-реранкер, без отдельной концепции retrieval_candidates.
- На BEIR SciFact оба дают ~0.66 MRR, ~0.81 Recall@10, потому что стек (эмбеддинги, реранк, лимиты) в docker одинаковый, а отличия в чанкинге на этом датасете мало влияют.
