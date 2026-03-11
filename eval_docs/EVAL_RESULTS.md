# Результаты оценки RAG (автоматический прогон)

| # | Вопрос | Режим | R | C | Comp | Cit | Ответ (сокращённо) |
|---|--------|-------|---|---|------|-----|--------------------|
| 1 | Что такое RAGfun?… | chat | 2 | 4 | 4 | 5 | **RAGfun** — это платформа для **RAG** (Retrieval-Augmented Generation, генераци… |
| 1 | Что такое RAGfun?… | agent | 2 | 4 | 4 | 4 | RAGfun — это платформа для RAG (Retrieval-Augmented Generation), которая объедин… |
| 1 | Что такое RAGfun?… | deep | 2 | 4 | 4 | 3 | # Report: RAG Fusion – An Analysis  ## 1) Executive Summary  - RAG Fusion is an … |
| 2 | Какой заголовок используется для ODS API… | chat | 5 | 4 | 4 | 5 | Согласно предоставленному контексту, для ODS API ключа используется заголовок **… |
| 2 | Какой заголовок используется для ODS API… | agent | 4 | 4 | 3 | 5 | Заголовок X-ODS-API-KEY используется для передачи ODS API‑ключа [1][2]. |
| 2 | Какой заголовок используется для ODS API… | deep | 5 | 4 | 4 | 3 | # Отчёт: Docs as Code и автоматизация документации  ## 1) Краткое резюме  - **Do… |
| 3 | На каком порту работает UI?… | chat | 3 | 4 | 3 | 5 | UI работает на порту **3301** [1][2]. |
| 3 | На каком порту работает UI?… | agent | 3 | 4 | 3 | 5 | UI работает на порту 3301 [1][2]. |
| 3 | На каком порту работает UI?… | deep | 2 | 4 | 4 | 3 | ```json {   "report": {     "1) Executive summary": [       "Ubiquiti UniFi cont… |
| 4 | Сколько режимов retrieval поддерживается… | chat | 3 | 4 | 4 | 5 | Согласно документации, система поддерживает **3 режима retrieval** [1]:  1. **bm… |
| 4 | Сколько режимов retrieval поддерживается… | agent | 2 | 4 | 3 | 4 | Три режима: bm25, vector и hybrid [1]. |
| 4 | Сколько режимов retrieval поддерживается… | deep | 5 | 4 | 4 | 3 | # Отчёт: Retrieval-Augmented Generation (RAG)  ## 1. Краткое резюме  - **Неопред… |
| 5 | Что такое ODS?… | chat | 2 | 4 | 4 | 5 | # ODS (Operational Data Store)  **ODS (Operational Data Store)** — это компонент… |
| 5 | Что такое ODS?… | agent | 2 | 4 | 4 | 4 | ODS (Operational Data Store) — это система изоляции тенантов в платформе RAGfun.… |
| 5 | Что такое ODS?… | deep | 2 | 4 | 4 | 3 | # Operational Data Store (ODS): A Comprehensive Report  ## 1. Executive Summary … |
| 6 | Что такое HyDE в agent-search?… | chat | 3 | 4 | 4 | 5 | # HyDE в agent-search  Согласно предоставленному контексту, **HyDE** (Hypothetic… |
| 6 | Что такое HyDE в agent-search?… | agent | 3 | 4 | 4 | 5 | HyDE (Hypothetical Document Embeddings) — это техника в agent-search, при которо… |
| 6 | Что такое HyDE в agent-search?… | deep | 3 | 4 | 4 | 3 | # HYDE (Hypothetical Document Embeddings): A Technical Report  ---  ## 1. Execut… |
| 7 | Какой фреймворк использует deep-research… | chat | 5 | 4 | 4 | 4 | Согласно контексту [1], deep-research использует **LangGraph** в качестве фреймв… |
| 7 | Какой фреймворк использует deep-research… | agent | 3 | 4 | 3 | 2 | Deep‑research построен на базе фреймворка **LangGraph**【1】. |
| 7 | Какой фреймворк использует deep-research… | deep | 2 | 4 | 4 | 3 | # Report: OpenAI Deep Research Overview  ## 1. Executive Summary  - OpenAI has r… |
| 8 | Чем agent-search отличается от deep-rese… | chat | 3 | 4 | 4 | 5 | ## Различия между agent-search и deep-research  Согласно документации, основное … |
| 8 | Чем agent-search отличается от deep-rese… | agent | 3 | 4 | 4 | 5 | Согласно документации [1], основное отличие заключается в подходе к обработке за… |
| 8 | Чем agent-search отличается от deep-rese… | deep | 2 | 4 | 4 | 3 | # Comparative Analysis: ChatGPT Agent vs Deep Research  ## 1. Executive Summary … |
| 9 | Перечисли основные сервисы RAGfun.… | chat | 5 | 4 | 5 | 4 | Основные сервисы RAGfun:  1. **Gate** — основной API-шлюз 2. **document-storage*… |
| 9 | Перечисли основные сервисы RAGfun.… | agent | 5 | 4 | 5 | 4 | Основные сервисы RAGfun [1]:  1. **Gate** — API-шлюз с аутентификацией и изоляци… |
| 9 | Перечисли основные сервисы RAGfun.… | deep | 2 | 4 | 5 | 3 | # RAG APIs: Overview, Implementation, and Applications  ## 1) Executive Summary … |
| 10 | Как создать тенанта?… | chat | 2 | 4 | 4 | 5 | Для создания тенанта необходимо выполнить **POST-запрос на `/v1/tenants`** с заг… |
| 10 | Как создать тенанта?… | agent | 5 | 4 | 4 | 4 | Чтобы создать тенанта, нужно отправить POST-запрос на `/v1/tenants` с заголовком… |
| 10 | Как создать тенанта?… | deep | 5 | 4 | 4 | 3 | # Отчет: Создание мультитенантной инфраструктуры и управление арендаторами  ## 1… |

## Сводка (средние)

- **chat**: R=3.3 C=4.0 Comp=4.0 Cit=4.8
- **agent**: R=3.2 C=4.0 Comp=3.7 Cit=4.2
- **deep**: R=3.0 C=4.0 Comp=4.1 Cit=3.0
