# BRIGHT Benchmark Validation

Валидация RAGfun (rugfunsota) на бенчмарке [BRIGHT](https://brightbenchmark.github.io/) в режиме **полного пайплайна** (retrieval + generation + LLM judge).

## Быстрый старт

### 1. Запуск стека rugfunsota

```bash
cd /home/ubuntu/szavodnov/RAGfun
docker compose up -d opensearch qdrant postgres pgbouncer rabbitmq rustfs
docker compose up -d infinity-embed infinity-rerank document-storage retrieval rag-gate
```

### 2. Сборка pipeline-tests (из rag_fun)

```bash
cd /home/ubuntu/ifedotov/rag_fun
docker compose -f docker-compose.e2e.yml build pipeline-tests
```

### 3. Запуск валидации

**Только retrieval (без LLM judge):**
```bash
/home/ubuntu/run_bright_validation.sh --stack rugfunsota --domain biology --limit 50
```

**Полный пайплайн (retrieval + generation + LLM judge):**
```bash
# Убедитесь, что в .env заданы GATE_LLM_BASE_URL, GATE_LLM_API_KEY, GATE_LLM_MODEL
/home/ubuntu/run_bright_validation.sh --stack rugfunsota --domain biology --limit 50 --judge
```

**Обе версии RAG (rag_fun и rugfunsota):**
```bash
# Запустите оба стека в разных терминалах, затем:
/home/ubuntu/run_bright_validation.sh --judge --limit 50
```

## Опции скрипта

| Опция | Описание |
|-------|----------|
| `--stack STACK` | `rag_fun` \| `rugfunsota` \| `both` |
| `--domain DOMAIN` | Домен BRIGHT (biology, economics, leetcode, ...) |
| `--limit N` | Лимит примеров (по умолчанию 50) |
| `--judge` | Включить LLM judge для full pipeline |
| `--all-domains` | Все 12 доменов BRIGHT |
| `--leaderboard` | Добавить leaderboard score (macro nDCG@10) |

## Результаты

Результаты сохраняются в `$OUT_DIR` (по умолчанию `/home/ubuntu/out/bright_validation/`):

- `bright_rag_fun_*.jsonl` — ifedotov/rag_fun
- `bright_rugfunsota_*.jsonl` — szavodnov/RAGfun

Формат: JSONL с метриками nDCG@10, Hit@k, Recall@10, и при `--judge` — judge_accuracy, judge_avg_score_0_5.

## Сравнение двух версий

Для сравнения rag_fun и rugfunsota на полном пайплайне:

1. Запустите rag_fun: `cd ifedotov/rag_fun && docker compose up -d`
2. Запустите rugfunsota: `cd szavodnov/RAGfun && docker compose up -d`
3. Выполните: `./run_bright_validation.sh --judge --limit 100`
4. Сравните aggregate-строки в выходных JSONL файлах
