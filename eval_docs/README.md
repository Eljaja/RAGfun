# Eval: ручная оценка RAG (human-in-the-loop)

## Документы (загружены в коллекцию `eval`)
- `doc_ragfun_overview.txt` — обзор RAGfun
- `doc_ods_tenant.txt` — ODS и тенанты
- `doc_agent_search.txt` — agent-search
- `doc_deep_research.txt` — deep-research
- `doc_deployment.txt` — deployment
- `doc_retrieval_modes.txt` — режимы retrieval
- `doc_faq.txt` — FAQ
- `doc_evaluation.txt` — оценка качества RAG

## Вопросы
См. `EVAL_QUESTIONS.md` — 50 вопросов по типам: простые фактоидные, определения, сравнение, списки, процесс, multi-hop, сложные.

## Рабочий лист
См. `EVAL_WORKSHEET.md` — шаблон для заполнения оценок (relevance, correctness, completeness, citation quality по 1–5).

## Full ODS: обязательный ключ

- Перед любым eval укажите tenant API key (в UI поле `ODS API key`).
- В full ODS режиме (`GATE_REQUIRE_TENANT_AUTH=true`) без `X-ODS-API-KEY` запросы будут отклоняться.
- Коллекции в UI загружаются tenant-scoped: используйте кнопку `Refresh` после смены ключа.

## Автоматический прогон
```bash
python3 run_eval.py
```

**Следить за прогрессом** (в другом терминале):
```bash
tail -f eval_docs/eval_progress.txt
```

Формат: `[HH:MM:SS] Q1/10: ... → chat... ✓ chat R=4 C=4 (1/30)`

## Как оценивать вручную
1. В UI: введите ODS API key, выберите коллекцию **eval**.
2. Для каждого вопроса из EVAL_QUESTIONS.md запустите 3 режима: обычный чат, agent-search, deep-research.
3. Заполните EVAL_WORKSHEET.md (или CSV/Excel).

## Быстрая проверка API c ключом
```bash
curl -X POST http://localhost:8092/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-ODS-API-KEY: <tenant_api_key>" \
  -d '{"query":"What is RAGfun?"}'
```
