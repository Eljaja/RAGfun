# Comparing chunking strategies on BRIGHT

## BRIGHT gold documents

**Gold documents** are those referenced by benchmark questions as ground truth (via `gold_ids`).

### Gold vs full corpus

1. **Gold only** (`--docs-from-gold N`):
   - Documents that are **actually needed** to answer BRIGHT questions
   - Used to evaluate retrieval quality
   - Smaller index → faster ingest and iteration
   - **Recommended for tests**: index only what eval needs

2. **Full category** (no `--docs-from-gold`):
   - Indexes every document in the category (can be 50k+)
   - Slower but more realistic
   - Use for final benchmarks

### Example

- Category `biology` may have ~57,000 documents
- Answering the questions may require only ~500 gold docs
- `--docs-from-gold 500` indexes just those → **~100× faster** ingest for smoke runs

## How many documents and categories?

### Quick run (~5–10 minutes)

```bash
python scripts/compare_chunking_strategies.py \
  --retrieval-url http://localhost:8085 \
  --splits biology \
  --docs-from-gold 500
```

- **1 split** (biology)
- **~500 documents** (gold)
- **~100 questions** for eval
- **Time**: ~5–10 minutes

### Medium run (~30–60 minutes)

```bash
python scripts/compare_chunking_strategies.py \
  --retrieval-url http://localhost:8085 \
  --splits biology,economics,psychology \
  --eval-splits biology,economics,psychology \
  --docs-from-gold 2000
```

- **3 splits**
- **~2000 documents** (gold)
- **~300 questions** for eval
- **Time**: ~30–60 minutes

### Full run (~2–4 hours)

```bash
python scripts/compare_chunking_strategies.py \
  --retrieval-url http://localhost:8085 \
  --splits all \
  --eval-splits all \
  --docs-from-gold 100000
```

- **12 splits**
- **~100,000 documents** (all gold)
- **1384 questions** for eval
- **Time**: ~2–4 hours

## BRIGHT splits

BRIGHT defines 12 splits:

1. `biology` — biology  
2. `earth_science` — Earth sciences  
3. `economics` — economics  
4. `psychology` — psychology  
5. `robotics` — robotics  
6. `stackoverflow` — programming Q&A  
7. `sustainable_living` — sustainable living  
8. `pony` — Pony language docs  
9. `leetcode` — programming problems  
10. `aops` — Art of Problem Solving math  
11. `theoremqa_theorems` — math theorems  
12. `theoremqa_questions` — math questions  

## Chunking strategies

### `semantic` (section-based; common in compose overrides)

- **Section splits**: chunks by Markdown-style headings  
- **Minimal overlap**: mainly when a section exceeds `max_tokens`  
- **Best for**: structured docs with clear headings  

### `semchunk` (library-driven; common default)

- **Semantic-ish splitting**: uses the `semchunk` library  
- **Overlap-aware**: respects textual continuity  
- **Best for**: general text, broader coverage  

## Running the comparison script

### 1. Quick test (recommended first)

```bash
cd /path/to/RAGfun
python scripts/compare_chunking_strategies.py \
  --retrieval-url http://localhost:8085 \
  --splits biology \
  --docs-from-gold 500
```

### 2. Reading output

The script prints a comparison table, for example:

```
CHUNKING STRATEGY COMPARISON
======================================================================
Strategy       nDCG@10    Recall@10    hit@10     hit@1      hit@3
----------------------------------------------------------------------
semchunk       0.4523     0.6234       0.7123     0.3456     0.5678
semantic       0.4456     0.6156       0.7034     0.3345     0.5567
======================================================================
```

### 3. Output files

- `results/chunking_strategy_comparison.json` — full summary  
- `results/bright_eval_semchunk.jsonl` — per-query details (`semchunk`)  
- `results/bright_eval_semantic.jsonl` — per-query details (`semantic`)  

## Metrics

- **nDCG@10** — ranking quality (order-sensitive)  
- **Recall@10** — share of relevant docs found in top-10  
- **hit@10** — fraction of questions with ≥1 relevant doc in top-10  
- **hit@1** — fraction with a relevant doc at rank 1  
- **hit@3** — fraction with a relevant doc in top-3  

## Recommendations

1. Start with the **quick** profile (one split, ~500 gold docs).  
2. If strategies are close, run the **medium** profile.  
3. For a final decision, run the **full** profile (all splits).  

## Notes

- **Re-indexing**: Changing strategy requires re-ingesting documents.  
- **Runtime**: Full benchmarks can take several hours.  
- **Resources**: Ensure enough RAM for OpenSearch (often ~16GB+ for large runs).  
