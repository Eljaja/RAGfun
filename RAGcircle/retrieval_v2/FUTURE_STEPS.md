# Future Retrieval Step Types

Stripping out anything that's indexing-time or brain-layer, here's what's left -- things that belong in this retrieval service.

## New Retrieval Step Types

**Metadata filtering** on existing search steps. Push filters into Qdrant/OpenSearch queries so you search a narrower candidate set. Tags, language, source document, ACL -- all already indexed by the doc processor. Not a separate step, just a `filters` field on `VectorSearchStep` and `BM25SearchStep`.

> **Priority consumer: generator_v2 factoid expand.** The old gate's factoid expansion searched *within* the top-scoring source documents (`doc_ids` filter) rather than doing a blind corpus-wide query. This is a meaningful quality regression in the current generator -- factoid sub-questions like "what was the exact date?" retrieve much better when scoped to the 1-2 documents that already scored highest. Once metadata filtering lands here, the generator's `retrieval_client.py` needs a `filters: dict | None` parameter, and `steps/enrich.py` (`FactoidExpandStep`) passes `{"source_id": [top_chunk.source_id for ...]}`. ~15 lines on the generator side once this service exposes the capability.

> **Shared SDK client.** Both `generator_v2` and `gate_v2` maintain their own HTTP retrieval wrappers. A thin shared client package (or at minimum a shared schema/types module) would make rolling out new retrieval features (like filters) a single-point change instead of updating N callers independently.

**Phrase search.** Exact phrase matching via OpenSearch `match_phrase`. Different from BM25's fuzzy term matching. When the user searches for "error code 5042", you want exact string match, not BM25's term-frequency weighting across "error" and "code" and "5042" separately.

## New Fusion Methods

**Weighted RRF.** Per-source weights on the existing RRF formula. "Trust vector 70%, BM25 30%." One multiplier change in `rrf()`, same rank-based properties, but tunable.

## New Ranking Step Types

**Score threshold.** Drop everything below a minimum score. Only valid after rerank (same constraint as adaptive_k). The brain or user says "don't give me anything the reranker scored below 0.5."

**Diversity / MMR.** After reranking gives you the most relevant chunks, diversity removes near-duplicates. Penalizes chunks that are too similar to already-selected chunks. Needs pairwise similarity (cosine on embeddings), the embedding endpoint is already available. Solves the "top 5 results are all from the same paragraph" problem.

## New Finalize Step Types

**Context expansion.** Given the final selected chunks, fetch neighboring chunks from the same document. Chunk #5 scored best, also return #4 and #6. Uses `source_id` and `chunk_index` which are already on every `ChunkResult`. One extra fetch to the stores per expanded chunk.

## Summary

| Step | Phase | Complexity | What it solves |
|---|---|---|---|
| Metadata filters | retrieve | Small -- fields on existing steps | Precision, multi-tenancy, ACL |
| Phrase search | retrieve | Small -- new step, one OS query type | Exact term matching |
| Weighted RRF | combine | Tiny -- one multiplier | Source preference tuning |
| Score threshold | rank | Tiny -- one comparison | Quality floor |
| Diversity / MMR | rank | Medium -- needs embeddings | Redundancy elimination |
| Context expansion | finalize | Small -- adjacent chunk fetch | Chunk boundary problem |

Six additions, all within the retrieval service boundary, all expressible as plan steps.
