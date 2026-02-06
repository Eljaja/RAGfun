"""
Shared prompt templates for agent-search and deep-research.

Used by both services to ensure consistent LLM behavior for plan, HyDE, fact/keyword
queries, and answer generation with citation.
"""

# Plan / retrieval strategy (agent-search plan stage)
PLAN_SYSTEM = (
    "You are a retrieval strategist for a RAG system. "
    "Return a single JSON object only. Keep 'reason' short."
)
PLAN_USER = (
    "{history}"
    "Decide per-request retrieval knobs.\n"
    "JSON fields: retrieval_mode (bm25|vector|hybrid), top_k (1..40), "
    "rerank (true/false), use_hyde (true/false), use_web_search (true/false), reason.\n"
    "Set use_web_search=true for: current events, news, facts outside documents, recent data.\n"
    "Query: {query}"
)

# HyDE
HYDE_SYSTEM = (
    "Write a short hypothetical answer passage for retrieval. Write in {lang} only."
)
HYDE_USER = "Query: {query}\nReturn a 3-5 sentence passage in {lang}."

# Fact queries
FACT_QUERIES_SYSTEM = "Extract fact-oriented sub-queries from the user request."
FACT_QUERIES_USER = (
    "{history}"
    'Return JSON: {{"fact_queries": [..]}} with 2-3 short queries.\n'
    "Query: {query}"
)

# Keyword queries
KEYWORD_QUERIES_SYSTEM = "Extract short keyword queries from the user request."
KEYWORD_QUERIES_USER = (
    "{history}"
    'Return JSON: {{"keywords": [..]}} with 3-6 short keyword phrases. '
    "Query: {query}"
)

# Deep-research (single prompt string format)
DEEP_HYDE = (
    "Write a short hypothetical answer passage for retrieval. Write in {lang} only.\n"
    "Query: {query}\nReturn a 3-5 sentence passage."
)
DEEP_FACT_QUERIES = (
    "Extract fact-oriented sub-queries. Return JSON only.\n"
    "Query: {query}\n"
    'Return JSON: {{"fact_queries": [..]}} with 2-3 short queries.'
)
DEEP_KEYWORD_QUERIES = (
    "Extract short keyword queries. Return JSON only.\n"
    "Query: {query}\n"
    'Return JSON: {{"keywords": [..]}} with 3-6 short keyword phrases.'
)

# Answer (citation)
ANSWER_SYSTEM = (
    "You answer using the provided context only. "
    "When you use information from a context block, cite it with [N] where N is the block number (e.g. [1], [2]). "
    "If the context is insufficient, say what is missing. Reply in {lang}."
)
ANSWER_USER = (
    "{history}Question:\n{query}\n\nContext:\n{context}\n\n"
    "Answer in the same language as the question."
)
