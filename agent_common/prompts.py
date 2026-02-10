"""
Shared prompt templates for agent-search.

Used for plan, HyDE, fact/keyword queries, and answer generation with citation.
"""

# Plan / retrieval strategy (agent-search plan stage)
PLAN_SYSTEM = (
    "You are a retrieval strategist for a RAG system. "
    "Return a single JSON object only. Keep 'reason' short."
)
PLAN_USER = (
    "{history}"
    "Decide per-request retrieval knobs.\n"
    "JSON fields: retrieval_mode (bm25|vector|hybrid), top_k (5..24), "
    "rerank (true/false), use_hyde (true/false), reason.\n"
    "top_k guidelines: 5-8 for simple factoid/definition; 10-14 for comparison/multi-entity; 16-24 for list/summary/multi-hop.\n"
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

# Answer (citation)
ANSWER_SYSTEM = (
    "You answer using the provided context only. "
    "When you use information from a context block, cite it with [N] where N is the block number (e.g. [1], [2]). "
    "If the context is insufficient, say what is missing. Reply in {lang}."
)
ANSWER_SYSTEM_WITH_TOOLS = (
    "You answer using the provided context. "
    "ALWAYS use the calculator tool for any numeric computation (arithmetic, sqrt, log, etc). "
    "Use execute_code for code that must run (e.g. list comprehensions, data transforms). "
    "Cite context with [N] when you use it. Reply in {lang}."
)
ANSWER_USER = (
    "{history}Question:\n{query}\n\nContext:\n{context}\n\n"
    "Answer in the same language as the question."
)
