"""LLM-related utilities: citation enforcement, extractive answer enforcement."""

from app.clients import LLMClient
from app.text_utils import _has_inline_citations, _norm_text_for_contains, _TOKEN_RE


async def _enforce_citations_or_refuse(
    *,
    llm: LLMClient,
    messages: list[dict[str, str]],
    answer: str,
    refs: list[int],
) -> str:
    """
    If sources exist and the model returned an answer without citations, do one strict rewrite attempt.
    If it still fails, refuse (to avoid "fake grounding" by auto-adding citations).
    """
    if not refs:
        return answer
    if _has_inline_citations(answer):
        return answer

    suffix = "".join([f"[{r}]" for r in refs])
    rewrite_prompt = (
        "Rewrite your answer to comply STRICTLY with citations.\n"
        f"- Every factual claim MUST include inline citations like {suffix}\n"
        "- If you cannot answer using the provided sources, reply exactly: \"I don't know\" (no citations).\n"
        "- Do NOT add any facts, numbers, commands, or code that are not explicitly present in the sources.\n"
        "- Do NOT use generic advice or filler.\n"
    )
    retry_messages = list(messages) + [{"role": "user", "content": rewrite_prompt}]
    rewritten = await llm.chat(messages=retry_messages)
    if _has_inline_citations(rewritten) or rewritten.strip() == "I don't know":
        return rewritten
    return "I don't know"


async def _enforce_extractive_or_refuse(
    *,
    llm: LLMClient,
    messages: list[dict[str, str]],
    answer: str,
    context_text: str,
) -> str:
    """
    Best-effort anti-hallucination for factoid QA when include_sources=False:
    if the produced short answer is not present in sources, do one rewrite attempt
    forcing a verbatim span. If it still fails, refuse.
    """
    ans = (answer or "").strip()
    if not ans:
        return ans
    ctx = (context_text or "").strip()
    if not ctx:
        return ans

    # Apply only to short answers (entity/one-liner). Lists are hard to substring-match reliably.
    ans_toks = _TOKEN_RE.findall(ans)
    if len(ans_toks) > 6:
        return ans
    if "," in ans or "\n" in ans:
        return ans

    n_ans = _norm_text_for_contains(ans)
    n_ctx = _norm_text_for_contains(ctx)
    if n_ans and n_ans in n_ctx:
        return ans

    rewrite_prompt = (
        "Your previous answer is NOT a verbatim span from the provided Sources.\n"
        "Rewrite your answer STRICTLY as follows:\n"
        "- Output ONLY the exact answer phrase copied verbatim from Sources (no extra words).\n"
        "- Do NOT add citations like [1].\n"
        "- If the exact answer phrase is not explicitly present in Sources, reply exactly with \"I don't know\" (English) OR \"Не знаю.\" (Russian), matching the user's language.\n"
    )
    retry_messages = list(messages) + [{"role": "user", "content": rewrite_prompt}]
    rewritten = (await llm.chat(messages=retry_messages)).strip()
    if rewritten == "I don't know":
        return rewritten
    n_rew = _norm_text_for_contains(rewritten)
    if n_rew and n_rew in n_ctx:
        return rewritten
    return "I don't know"

