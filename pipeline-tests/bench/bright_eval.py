from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from urllib.parse import quote
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed
from tqdm import tqdm

GATE_BASE_URL = os.getenv("GATE_BASE_URL", "http://rag-gate:8090").rstrip("/")
RETRIEVAL_BASE_URL = os.getenv("RETRIEVAL_BASE_URL", "http://retrieval:8080").rstrip("/")

# BRIGHT leaderboard (short-doc) score is reported as average nDCG@10 across 12 datasets/domains.
# Source: https://brightbenchmark.github.io/
BRIGHT_12_DOMAINS: list[str] = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

# Judge config:
# - Prefer explicit JUDGE_* vars
# - Fall back to the same provider config as rag-gate (GATE_LLM_*) to avoid extra wiring
JUDGE_LLM_BASE_URL = (os.getenv("JUDGE_LLM_BASE_URL") or os.getenv("GATE_LLM_BASE_URL") or "").rstrip("/")
JUDGE_LLM_API_KEY = os.getenv("JUDGE_LLM_API_KEY") or os.getenv("GATE_LLM_API_KEY") or ""
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL") or os.getenv("GATE_LLM_MODEL") or ""


@dataclass(frozen=True)
class BrightExample:
    i: int
    domain: str
    query: str
    gold_ids: list[str]
    excluded_ids: list[str]
    gold_answer: str | None = None


def _chunked(xs: list[str], n: int) -> Iterable[list[str]]:
    n = max(1, int(n))
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _dedupe_keep_order(xs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_citations_like_brackets(s: str) -> str:
    # Gate may add inline citations like [1][2] – ignore them for judging.
    return re.sub(r"\[\d+\]", "", s or "").strip()


def _cheap_judge_if_obvious(*, gold: str, pred: str) -> dict[str, Any] | None:
    """
    Deterministic pre-judge to avoid LLM-judge hallucinations on simple cases.
    If the gold answer is already contained in the system answer (after light normalization),
    mark as correct with full score.
    """
    g = _norm_text(gold or "")
    p = _norm_text(_strip_citations_like_brackets(pred or ""))
    if not g or not p:
        return None
    if g in p:
        return {"is_correct": True, "score_0_5": 5, "notes": "cheap_match: gold is substring of pred"}
    return None


def _judge_answer(client: httpx.Client, *, question: str, gold: str, pred: str) -> dict[str, Any] | None:
    if not (JUDGE_LLM_BASE_URL and JUDGE_LLM_API_KEY and JUDGE_LLM_MODEL):
        return None

    cheap = _cheap_judge_if_obvious(gold=gold, pred=pred)
    if cheap is not None:
        return cheap

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(12))
    def _call() -> dict[str, Any]:
        system = (
            "You are a strict evaluator of QA answers.\n"
            "You are given: Question, Reference Answer, and System Answer.\n"
            "Judge correctness of the system answer relative to the reference.\n"
            "Ignore citation markers like [1], [2] if present.\n"
            "Do NOT use your world knowledge. Only compare system answer to the reference.\n"
            "If partially correct, give a mid score.\n"
            "Output ONLY valid JSON, no markdown.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "is_correct": true|false,\n'
            '  "score_0_5": 0..5,\n'
            '  "notes": "short explanation"\n'
            "}\n"
        )
        user = (
            f"Question: {question}\n"
            f"Reference Answer: {gold}\n"
            f"System Answer: {_strip_citations_like_brackets(pred)}\n"
        )
        payload = {
            "model": JUDGE_LLM_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {JUDGE_LLM_API_KEY}"}
        r = client.post(f"{JUDGE_LLM_BASE_URL}/chat/completions", json=payload, headers=headers)
        # Retry on rate limits and transient server errors
        if r.status_code == 429 or (500 <= r.status_code <= 599):
            raise RuntimeError(f"judge_http_{r.status_code}")
        r.raise_for_status()
        data = r.json()
        content = str(data["choices"][0]["message"]["content"])
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", content)
            if not m:
                return {"is_correct": False, "score_0_5": 0, "notes": "judge_output_parse_failed"}
            try:
                return json.loads(m.group(0))
            except Exception:
                return {"is_correct": False, "score_0_5": 0, "notes": "judge_output_parse_failed"}

    return _call()


def _dcg_binary(rels: list[int]) -> float:
    # DCG with binary relevance and log2 discount.
    # rels[i] corresponds to rank i (0-based).
    import math

    s = 0.0
    for i, rel in enumerate(rels):
        if rel <= 0:
            continue
        s += 1.0 / math.log2(i + 2)
    return s


def _ndcg_at_k(*, ranked_doc_ids: list[str], gold_ids: list[str], k: int) -> float:
    k = max(1, int(k))
    gold = set([str(x) for x in (gold_ids or []) if str(x)])
    if not gold:
        return 0.0
    top = ranked_doc_ids[:k]
    rels = [1 if d in gold else 0 for d in top]
    dcg = _dcg_binary(rels)
    ideal_rels = [1] * min(k, len(gold))
    idcg = _dcg_binary(ideal_rels)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _hit_at_k(*, ranked_doc_ids: list[str], gold_ids: list[str], k: int) -> int:
    k = max(1, int(k))
    gold = set([str(x) for x in (gold_ids or []) if str(x)])
    if not gold:
        return 0
    return 1 if any(d in gold for d in ranked_doc_ids[:k]) else 0


def _recall_at_k(*, ranked_doc_ids: list[str], gold_ids: list[str], k: int) -> float:
    k = max(1, int(k))
    gold = [str(x) for x in (gold_ids or []) if str(x)]
    if not gold:
        return 0.0
    gold_set = set(gold)
    got = sum(1 for d in set(ranked_doc_ids[:k]) if d in gold_set)
    return float(got / max(1, len(gold_set)))


@retry(wait=wait_fixed(1), stop=stop_after_delay(240))
def _wait_gate_ready() -> None:
    with httpx.Client(timeout=5.0) as c:
        r = c.get(f"{GATE_BASE_URL}/v1/readyz")
        if r.status_code != 200:
            raise RuntimeError(f"gate readyz status={r.status_code}")
        j = r.json()
        if not j.get("ready"):
            raise RuntimeError(f"gate not ready: {j}")


@retry(wait=wait_fixed(1), stop=stop_after_delay(240))
def _wait_retrieval_ready() -> None:
    with httpx.Client(timeout=5.0) as c:
        r = c.get(f"{RETRIEVAL_BASE_URL}/v1/readyz")
        if r.status_code != 200:
            raise RuntimeError(f"retrieval readyz status={r.status_code}")
        j = r.json()
        if not j.get("ready"):
            raise RuntimeError(f"retrieval not ready: {j}")


def _load_examples(*, domain: str, limit: int | None) -> list[BrightExample]:
    split = domain if (limit is None) else f"{domain}[:{int(limit)}]"
    ds = load_dataset("xlangai/BRIGHT", "examples", split=split)
    out: list[BrightExample] = []
    for idx, row in enumerate(ds):
        q = str(row.get("query") or "").strip()
        if not q:
            continue
        gold_ids = [str(x) for x in (row.get("gold_ids") or []) if str(x)]
        excluded_ids = [str(x) for x in (row.get("excluded_ids") or []) if str(x)]

        # Best-effort: BRIGHT is primarily a retrieval benchmark, but some domains/configs may include answers.
        gold_answer: str | None = None
        for key in ("answer", "gold", "reference", "gold_answer", "reference_answer", "final_answer"):
            if key in row and row.get(key):
                gold_answer = str(row.get(key) or "").strip()
                break
        if gold_answer is None:
            # Sometimes stored as list of answers.
            for key in ("answers", "gold_answers", "references"):
                v = row.get(key)
                if isinstance(v, list) and v:
                    gold_answer = str(v[0] or "").strip()
                    break

        out.append(
            BrightExample(
                i=idx,
                domain=domain,
                query=q,
                gold_ids=gold_ids,
                excluded_ids=excluded_ids,
                gold_answer=gold_answer or None,
            )
        )
    return out


def _fetch_documents_by_ids_fast_or_fallback(
    *,
    needed_ids: set[str],
    documents_split: str,
    allow_streaming_fallback: bool,
) -> dict[str, str]:
    """
    Resolve BRIGHT documents by id and return {doc_id: content}.

    Strategy (fast -> slower, but robust):
    1) Fast path: if ids are numeric and match row indices, use Dataset.select()
    2) Arrow filter (if underlying Arrow table is accessible): pyarrow.compute.is_in on 'id' column
    3) Batched scan over the non-streaming Dataset (early-exit once all ids found)
    4) Optional: streaming scan (slowest; can still early-exit)
    """
    needed_ids = {str(x) for x in needed_ids if str(x)}
    if not needed_ids:
        return {}

    need = set(needed_ids)

    # 1) Fast path: ids == row index (verify).
    all_numeric = all(x.isdigit() for x in need)
    if all_numeric:
        try:
            docs = load_dataset("xlangai/BRIGHT", "documents", split=documents_split)
            n = len(docs)
            idxs = [int(x) for x in need if int(x) >= 0]
            if idxs and max(idxs) < n:
                picked = docs.select(idxs)
                got_fast: dict[str, str] = {}
                for row in picked:
                    did = str(row.get("id") or "").strip()
                    content = str(row.get("content") or "")
                    if did:
                        got_fast[did] = content
                if need.issubset(set(got_fast.keys())):
                    return got_fast
        except Exception:
            # If anything goes wrong (schema mismatch, split missing), continue to robust paths.
            pass

    # Load once for robust paths (non-streaming, local Arrow cache).
    docs = load_dataset("xlangai/BRIGHT", "documents", split=documents_split)

    # 2) Arrow filter: fastest robust method if Arrow table is accessible.
    try:
        import pyarrow as pa
        import pyarrow.compute as pc

        table = getattr(getattr(docs, "data", None), "table", None)
        if table is not None and "id" in table.column_names and "content" in table.column_names:
            # is_in expects a ValueSet; pa.array is fine for moderate sets.
            mask = pc.is_in(table["id"], value_set=pa.array(list(need)))
            filtered = table.filter(mask)
            got_arrow: dict[str, str] = {}
            # Convert only matching rows to python
            ids = filtered["id"].to_pylist()
            contents = filtered["content"].to_pylist()
            for did, content in zip(ids, contents, strict=False):
                sdid = str(did).strip()
                if sdid:
                    got_arrow[sdid] = str(content or "")
            if need.issubset(set(got_arrow.keys())):
                return got_arrow
    except Exception:
        pass

    # 3) Batched scan (early-exit once all found)
    got_scan: dict[str, str] = {}
    remaining = set(need)
    bs = 2048
    try:
        total = len(docs)
    except Exception:
        total = None
    rng = range(0, (total or 0), bs) if total is not None else range(0, 0, bs)
    if total is not None:
        for off in tqdm(rng, desc="Scanning BRIGHT documents (batched)", unit="batch"):
            batch = docs[off : off + bs]
            ids = batch.get("id") or []
            contents = batch.get("content") or []
            for did, content in zip(ids, contents, strict=False):
                sdid = str(did).strip()
                if sdid in remaining:
                    got_scan[sdid] = str(content or "")
                    remaining.discard(sdid)
            if not remaining:
                return got_scan

    # 4) Optional: streaming scan (slowest; early-exit)
    if not allow_streaming_fallback:
        missing = sorted(list(need - set(got_scan.keys())))
        raise RuntimeError(
            f"Missing {len(missing)} docs after batched scan; sample missing: {missing[:5]}. "
            "Re-run with --allow_streaming_fallback to do a streaming scan."
        )

    ds_stream = load_dataset("xlangai/BRIGHT", "documents", split=documents_split, streaming=True)
    got: dict[str, str] = dict(got_scan)
    remaining = set(need) - set(got.keys())
    for row in tqdm(ds_stream, desc="Scanning BRIGHT documents (streaming)", unit="doc"):
        did = str(row.get("id") or "").strip()
        if not did or did not in remaining:
            continue
        got[did] = str(row.get("content") or "")
        remaining.discard(did)
        if not remaining:
            break
    missing = sorted(list(need - set(got.keys())))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} docs after streaming scan. Sample missing: {missing[:5]}")
    return got


def _ensure_indexed_via_gate(
    *,
    gate: httpx.Client,
    doc_text_by_id: dict[str, str],
    project_id: str,
    tenant_id: str | None,
    tags: list[str],
    upload_chunk: int,
    check_chunk: int,
    refresh: bool,
) -> None:
    doc_ids = sorted([d for d in doc_text_by_id.keys() if d])
    if not doc_ids:
        return

    def _status(doc_id: str) -> dict[str, Any] | None:
        try:
            did = quote(str(doc_id), safe="/")
            r = gate.get(f"{GATE_BASE_URL}/v1/documents/{did}/status")
            if r.status_code != 200:
                return None
            j = r.json()
            return j if isinstance(j, dict) and j.get("ok") else None
        except Exception:
            return None

    def _is_indexed(doc_id: str) -> bool:
        st = _status(doc_id)
        return bool(st and st.get("indexed") is True)

    # Best-effort skip already indexed docs to save time.
    already: set[str] = set()
    try:
        import concurrent.futures

        workers = max(1, min(32, int(check_chunk) if check_chunk else 16))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut_by_id = {ex.submit(_is_indexed, did): did for did in doc_ids}
            for fut in tqdm(concurrent.futures.as_completed(fut_by_id), total=len(fut_by_id), desc="Checking indexed docs"):
                did = fut_by_id[fut]
                try:
                    if fut.result():
                        already.add(did)
                except Exception:
                    pass
    except Exception:
        already = set()

    missing = [d for d in doc_ids if d not in already]
    if not missing:
        return

    tags_csv = ",".join([t for t in tags if t])
    for part in tqdm(list(_chunked(missing, upload_chunk)), desc="Uploading docs via gate", unit="batch"):
        for did in part:
            txt = doc_text_by_id.get(did) or ""
            raw = txt.encode("utf-8", errors="replace")
            files = {"file": ("doc.txt", raw, "text/plain")}
            data = {
                "doc_id": did,
                "title": (txt[:120] or f"bright:{did}").replace("\n", " ").strip(),
                "uri": f"bright://documents/{did}",
                "source": "bright",
                "lang": "en",
                "tags": tags_csv,
                "project_id": project_id,
                "refresh": "true" if refresh else "false",
            }
            if tenant_id:
                data["tenant_id"] = tenant_id
            r = gate.post(f"{GATE_BASE_URL}/v1/documents/upload", files=files, data=data)
            if r.status_code not in (200, 202):
                raise RuntimeError(f"upload failed doc_id={did}: {r.status_code} {r.text[:300]}")

    # Wait for indexing to show up in gate status.
    @retry(wait=wait_fixed(2), stop=stop_after_delay(3600))
    def _wait_all() -> None:
        left = 0
        for did in missing:
            if not _is_indexed(did):
                left += 1
        if left > 0:
            raise RuntimeError(f"Not indexed yet: {left} remaining")

    _wait_all()


def _search_retrieval(
    *,
    retrieval: httpx.Client,
    query: str,
    mode: str,
    top_k: int,
    project_id: str,
    tenant_id: str | None,
    rerank: bool | None,
    max_chunks_per_doc: int | None,
) -> list[str]:
    payload: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "top_k": int(top_k),
        "include_sources": False,
        "sources_level": "basic",
        "group_by_doc": True,
        "rerank": rerank,
        "max_chunks_per_doc": max_chunks_per_doc,
        "filters": {"project_id": project_id},
        "acl": [],
    }
    if tenant_id:
        payload["filters"]["tenant_id"] = tenant_id

    r = retrieval.post(f"{RETRIEVAL_BASE_URL}/v1/search", json=payload)
    r.raise_for_status()
    j = r.json()
    hits = j.get("hits") or []
    doc_ids = [str((h or {}).get("doc_id") or "").strip() for h in hits]
    doc_ids = [d for d in doc_ids if d]
    return _dedupe_keep_order(doc_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--domain",
        default="",
        help="BRIGHT domain split (single), e.g. biology/economics/leetcode/... (deprecated; prefer --domains)",
    )
    ap.add_argument(
        "--domains",
        default="",
        help="Comma-separated list of BRIGHT domains to run in one go, e.g. biology,economics,leetcode",
    )
    ap.add_argument("--limit", type=int, default=0, help="<=0 means all examples in the domain split")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    ap.add_argument("--rerank", default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--max_chunks_per_doc", type=int, default=1)
    ap.add_argument(
        "--leaderboard_score",
        action="store_true",
        help="Append BRIGHT leaderboard-style score row: macro-average nDCG@10 across 12 domains (short-doc track).",
    )
    ap.add_argument(
        "--require_all_domains",
        action="store_true",
        help="With --leaderboard_score: fail if not all 12 BRIGHT domains were evaluated in this run.",
    )
    ap.add_argument(
        "--judge",
        action="store_true",
        help="Also run generation via gate /v1/chat and score with LLM-judge (requires gold answers in dataset + JUDGE_LLM_* or GATE_LLM_* env).",
    )
    ap.add_argument("--judge_sleep_s", type=float, default=0.2, help="Sleep between judge calls to reduce 429s.")

    ap.add_argument(
        "--collection_per_domain",
        action="store_true",
        help="Use separate project_id (collection) per domain: index and search within the domain's collection instead of mixing domains.",
    )
    ap.add_argument(
        "--project_id_prefix",
        default="",
        help="Prefix for per-domain project_id when --collection_per_domain is set (default: 'bright' or derived from --project_id).",
    )

    ap.add_argument(
        "--corpus_mode",
        default="gold_only",
        choices=["gold_only"],
        help="For now: index only documents that appear in gold_ids for the chosen domain split.",
    )
    ap.add_argument(
        "--allow_streaming_fallback",
        action="store_true",
        help="Allow slow streaming scan if fast/batched paths can't resolve all docs (usually not needed).",
    )

    ap.add_argument("--project_id", default="", help="Collection id to isolate BRIGHT docs (default auto).")
    ap.add_argument("--tenant_id", default="", help="Optional tenant_id")
    ap.add_argument("--tag", default="bright_eval", help="Tag to attach to uploaded docs")

    ap.add_argument("--upload_batch", type=int, default=25)
    ap.add_argument("--check_batch", type=int, default=250)
    ap.add_argument("--refresh", action="store_true", help="Force refresh per doc upload (faster availability; slower indexing).")

    ap.add_argument("--out", default="/out/bright_eval.jsonl")
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    # Domains selection:
    # - --domains: multi-domain run
    # - --domain: legacy single-domain run
    domains_raw = [d.strip() for d in (args.domains or "").split(",") if d.strip()]
    if not domains_raw:
        d = (args.domain or "").strip()
        if not d:
            raise SystemExit("Provide --domains (comma-separated) or --domain (single).")
        domains_raw = [d]
    # keep stable order + unique
    domains = _dedupe_keep_order(domains_raw)

    limit = None if args.limit is None or args.limit <= 0 else int(args.limit)
    top_k = max(1, int(args.top_k))
    max_chunks_per_doc = None if args.max_chunks_per_doc is None or args.max_chunks_per_doc <= 0 else int(args.max_chunks_per_doc)

    rerank: bool | None
    if args.rerank == "auto":
        rerank = None
    elif args.rerank == "true":
        rerank = True
    else:
        rerank = False

    tenant_id = (args.tenant_id or "").strip() or None
    project_id = (args.project_id or "").strip()
    project_id_prefix = (args.project_id_prefix or "").strip()
    if args.collection_per_domain:
        # For per-domain collections, allow using --project_id as prefix for convenience.
        if not project_id_prefix:
            project_id_prefix = project_id or "bright"
        project_id = ""  # unused in this mode
    else:
        if not project_id:
            # Multi-domain default: put everything into a single isolated collection.
            if len(domains) == 1:
                project_id = f"bright:{domains[0]}:gold_only"
            else:
                project_id = f"bright:multi:{len(domains)}:gold_only"

    def _project_for_domain(dom: str) -> str:
        if args.collection_per_domain:
            return f"{project_id_prefix}:{dom}:gold_only"
        return project_id

    # Load examples for all requested domains.
    examples: list[BrightExample] = []
    for dom in domains:
        examples.extend(_load_examples(domain=dom, limit=limit))
    if not examples:
        raise SystemExit("No examples loaded (domain split name correct?)")
    any_gold_answer = any(bool((ex.gold_answer or "").strip()) for ex in examples)

    # Build needed doc ids per domain, because BRIGHT 'documents' are also split by domain.
    needed_by_domain: dict[str, set[str]] = {d: set() for d in domains}
    for ex in examples:
        for gid in ex.gold_ids:
            if gid:
                needed_by_domain.setdefault(ex.domain, set()).add(str(gid))

    needed_total = sum(len(v) for v in needed_by_domain.values())
    if needed_total <= 0:
        raise SystemExit("No gold_ids found in examples (unexpected for BRIGHT).")

    # Fetch documents per domain and merge into one map for uploading.
    doc_text_by_id: dict[str, str] = {}
    for dom in domains:
        needed_ids = needed_by_domain.get(dom) or set()
        if not needed_ids:
            continue
        dom_docs = _fetch_documents_by_ids_fast_or_fallback(
            needed_ids=needed_ids,
            documents_split=dom,
            allow_streaming_fallback=bool(args.allow_streaming_fallback),
        )
        doc_text_by_id.update(dom_docs)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    done: set[int] = set()
    if args.resume and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                _ = f.readline()
                for line in f:
                    try:
                        j = json.loads(line)
                        if isinstance(j, dict) and "i" in j:
                            done.add(int(j["i"]))
                    except Exception:
                        continue
        except Exception:
            done = set()

    f_out = open(out_path, "a" if (args.resume and os.path.exists(out_path)) else "w", encoding="utf-8")
    if not (args.resume and os.path.exists(out_path)):
        f_out.write(
            json.dumps(
                {
                    "summary": {
                        "started_at": time.time(),
                        "gate_base_url": GATE_BASE_URL,
                        "retrieval_base_url": RETRIEVAL_BASE_URL,
                        "judge_llm_base_url": JUDGE_LLM_BASE_URL or None,
                        "judge_llm_model": JUDGE_LLM_MODEL or None,
                        "domains": domains,
                        "collection_per_domain": bool(args.collection_per_domain),
                        "project_id_prefix": project_id_prefix or None,
                        "project_id": project_id or None,
                        "top_k": top_k,
                        "mode": args.mode,
                        "rerank": args.rerank,
                        "tenant_id": tenant_id,
                        "corpus_mode": args.corpus_mode,
                        "docs_indexed_target": len(doc_text_by_id),
                        "judge_enabled": bool(args.judge),
                        "judge_has_gold_answers": bool(any_gold_answer),
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        f_out.flush()

    _wait_gate_ready()
    _wait_retrieval_ready()

    with httpx.Client(timeout=300.0) as gate, httpx.Client(timeout=120.0) as retrieval:
        # Index per domain if requested, so collections don't mix.
        if args.collection_per_domain:
            for dom in domains:
                dom_needed = needed_by_domain.get(dom) or set()
                if not dom_needed:
                    continue
                dom_doc_map = {did: doc_text_by_id[did] for did in dom_needed if did in doc_text_by_id}
                tags = ["bright", args.tag, dom]
                _ensure_indexed_via_gate(
                    gate=gate,
                    doc_text_by_id=dom_doc_map,
                    project_id=_project_for_domain(dom),
                    tenant_id=tenant_id,
                    tags=tags,
                    upload_chunk=int(args.upload_batch),
                    check_chunk=int(args.check_batch),
                    refresh=bool(args.refresh),
                )
        else:
            tags = ["bright", args.tag, "multi" if len(domains) > 1 else domains[0]]
            _ensure_indexed_via_gate(
                gate=gate,
                doc_text_by_id=doc_text_by_id,
                project_id=project_id,
                tenant_id=tenant_id,
                tags=tags,
                upload_chunk=int(args.upload_batch),
                check_chunk=int(args.check_batch),
                refresh=bool(args.refresh),
            )

        # Metrics accumulators
        n = 0
        ndcg10_sum = 0.0
        hit1_sum = 0
        hit3_sum = 0
        hit10_sum = 0
        recall10_sum = 0.0
        judged = 0
        judge_correct = 0
        judge_score_sum = 0.0
        judge_errors = 0
        judge_skipped = 0

        # Per-domain accumulators
        per_dom: dict[str, dict[str, Any]] = {}
        for dom in domains:
            per_dom[dom] = {
                "count": 0,
                "ndcg10_sum": 0.0,
                "hit1_sum": 0,
                "hit3_sum": 0,
                "hit10_sum": 0,
                "recall10_sum": 0.0,
                "judged": 0,
                "judge_correct": 0,
                "judge_score_sum": 0.0,
                "judge_errors": 0,
                "judge_skipped": 0,
            }

        @retry(wait=wait_fixed(2), stop=stop_after_attempt(12))
        def _chat(payload: dict[str, Any]) -> httpx.Response:
            try:
                return gate.post(f"{GATE_BASE_URL}/v1/chat", json=payload)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as e:
                raise RuntimeError(f"chat_transport_error:{type(e).__name__}") from e

        for ex in tqdm(examples, desc="Evaluating BRIGHT retrieval"):
            if done and ex.i in done:
                continue

            ex_project_id = _project_for_domain(ex.domain)

            ranked = _search_retrieval(
                retrieval=retrieval,
                query=ex.query,
                mode=args.mode,
                top_k=top_k,
                project_id=ex_project_id,
                tenant_id=tenant_id,
                rerank=rerank,
                max_chunks_per_doc=max_chunks_per_doc,
            )

            # Apply excluded_ids (if present) by removing them from the ranked list.
            if ex.excluded_ids:
                excluded = set(ex.excluded_ids)
                ranked = [d for d in ranked if d not in excluded]

            ndcg10 = _ndcg_at_k(ranked_doc_ids=ranked, gold_ids=ex.gold_ids, k=10)
            hit1 = _hit_at_k(ranked_doc_ids=ranked, gold_ids=ex.gold_ids, k=1)
            hit3 = _hit_at_k(ranked_doc_ids=ranked, gold_ids=ex.gold_ids, k=3)
            hit10 = _hit_at_k(ranked_doc_ids=ranked, gold_ids=ex.gold_ids, k=10)
            recall10 = _recall_at_k(ranked_doc_ids=ranked, gold_ids=ex.gold_ids, k=10)

            pred_answer: str | None = None
            judge_obj: dict[str, Any] | None = None
            judge_skip_reason: str | None = None
            if args.judge:
                if not (ex.gold_answer or "").strip():
                    judge_skipped += 1
                    judge_skip_reason = "no_gold_answer_in_dataset"
                elif not (JUDGE_LLM_BASE_URL and JUDGE_LLM_API_KEY and JUDGE_LLM_MODEL):
                    judge_skipped += 1
                    judge_skip_reason = "judge_not_configured"
                else:
                    # Ask gate to generate an answer using the indexed BRIGHT collection only.
                    payload = {
                        "query": ex.query,
                        "history": [],
                        "retrieval_mode": args.mode,
                        "top_k": top_k,
                        "filters": {"project_id": ex_project_id, **({"tenant_id": tenant_id} if tenant_id else {})},
                        "acl": [],
                        # Keep this False to avoid extra citation-enforcement retries.
                        "include_sources": False,
                    }
                    rr = _chat(payload)
                    if rr.status_code != 200:
                        judge_errors += 1
                        judge_obj = {
                            "error": "gate_chat_failed",
                            "status_code": rr.status_code,
                            "detail": (rr.text or "")[:300],
                        }
                    else:
                        j = rr.json()
                        pred_answer = str(j.get("answer") or "")
                        with httpx.Client(timeout=60.0) as jc:
                            try:
                                judge_obj = _judge_answer(
                                    jc,
                                    question=ex.query,
                                    gold=str(ex.gold_answer or ""),
                                    pred=pred_answer,
                                )
                            except Exception as e:
                                judge_errors += 1
                                judge_obj = {"error": "judge_exception", "detail": f"{type(e).__name__}:{str(e)[:200]}"}
                        # Update aggregates only if judge returned a structured response.
                        if isinstance(judge_obj, dict) and "is_correct" in judge_obj:
                            judged += 1
                            try:
                                if bool(judge_obj.get("is_correct")):
                                    judge_correct += 1
                            except Exception:
                                pass
                            try:
                                judge_score_sum += float(judge_obj.get("score_0_5") or 0.0)
                            except Exception:
                                pass
                        # Avoid rate limits
                        if args.judge_sleep_s and args.judge_sleep_s > 0:
                            time.sleep(float(args.judge_sleep_s))

            row = {
                "i": ex.i,
                "domain": ex.domain,
                "query": ex.query,
                "gold_ids": ex.gold_ids,
                "excluded_ids": ex.excluded_ids,
                "retrieved_doc_ids": ranked[:top_k],
                "metrics": {"ndcg@10": ndcg10, "hit@1": hit1, "hit@3": hit3, "hit@10": hit10, "recall@10": recall10},
                "project_id": ex_project_id,
                "pred_answer": pred_answer,
                "gold_answer": ex.gold_answer,
                "judge": judge_obj,
                "judge_skip_reason": judge_skip_reason,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

            n += 1
            ndcg10_sum += ndcg10
            hit1_sum += hit1
            hit3_sum += hit3
            hit10_sum += hit10
            recall10_sum += recall10

            dom_acc = per_dom.get(ex.domain)
            if dom_acc is not None:
                dom_acc["count"] += 1
                dom_acc["ndcg10_sum"] += ndcg10
                dom_acc["hit1_sum"] += hit1
                dom_acc["hit3_sum"] += hit3
                dom_acc["hit10_sum"] += hit10
                dom_acc["recall10_sum"] += recall10
                if args.judge:
                    if judge_skip_reason is not None:
                        dom_acc["judge_skipped"] += 1
                    elif isinstance(judge_obj, dict) and "is_correct" in judge_obj:
                        dom_acc["judged"] += 1
                        try:
                            if bool(judge_obj.get("is_correct")):
                                dom_acc["judge_correct"] += 1
                        except Exception:
                            pass
                        try:
                            dom_acc["judge_score_sum"] += float(judge_obj.get("score_0_5") or 0.0)
                        except Exception:
                            pass
                    elif isinstance(judge_obj, dict) and judge_obj.get("error"):
                        dom_acc["judge_errors"] += 1

        # Append final aggregate row (JSONL-friendly)
        if n > 0:
            agg = {
                "aggregate": {
                    "count": n,
                    "ndcg@10": ndcg10_sum / n,
                    "hit@1": hit1_sum / n,
                    "hit@3": hit3_sum / n,
                    "hit@10": hit10_sum / n,
                    "recall@10": recall10_sum / n,
                    "judge_count": judged,
                    "judge_accuracy": (judge_correct / judged) if judged > 0 else None,
                    "judge_avg_score_0_5": (judge_score_sum / judged) if judged > 0 else None,
                    "judge_errors": judge_errors,
                    "judge_skipped": judge_skipped,
                }
            }
            f_out.write(json.dumps(agg, ensure_ascii=False) + "\n")
            f_out.flush()

            # Macro aggregates by domain (mean over queries per domain).
            by_dom_out: dict[str, Any] = {}
            for dom, a in per_dom.items():
                cnt = int(a.get("count") or 0)
                if cnt <= 0:
                    continue
                judged_cnt = int(a.get("judged") or 0)
                by_dom_out[dom] = {
                    "count": cnt,
                    "ndcg@10": float(a["ndcg10_sum"]) / cnt,
                    "hit@1": float(a["hit1_sum"]) / cnt,
                    "hit@3": float(a["hit3_sum"]) / cnt,
                    "hit@10": float(a["hit10_sum"]) / cnt,
                    "recall@10": float(a["recall10_sum"]) / cnt,
                    "judge_count": judged_cnt,
                    "judge_accuracy": (float(a["judge_correct"]) / judged_cnt) if judged_cnt > 0 else None,
                    "judge_avg_score_0_5": (float(a["judge_score_sum"]) / judged_cnt) if judged_cnt > 0 else None,
                    "judge_errors": int(a.get("judge_errors") or 0),
                    "judge_skipped": int(a.get("judge_skipped") or 0),
                }
            f_out.write(json.dumps({"aggregate_by_domain": by_dom_out}, ensure_ascii=False) + "\n")
            f_out.flush()

            if args.leaderboard_score:
                # Leaderboard uses macro nDCG@10 across 12 domains (short-doc track).
                # We compute it from per-domain nDCG@10 means.
                missing = [d for d in BRIGHT_12_DOMAINS if d not in by_dom_out]
                present = [d for d in BRIGHT_12_DOMAINS if d in by_dom_out]
                if args.require_all_domains and missing:
                    raise SystemExit(f"--require_all_domains: missing domains: {missing}")
                score = None
                if present:
                    score = sum(float(by_dom_out[d]["ndcg@10"]) for d in present) / float(len(present))
                f_out.write(
                    json.dumps(
                        {
                            "leaderboard_score": {
                                "track": "short_doc",
                                "metric": "ndcg@10",
                                "macro_avg_over_domains": True,
                                "domains_expected": BRIGHT_12_DOMAINS,
                                "domains_used": present,
                                "domains_missing": missing,
                                "score": score,
                            }
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f_out.flush()

    f_out.close()


if __name__ == "__main__":
    main()

