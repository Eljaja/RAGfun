from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed
from tqdm import tqdm

GATE_BASE_URL = os.getenv("GATE_BASE_URL", "http://rag-gate:8090").rstrip("/")

# Judge config:
# - Prefer explicit JUDGE_* vars
# - Fall back to the same provider config as rag-gate (GATE_LLM_*) to avoid extra wiring
JUDGE_LLM_BASE_URL = (os.getenv("JUDGE_LLM_BASE_URL") or os.getenv("GATE_LLM_BASE_URL") or "").rstrip("/")
JUDGE_LLM_API_KEY = os.getenv("JUDGE_LLM_API_KEY") or os.getenv("GATE_LLM_API_KEY") or ""
JUDGE_LLM_MODEL = os.getenv("JUDGE_LLM_MODEL") or os.getenv("GATE_LLM_MODEL") or "gpt-4o-mini"


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_citations_like_brackets(s: str) -> str:
    # Gate adds inline citations like [1][2] – ignore them for judging.
    return re.sub(r"\[\d+\]", "", s or "").strip()


def _cheap_judge_if_obvious(*, gold: str, pred: str) -> dict[str, Any] | None:
    """
    Deterministic pre-judge to avoid LLM-judge hallucinations on simple factoid datasets.
    If the gold answer string is already contained in the system answer (after light normalization),
    mark as correct with full score.
    """
    g = _norm_text(gold or "")
    p = _norm_text(_strip_citations_like_brackets(pred or ""))
    if not g or not p:
        return None
    if g in p:
        return {"is_correct": True, "score_0_5": 5, "notes": "cheap_match: gold is substring of pred"}
    return None


def _best_oracle_overlap(*, hits: list[dict[str, Any]], oracle_context: str) -> dict[str, Any]:
    oracle = _norm_text(oracle_context or "")
    if not oracle:
        return {"best_overlap": None, "matched": None}
    best = 0.0
    matched = False
    for h in hits:
        t = _norm_text(str(h.get("text") or ""))
        if not t:
            continue
        # Cheap token overlap proxy
        toks = [x for x in re.split(r"[^a-zа-я0-9]+", t) if x]
        if not toks:
            continue
        inter = sum(1 for x in toks if x in oracle)
        score = inter / max(1, len(toks))
        best = max(best, score)
        if score >= 0.6:
            matched = True
    return {"best_overlap": best, "matched": matched}


@retry(wait=wait_fixed(1), stop=stop_after_delay(240))
def _wait_gate_ready() -> None:
    with httpx.Client(timeout=5.0) as c:
        r = c.get(f"{GATE_BASE_URL}/v1/readyz")
        if r.status_code != 200:
            raise RuntimeError(f"gate readyz status={r.status_code}")
        j = r.json()
        if not j.get("ready"):
            raise RuntimeError(f"gate not ready: {j}")


def _delete_all_documents(client: httpx.Client) -> None:
    # Best-effort: this endpoint requires storage enabled (it is, in e2e compose).
    r = client.delete(f"{GATE_BASE_URL}/v1/documents", params={"confirm": "true"})
    if r.status_code not in (200, 207):
        raise RuntimeError(f"delete_all_documents failed: {r.status_code} {r.text[:300]}")


@retry(wait=wait_fixed(1), stop=stop_after_delay(600))
def _wait_doc_indexed(client: httpx.Client, *, doc_id: str) -> None:
    r = client.get(f"{GATE_BASE_URL}/v1/documents/{doc_id}/status")
    if r.status_code != 200:
        raise RuntimeError(f"status failed {doc_id}: {r.status_code}")
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(f"status not ok {doc_id}: {j}")
    if not bool(j.get("indexed")):
        raise RuntimeError(f"not indexed yet {doc_id}")


def _doc_status(client: httpx.Client, *, doc_id: str) -> dict[str, Any] | None:
    """Best-effort doc status for 'ensure indexed' logic."""
    try:
        r = client.get(f"{GATE_BASE_URL}/v1/documents/{doc_id}/status")
        if r.status_code != 200:
            return None
        j = r.json()
        return j if j.get("ok") else None
    except Exception:
        return None


def _upload_document_only(*, doc_id: str, path: Path, refresh: bool) -> None:
    raw = path.read_bytes()
    files = {"file": (path.name, raw, "text/plain")}
    data = {
        "doc_id": doc_id,
        "title": f"ruwiki:{doc_id}",
        "uri": f"https://ru.wikipedia.org/?curid={doc_id}",
        "source": "ru_wiki",
        "lang": "ru",
        "tags": "ru_rag_test_dataset",
        "refresh": "true" if refresh else "false",
    }
    with httpx.Client(timeout=300.0) as client:
        r = client.post(f"{GATE_BASE_URL}/v1/documents/upload", files=files, data=data)
    # Gate can return 202 Accepted if async ingestion is enabled.
    if r.status_code not in (200, 202):
        raise RuntimeError(f"upload failed doc_id={doc_id}: {r.status_code} {r.text[:300]}")


def _wait_doc_indexed_one(*, doc_id: str) -> None:
    with httpx.Client(timeout=30.0) as client:
        _wait_doc_indexed(client, doc_id=doc_id)


def _status_is_indexed_one(*, doc_id: str) -> bool:
    with httpx.Client(timeout=10.0) as client:
        st = _doc_status(client, doc_id=doc_id)
        return bool(st and st.get("indexed") is True)


def _bulk_upload_then_wait(
    *,
    items: list[tuple[str, Path]],
    concurrency: int,
    desc: str,
) -> None:
    """
    Upload all docs first (concurrently), then wait for indexing (concurrently).
    items: [(doc_id, path), ...]
    """
    if not items:
        return
    workers = max(1, int(concurrency))

    # Phase 1: upload
    upload_errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for doc_id, path in items:
            # refresh per-doc doesn't rely on ordering; keep it true to make docs searchable ASAP
            futs.append(ex.submit(_upload_document_only, doc_id=doc_id, path=path, refresh=True))
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc=f"{desc}: upload", leave=False):
            try:
                f.result()
            except Exception as e:
                upload_errors.append(f"{type(e).__name__}: {str(e)[:200]}")
    if upload_errors:
        raise RuntimeError(f"bulk_upload_failed ({len(upload_errors)}): {upload_errors[:5]}")

    # Phase 2: wait indexed
    wait_errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_wait_doc_indexed_one, doc_id=doc_id) for doc_id, _ in items]
        for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc=f"{desc}: wait_index", leave=False):
            try:
                f.result()
            except Exception as e:
                wait_errors.append(f"{type(e).__name__}: {str(e)[:200]}")
    if wait_errors:
        raise RuntimeError(f"bulk_wait_index_failed ({len(wait_errors)}): {wait_errors[:5]}")


def _extract_doc_ids_from_chat(chat_json: dict[str, Any]) -> list[str]:
    # Prefer stable unique sources list (deduped per doc_id, ordered by first appearance).
    srcs = chat_json.get("sources") or []
    out: list[str] = []
    for s in srcs:
        did = str((s or {}).get("doc_id") or "").strip()
        if not did:
            continue
        out.append(did)
    if out:
        return out
    # Fallback: context chunks.
    for c in (chat_json.get("context") or []):
        did = str((c or {}).get("doc_id") or "").strip()
        if did:
            out.append(did)
    # preserve order, unique
    uniq: list[str] = []
    seen: set[str] = set()
    for d in out:
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq


def _parse_df_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    cols = list(df.columns)

    def pick(cands: list[str]) -> str:
        for c in cands:
            if c in cols:
                return c
        # best-effort fuzzy by normalized
        norm_map = {re.sub(r"\s+", " ", str(c).strip().lower()): str(c) for c in cols}
        for c in cands:
            k = re.sub(r"\s+", " ", c.strip().lower())
            if k in norm_map:
                return norm_map[k]
        raise KeyError(f"missing expected column; have={cols}")

    q_col = pick(["Вопрос", "вопрос", "question"])
    a_col = pick(["Правильный ответ", "правильный ответ", "answer", "gold"])
    ctx_col = pick(["Контекст", "контекст", "context", "oracle_context"])
    f_col = pick(["Название файла", "название файла", "Файл", "файл", "file", "filename", "doc_id"])
    return q_col, a_col, ctx_col, f_col


def _judge_answer(client: httpx.Client, *, question: str, gold: str, pred: str) -> dict[str, Any] | None:
    if not (JUDGE_LLM_BASE_URL and JUDGE_LLM_API_KEY and JUDGE_LLM_MODEL):
        return None

    cheap = _cheap_judge_if_obvious(gold=gold, pred=pred)
    if cheap is not None:
        return cheap

    @retry(wait=wait_fixed(5), stop=stop_after_attempt(12))
    def _call() -> dict[str, Any]:
        system = (
            "Ты строгий проверяющий качества ответов на русском языке.\n"
            "Тебе дают Вопрос, Эталонный ответ и Ответ системы.\n"
            "Оцени корректность ответа системы относительно эталона.\n"
            "Игнорируй ссылочные маркеры вида [1], [2] — это цитаты.\n"
            "НЕ используй свои знания о мире. Сравнивай ответ системы только с эталоном.\n"
            "Если ответ частично верный, ставь средний балл.\n"
            "Выводи ТОЛЬКО валидный JSON без обёрток и без markdown.\n\n"
            "Схема JSON:\n"
            "{\n"
            '  "is_correct": true|false,\n'
            '  "score_0_5": 0..5,\n'
            '  "notes": "короткое объяснение"\n'
            "}\n"
        )
        user = (
            f"Вопрос: {question}\n"
            f"Эталонный ответ: {gold}\n"
            f"Ответ системы: {_strip_citations_like_brackets(pred)}\n"
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


@dataclass(frozen=True)
class Example:
    question: str
    gold: str
    oracle_context: str
    expected_doc_id: str


def _load_examples(pkl_path: Path, limit: int | None) -> list[Example]:
    df = pd.read_pickle(pkl_path)
    q_col, a_col, ctx_col, f_col = _parse_df_columns(df)

    exs: list[Example] = []
    for _, row in df.iterrows():
        q = str(row.get(q_col) or "").strip()
        if not q:
            continue
        gold = str(row.get(a_col) or "").strip()
        ctx = str(row.get(ctx_col) or "").strip()
        fn = str(row.get(f_col) or "").strip()
        # filename may be like "12345" or "12345.txt"
        doc_id = Path(fn).stem if fn else ""
        exs.append(Example(question=q, gold=gold, oracle_context=ctx, expected_doc_id=doc_id))
        if limit is not None and len(exs) >= limit:
            break
    return exs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_pkl", default="/ru_data/ru_rag_test_dataset.pkl")
    ap.add_argument("--files_dir", default="/ru_data/files")
    ap.add_argument("--limit", type=int, default=0, help="<=0 means all")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--out", default="/out/ru_rag_eval.jsonl")
    ap.add_argument("--no_reindex", action="store_true", help="Skip wiping + re-uploading dataset docs.")
    ap.add_argument("--upload_concurrency", type=int, default=10, help="Parallelism for uploads/status/wait (threads).")
    ap.add_argument("--judge_sleep_s", type=float, default=0.2, help="Sleep between judge calls to avoid 429.")
    ap.add_argument(
        "--skip_missing_files",
        action="store_true",
        help="Skip examples whose expected doc_id file is missing in files_dir (default: enabled).",
    )
    ap.add_argument("--resume", action="store_true", help="Resume from existing --out JSONL if present.")
    args = ap.parse_args()

    pkl_path = Path(args.dataset_pkl)
    files_dir = Path(args.files_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    limit = None if args.limit is None or args.limit <= 0 else int(args.limit)
    top_k = max(1, int(args.top_k))

    examples = _load_examples(pkl_path, limit=limit)
    if not examples:
        raise SystemExit("No examples loaded from pickle (schema mismatch?)")

    # Default: skip questions whose referenced wiki file is not bundled.
    if not args.skip_missing_files:
        args.skip_missing_files = True

    missing_doc_ids: list[str] = []
    if args.skip_missing_files:
        avail: set[str] = set()
        for p in files_dir.glob("*.txt"):
            avail.add(p.stem)
        filtered: list[Example] = []
        for ex in examples:
            if ex.expected_doc_id and ex.expected_doc_id not in avail:
                missing_doc_ids.append(ex.expected_doc_id)
                continue
            filtered.append(ex)
        examples = filtered

    started = time.time()

    # Streaming output + resume support (do not lose progress on crashes).
    out_exists = out_path.exists()
    done_idx: set[int] = set()
    if args.resume and out_exists:
        try:
            with out_path.open("r", encoding="utf-8") as f:
                _ = f.readline()  # summary line
                for line in f:
                    try:
                        r = json.loads(line)
                        if isinstance(r, dict) and "i" in r:
                            done_idx.add(int(r["i"]))
                    except Exception:
                        continue
        except Exception:
            done_idx = set()

    f_out = out_path.open("a" if (args.resume and out_exists) else "w", encoding="utf-8")
    if not (args.resume and out_exists):
        # Write something immediately so partial runs leave a non-empty file.
        f_out.write(json.dumps({"summary": {"started_at": time.time(), "gate_base_url": GATE_BASE_URL}}, ensure_ascii=False) + "\n")
        f_out.flush()

    with httpx.Client(timeout=300.0) as c:
        _wait_gate_ready()

        # Indexing:
        # - If no_reindex=false: wipe and index all required docs up-front.
        # - If no_reindex=true: do NOT wipe; still upload missing docs, then wait for indexing.
        needed: list[str] = sorted({ex.expected_doc_id for ex in examples if ex.expected_doc_id})
        if not needed:
            raise SystemExit("No expected_doc_id values found in examples.")

        if not args.no_reindex:
            _delete_all_documents(c)

        # Build list of (doc_id, path) to upload (skip those already indexed when no_reindex=True).
        items: list[tuple[str, Path]] = []
        if args.no_reindex:
            # Check indexed status in parallel to avoid long sequential stalls.
            workers = max(1, int(args.upload_concurrency))
            indexed: set[str] = set()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                fut_by_id = {ex.submit(_status_is_indexed_one, doc_id=doc_id): doc_id for doc_id in needed}
                for fut in tqdm(concurrent.futures.as_completed(fut_by_id), total=len(fut_by_id), desc="Checking indexed docs"):
                    doc_id = fut_by_id[fut]
                    try:
                        if fut.result():
                            indexed.add(doc_id)
                    except Exception:
                        # treat as not indexed; we'll upload it
                        pass
            for doc_id in needed:
                if doc_id in indexed:
                    continue
                p = files_dir / f"{doc_id}.txt"
                if p.exists():
                    items.append((doc_id, p))
        else:
            for doc_id in needed:
                p = files_dir / f"{doc_id}.txt"
                if p.exists():
                    items.append((doc_id, p))

        # Upload all first (10 threads), then wait for indexing (10 threads)
        _bulk_upload_then_wait(items=items, concurrency=int(args.upload_concurrency), desc="Indexing needed ru_wiki files")

        judged = 0
        judge_correct = 0
        judge_score_sum = 0.0
        judge_errors = 0

        file_hit_1 = 0
        file_hit_3 = 0
        file_hit_5 = 0

        @retry(wait=wait_fixed(2), stop=stop_after_attempt(12))
        def _chat(payload: dict[str, Any]) -> httpx.Response:
            try:
                return c.post(f"{GATE_BASE_URL}/v1/chat", json=payload)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException) as e:
                raise RuntimeError(f"chat_transport_error:{type(e).__name__}") from e

        for idx, ex in enumerate(tqdm(examples, desc="Evaluating QA")):
            if done_idx and idx in done_idx:
                continue

            payload = {
                "query": ex.question,
                "history": [],
                "retrieval_mode": "hybrid",
                "top_k": top_k,
                "filters": None,
                "acl": [],
                # Important: keep this False for eval runs to avoid gate's extra "citation enforcement" LLM retry,
                # which can easily trigger provider rate limits (and cause 500s).
                "include_sources": False,
            }
            r = _chat(payload)
            if r.status_code != 200:
                row = {
                    "i": idx,
                    "question": ex.question,
                    "gold": ex.gold,
                    "expected_doc_id": ex.expected_doc_id,
                    "status_code": r.status_code,
                    "error": (r.text or "")[:500],
                }
                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
                continue
            j = r.json()
            pred = str(j.get("answer") or "")
            doc_ids = _extract_doc_ids_from_chat(j)
            expected = ex.expected_doc_id

            hit1 = bool(expected and len(doc_ids) >= 1 and doc_ids[0] == expected)
            hit3 = bool(expected and expected in doc_ids[:3])
            hit5 = bool(expected and expected in doc_ids[:5])
            if hit1:
                file_hit_1 += 1
            if hit3:
                file_hit_3 += 1
            if hit5:
                file_hit_5 += 1

            retrieval_json = j.get("retrieval") or {}
            overlap = _best_oracle_overlap(hits=list(retrieval_json.get("hits") or []), oracle_context=ex.oracle_context)

            judge = None
            judge_error = None
            if args.judge_sleep_s and args.judge_sleep_s > 0:
                time.sleep(float(args.judge_sleep_s))
            try:
                judge = _judge_answer(c, question=ex.question, gold=ex.gold, pred=pred)
            except Exception as e:
                judge_error = f"{type(e).__name__}: {str(e)[:200]}"
                judge_errors += 1
            if judge is not None:
                judged += 1
                try:
                    sc = float(judge.get("score_0_5"))
                except Exception:
                    sc = 0.0
                judge_score_sum += sc
                if bool(judge.get("is_correct")):
                    judge_correct += 1

            row = {
                "i": idx,
                "question": ex.question,
                "gold": ex.gold,
                "pred": pred,
                "expected_doc_id": expected,
                "retrieved_doc_ids": doc_ids,
                "file_hit@1": hit1,
                "file_hit@3": hit3,
                "file_hit@5": hit5,
                "oracle_overlap": overlap,
                "judge": judge,
                "judge_error": judge_error,
                "sources": j.get("sources") or [],
                "partial": bool(j.get("partial")),
                "degraded": j.get("degraded") or [],
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_out.flush()

    elapsed = time.time() - started
    total = len(examples)
    summary: dict[str, Any] = {
        "dataset_pkl": str(pkl_path),
        "files_dir": str(files_dir),
        "limit": limit,
        "questions": total,
        "skipped_missing_files": len(missing_doc_ids),
        "missing_doc_ids_sample": sorted(set(missing_doc_ids))[:20],
        "seconds": elapsed,
        "gate_base_url": GATE_BASE_URL,
        "top_k": top_k,
        "file_hit@1": file_hit_1 / total if total else None,
        "file_hit@3": file_hit_3 / total if total else None,
        "file_hit@5": file_hit_5 / total if total else None,
        "llm_judge_enabled": bool(JUDGE_LLM_BASE_URL and JUDGE_LLM_API_KEY),
        "llm_judge_count": judged,
        "llm_judge_errors": judge_errors,
        "llm_judge_accuracy": (judge_correct / judged) if judged else None,
        "llm_judge_avg_score_0_5": (judge_score_sum / judged) if judged else None,
        "judge_sleep_s": float(args.judge_sleep_s),
        "resume": bool(args.resume),
    }
    f_out.close()
    # Rewrite summary line without touching rows.
    try:
        lines = out_path.read_text(encoding="utf-8").splitlines()
        if lines:
            lines[0] = json.dumps({"summary": summary}, ensure_ascii=False)
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()















