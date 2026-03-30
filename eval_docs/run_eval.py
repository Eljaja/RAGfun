#!/usr/bin/env python3
"""
Run RAG evaluation via API: chat, agent-search, deep-research.
Collects answers and produces evaluation report.
Progress: tail -f eval_progress.txt
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

API_KEY = os.getenv("ODS_API_KEY", "")
PROGRESS_FILE = Path(__file__).parent / "eval_progress.txt"
BASE = "http://localhost:8092"
AGENT_BASE = "http://localhost:8093"
DEEP_BASE = "http://localhost:8094"
FILTERS = {"project_ids": ["eval"]}

QUESTIONS = [
    "Что такое RAGfun?",
    "Какой заголовок используется для ODS API ключа?",
    "На каком порту работает UI?",
    "Сколько режимов retrieval поддерживается?",
    "Что такое ODS?",
    "Что такое HyDE в agent-search?",
    "Какой фреймворк использует deep-research?",
    "Чем agent-search отличается от deep-research?",
    "Перечисли основные сервисы RAGfun.",
    "Как создать тенанта?",
]


def curl_stream(url: str, headers: dict, body: dict, timeout: int = 120) -> str:
    cmd = [
        "curl", "-sS", "-X", "POST", url,
        "-H", "Content-Type: application/json",
        "-H", f"X-ODS-API-KEY: {API_KEY}",
        "-d", json.dumps(body),
        "--max-time", str(timeout),
    ]
    for k, v in headers.items():
        cmd.extend(["-H", f"{k}: {v}"])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
    if result.returncode != 0:
        return f"[ERROR: {result.stderr or result.stdout[:200]}]"
    return result.stdout


def extract_answer_sse(text: str) -> tuple[str, list]:
    """Extract answer and context from SSE stream."""
    answer = ""
    context = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:].strip())
                if data.get("type") == "done":
                    answer = data.get("answer", "")
                    context = data.get("context", [])
                    break
                elif data.get("type") == "token" and data.get("content"):
                    answer += data.get("content", "")
                elif data.get("type") == "error":
                    return f"[ERROR: {data.get('error', '')}]", []
            except json.JSONDecodeError:
                pass
    return answer.strip(), context


def call_chat(q: str) -> tuple[str, list]:
    body = {"query": q, "include_sources": True, "filters": FILTERS}
    out = curl_stream(f"{BASE}/v1/chat/stream", {}, body, timeout=60)
    return extract_answer_sse(out)


def call_agent(q: str) -> tuple[str, list]:
    body = {"query": q, "include_sources": True, "filters": FILTERS}
    out = curl_stream(f"{AGENT_BASE}/v1/agent/stream", {}, body, timeout=90)
    return extract_answer_sse(out)


def call_deep(q: str) -> tuple[str, list]:
    body = {"query": q, "include_sources": True, "filters": FILTERS}
    out = curl_stream(f"{DEEP_BASE}/v1/deep-research/stream", {}, body, timeout=180)
    return extract_answer_sse(out)


def evaluate(answer: str, context: list, q: str) -> dict:
    """Simple heuristic evaluation (1-5)."""
    def score_relevance(a: str, q_lower: str) -> int:
        if not a or a.startswith("[ERROR"):
            return 1
        a_lower = a.lower()
        q_words = set(q_lower.split()) - {"что", "какой", "как", "какие", "чем", "где", "когда", "объясни", "назови", "перечисли"}
        matches = sum(1 for w in q_words if len(w) > 2 and w in a_lower)
        if matches >= len(q_words) * 0.5 and len(a) > 50:
            return 5 if len(a) > 100 else 4
        if matches > 0:
            return 3
        return 2

    def score_correctness(a: str) -> int:
        if not a or a.startswith("[ERROR"):
            return 1
        # No hallucination check; assume reasonable
        return 4 if len(a) > 30 else 3

    def score_completeness(a: str, q: str) -> int:
        if not a or a.startswith("[ERROR"):
            return 1
        if "перечисли" in q.lower() or "какие" in q.lower():
            return 5 if "," in a or ";" in a or " и " in a else 3
        return 4 if len(a) > 80 else 3

    def score_citation(a: str, ctx: list) -> int:
        if not ctx:
            return 3  # no context to cite
        refs = re.findall(r"\[\d+\]", a)
        if not refs:
            return 2
        return 5 if len(refs) >= min(2, len(ctx)) else 4

    q_lower = q.lower()
    return {
        "relevance": score_relevance(answer, q_lower),
        "correctness": score_correctness(answer),
        "completeness": score_completeness(answer, q),
        "citation": score_citation(answer, context),
    }


def write_progress(msg: str, append: bool = True):
    with open(PROGRESS_FILE, "a" if append else "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        f.flush()


def main():
    if not API_KEY:
        raise SystemExit("Set ODS_API_KEY before running eval_docs/run_eval.py")
    total = len(QUESTIONS) * 3  # chat, agent, deep per question
    done = 0
    write_progress(f"START: {len(QUESTIONS)} questions × 3 modes = {total} calls", append=False)
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        write_progress(f"Q{i}/{len(QUESTIONS)}: {q[:50]}...")
        print(f"[{i}/{len(QUESTIONS)}] {q[:50]}...", flush=True)
        row = {"q": q, "chat": {}, "agent": {}, "deep": {}}
        for mode, fn in [("chat", call_chat), ("agent", call_agent), ("deep", call_deep)]:
            try:
                write_progress(f"  → {mode}...")
                ans, ctx = fn(q)
                row[mode] = {"answer": ans[:500], "scores": evaluate(ans, ctx, q)}
                done += 1
                s = row[mode]["scores"]
                write_progress(f"  ✓ {mode} R={s['relevance']} C={s['correctness']} ({done}/{total})")
            except Exception as e:
                row[mode] = {"answer": f"[EXCEPTION: {e}]", "scores": {"relevance": 1, "correctness": 1, "completeness": 1, "citation": 1}}
                done += 1
                write_progress(f"  ✗ {mode} FAILED: {e} ({done}/{total})")
        results.append(row)

    out_path = Path(__file__).parent / "EVAL_RESULTS.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Результаты оценки RAG (автоматический прогон)\n\n")
        f.write("| # | Вопрос | Режим | R | C | Comp | Cit | Ответ (сокращённо) |\n")
        f.write("|---|--------|-------|---|---|------|-----|--------------------|\n")
        for i, r in enumerate(results, 1):
            for mode in ["chat", "agent", "deep"]:
                s = r[mode]["scores"]
                ans_short = (r[mode]["answer"][:80] + "…") if len(r[mode]["answer"]) > 80 else r[mode]["answer"]
                ans_short = ans_short.replace("|", " ").replace("\n", " ")
                f.write(f"| {i} | {r['q'][:40]}… | {mode} | {s['relevance']} | {s['correctness']} | {s['completeness']} | {s['citation']} | {ans_short} |\n")
        f.write("\n## Сводка (средние)\n\n")
        for mode in ["chat", "agent", "deep"]:
            avg = {k: sum(r[mode]["scores"][k] for r in results) / len(results) for k in ["relevance", "correctness", "completeness", "citation"]}
            f.write(f"- **{mode}**: R={avg['relevance']:.1f} C={avg['correctness']:.1f} Comp={avg['completeness']:.1f} Cit={avg['citation']:.1f}\n")

    write_progress(f"DONE. Results: {out_path}")
    print(f"\nDone. Results: {out_path}")


if __name__ == "__main__":
    main()
