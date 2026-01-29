# services/retriever.py
import asyncio
from store import BM25Store
import httpx
from dataclasses import dataclass

from store import QdrantStore
from embed_caller import Embedder


@dataclass
class SearchResult:
    text: str
    source_id: str
    chunk_index: int
    score: float


# services/generator.py


@dataclass
class Answer:
    text: str
    sources: list[str]


class ChatGenerator:
    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url}/chat/completions"
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, query: str, chunks: list) -> Answer:
        context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))

        resp = await self.client.post(self.url,
                                      # headers = {"Authorization": "Bearer sk-4bTqr1ZrEYW-Vk6CtcaZjz74cml7wSgtQneRgqj7cpY"},
                                      headers={
                                          "Authorization": "Bearer sk-FQ0o8MR5WWeLgqCTP_cZSxtpL8xbpopVCTFxSs5GQk0"},
                                      json={
                                          "model": self.model,

                                          "messages": [
                                              {"role": "system", "content": "Answer based on the provided context. Cite ONLY PROVIDED sources using square brackets and the name of the source in the brackets."},
                                              {"role": "user",
                                                  "content": f"Context:\n{context}\n\nQuestion: {query}"}
                                          ]
                                      })
        resp.raise_for_status()
        print(resp.json())
        return Answer(
            text=resp.json()["choices"][0]["message"]["content"],
            sources=[c.source_id for c in chunks]
        )

    async def close(self):
        await self.client.aclose()


class Retriever:
    def __init__(self, store, embedder):
        self.store = store
        self.embedder = embedder

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        vector = (await self.embedder.embed([query]))[0]

        results = await self.store.client.query_points(
            self.store.collection,
            query=vector,
            limit=top_k,
            with_payload=True
        )

        # print(results)

        return [
            SearchResult(
                text=r.payload["text"],
                source_id=r.payload["source_id"],
                chunk_index=r.payload["chunk_index"],
                score=r.score
            )
            for r in results.points
        ]


# services/reflector.py


@dataclass
class ReflectionResult:
    complete: bool
    missing: str | None = None
    requery: str | None = None


class Reflector:
    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url}/chat/completions"
        self.model = model
        self.client = httpx.AsyncClient(timeout=60.0)

    async def evaluate(self, query: str, chunks: list, answer: str) -> ReflectionResult:
        context = "\n".join(c.text for c in chunks)

        resp = await self.client.post(self.url,

                                      # headers = {"Authorization": "Bearer sk-4bTqr1ZrEYW-Vk6CtcaZjz74cml7wSgtQneRgqj7cpY"},
                                      # "Authorization: Bearer sk-sVxGci3uT8natU0U0xfSYuniSLrGLKu8KWkDxIKa82o"
                                      headers={
                                          "Authorization": "Bearer sk-FQ0o8MR5WWeLgqCTP_cZSxtpL8xbpopVCTFxSs5GQk0"},
                                      json={
                                          "model": self.model,
                                          "messages": [
                                              {"role": "system", "content": """Evaluate if the answer fully addresses the question based on the context.
Respond in JSON:
{"complete": true/false, "missing": "what's missing or null", "requery": "better search query or null"}"""},
                                              {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer: {answer}"}
                                          ]
                                      })
        resp.raise_for_status()

        import json
        print(resp.json())
        raw = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(raw)

        return ReflectionResult(
            complete=data.get("complete", True),
            missing=data.get("missing"),
            requery=data.get("requery")
        )

    async def close(self):
        await self.client.aclose()


# services/reranker.py


class Reranker:
    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url}/rerank"
        self.model = model
        self.client = httpx.AsyncClient()

    async def rerank(self, query: str, chunks: list, top_n: int = 5) -> list:
        resp = await self.client.post(self.url, json={
            "model": self.model,
            "query": query,
            "documents": [c.text for c in chunks]
        })
        resp.raise_for_status()

        ranked = sorted(
            resp.json()["results"], key=lambda x: x["relevance_score"], reverse=True)
        return [chunks[r["index"]] for r in ranked[:top_n]]

    async def close(self):
        await self.client.aclose()


bm25 = BM25Store("http://localhost:8905", "basa")


# services/fusion.py
def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
    rankings: list of ranked doc_id lists
    returns: [(doc_id, score)] sorted by fused score
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, qdrant, bm25, embedder):
        self.qdrant = qdrant
        self.bm25 = bm25
        self.embedder = embedder

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        vector = (await self.embedder.embed([query]))[0]

        vector_hits = await self.qdrant.client.query_points(
            self.qdrant.collection, query=vector, limit=top_k * 2, with_payload=True
        )
        bm25_hits = await self.bm25.search(query, top_k * 2)

        print(bm25_hits)

        # Payload map from vector results
        payloads = {
            f"{p.payload['source_id']}_{p.payload['chunk_index']}": p.payload
            for p in vector_hits.points
        }
        print("THIS IS NEW")

        print(payloads.keys())

        # Add BM25-only payloads
        for doc_id, _ in bm25_hits:
            if doc_id not in payloads:
                doc = await self.bm25.get(doc_id)
                if doc:
                    payloads[doc_id] = doc
                    # print("*****")
                    # print(doc)

        # RRF
        fused = rrf([
            [f"{p.payload['source_id']}_{p.payload['chunk_index']}" for p in vector_hits.points],
            [doc_id for doc_id, _ in bm25_hits]
        ])

        return [
            SearchResult(
                text=payloads[doc_id]["text"],
                source_id=payloads[doc_id]["source_id"],
                chunk_index=payloads[doc_id]["chunk_index"],
                score=score
            )
            for doc_id, score in fused[:top_k]
            if doc_id in payloads
        ]


# TODO a little closure that calls the same endpoint and then we can
# only pass a prompt and maybe some params to get the desired outcome

def one_pass_agent_closure(client: httpx.AsyncClient, chat_endpoint: str, headers: dict) -> callable:
    """
    Create a reusable async callable for a specific LLM endpoint.

    Parameters
    ----------
    client : httpx.AsyncClient
        An already‑initialized async HTTP client that will be used for the request.
    chat_endpoint : str
        The full URL of the chat/completions endpoint.
    headers : dict
        HTTP headers (e.g., Authorization) to include with each request.

    Returns
    -------
    callable
        An async function ``call(payload)`` that sends ``payload`` as JSON to the
        ``chat_endpoint`` and returns the parsed JSON response.
    """
    async def call(payload):
        async with client:
            resp = await client.post(chat_endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
    return call


async def main():

    https_client = httpx.AsyncClient(timeout=30.0)

    one_pass_agent = one_pass_agent_closure(
        https_client,
        "https://llm.c.singularitynet.io/v1/chat/completions",
        headers={
            "Authorization": "Bearer sk-FQ0o8MR5WWeLgqCTP_cZSxtpL8xbpopVCTFxSs5GQk0"},)

    query = "lalal"
    context = "ahaha"
    answer = "this is ridiculous"

    #{"complete": true/false, "missing": "what's missing or null", "requery": "better search query or null"}

    from pydantic import BaseModel, Field

    from typing import Optional


    class ReflectionResult(BaseModel):
        """
        Result returned by the LLM reflector.

        * ``complete`` – whether the answer fully satisfies the original question.
        * ``missing``  – a short description of what is still missing (if anything).
        * ``requery``  – an optional improved query to run on the next iteration.
        """

        # complete: bool = Field(
        #     default=False,
        #     # description="True when the answer fully addresses the question. DO NOT OMIT",
        # )
        # missing: str = Field(
        #     default=None,
        #     max_length=500,
        #     # description="Explanation of what is missing; omitted when ``complete`` is True.",
        # )
        # requery: str = Field(
        #     default=None,
        #     max_length=200,
        #     # description="A refined search query for the next pass; omitted when not needed.",
        # )
        complete: bool
        missing_context: str | None
        requery: str | None
        

    payload = {
                "model": "minimax/minimax-m2.1",
                "messages": [
                    {"role": "system", "content": """Evaluate if the answer fully addresses the question based on the context."""},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer: {answer}"},
                ],
                "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "reflection-result",
                            "schema": ReflectionResult.model_json_schema()
                        },
                    },
            }
    print(payload)

    resp = await one_pass_agent(payload=payload)
    print(resp)


    # em = Embedder("http://localhost:8902",
    #               model="sentence-transformers/e5-base-v2")

    # store = QdrantStore("http://localhost:8903", "basa4", 768)

    # bm25 = BM25Store("http://localhost:8905", "basa2")

    # r = HybridRetriever(store, bm25, em)

    # # r = Retriever(store, em)

    # rer = Reranker("http://localhost:8904", "BAAI/bge-reranker-v2-m3")

    # g = ChatGenerator("https://llm.c.singularitynet.io/v1",
    #                   "openai/gpt-oss-120b")
    # # g = ChatGenerator("http://localhost:8000/v1", "Qwen/Qwen2.5-1.5B-Instruct")
    # reflector = Reflector(
    #     "https://llm.c.singularitynet.io/v1", "openai/gpt-oss-120b")
    # # reflector = Reflector("http://localhost:8000/v1", "Qwen/Qwen2.5-1.5B-Instruct")
    # # Check collection size
    # # info = await store.client.get_collection(store.collection)
    # # print(f"Points in collection: {info.points_count}")

    # query = "Кто основал Microsoft"
    # final_answer = None
    # max_retries = 0
    # for _ in range(max_retries + 1):
    #     # res = await r.search("Появление Microsoft в популярной культуре", top_k=5)
    #     # a = [print(r.text) for r in res]
    #     # for r in res:
    #     #    print(r.score)
    #     #    print(r.text)

    #     chunks = await r.search(query, top_k=3)  # retrieve more
    #     results = await rer.rerank(query, chunks, top_n=5)  # narrow down

    #     for res in results:
    #         print(res.text)
    #         print("-----")
    #     answer = await g.generate(query, results)

    #     reflection = await reflector.evaluate(query, results, answer.text)
    #     print(reflection)

    #     if reflection.complete:
    #         final_answer = answer
    #         break

    #     query = reflection.requery or query

    # # return answer  # best effort
    # # print(reflection)

    # # if reflection.complete:
    # #     print("DONE WOW WOW WOW")
    # print(final_answer or answer)
    # # await r.close()
    # await em.close()
    # await bm25.close()
    # await store.close()
    # await reflector.close()
    # await g.close()
    await https_client.aclose()


asyncio.run(main())
