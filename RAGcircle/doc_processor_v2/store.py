# services/store.py
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStore:
    def __init__(self, url: str, collection: str, dimension: int):
        self.client = AsyncQdrantClient(url=url)
        self.collection = collection
        self.dimension = dimension
    
    async def ensure_collection(self):
        exists = await self.client.collection_exists(self.collection)
        if not exists:
            await self.client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )
    
    async def upsert(self, chunks: list, vectors: list[list[float]]):
        points = [
            PointStruct(
                id=hash(f"{c.source_id}_{c.chunk_index}") & 0x7FFFFFFFFFFFFFFF,  # qdrant wants positive int or uuid
                vector=v,
                payload={"text": c.text, "source_id": int(c.source_id), "chunk_index": c.chunk_index}
            )
            for c, v in zip(chunks, vectors)
        ]
        await self.client.upsert(self.collection, points)
    
    async def close(self):
        await self.client.close()



# services/bm25_store.py
from opensearchpy import AsyncOpenSearch

class BM25Store:
    def __init__(self, url: str, index: str):
        self.client = AsyncOpenSearch(hosts=[url], use_ssl=False)
        self.index = index
    
    async def ensure_index(self):
        exists = await self.client.indices.exists(self.index)
        if not exists:
            if not exists:
                await self.client.indices.create(self.index, body={
                    "settings": {
                        "analysis": {
                            "filter": {
                                "russian_stemmer": {"type": "stemmer", "language": "russian"}
                            },
                            "analyzer": {
                                "russian": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "russian_stemmer"]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "russian"},
                            "source_id": {"type": "keyword"},
                            "chunk_index": {"type": "integer"}
                        }
                    }
                })
    
    async def upsert(self, chunks: list):
        for c in chunks:
            await self.client.index(
                index=self.index,
                id=f"{c.source_id}_{c.chunk_index}",
                body={"text": c.text, "source_id": c.source_id, "chunk_index": c.chunk_index}
            )
    
    async def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        resp = await self.client.search(
            index=self.index,
            body={"query": {"match": {"text": query}}, "size": top_k}
        )
        return [
            (hit["_id"], hit["_score"])
            for hit in resp["hits"]["hits"]
        ]
    
    async def close(self):
        await self.client.close()

    async def get(self, doc_id: str) -> dict | None:
        
        resp = await self.client.get(index=self.index, id=doc_id)
        print("THIS IS RESP", resp)
        return resp["_source"]
        