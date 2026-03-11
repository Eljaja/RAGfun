#!/usr/bin/env python3
"""
Quick script to check available collections/indexes in Qdrant and OpenSearch.
"""
import asyncio
import argparse
import textwrap
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter
from opensearchpy import AsyncOpenSearch


async def check_qdrant_collections(url: str = "http://localhost:8903"):
    """List all collections in Qdrant."""
    print(f"\n{'='*60}")
    print(f"Qdrant Collections (URL: {url})")
    print(f"{'='*60}")
    
    try:
        client = AsyncQdrantClient(url=url)
        collections = await client.get_collections()
        
        if not collections.collections:
            print("No collections found.")
        else:
            print(f"\nFound {len(collections.collections)} collection(s):\n")
            for coll in collections.collections:
                # Get collection info for details
                info = await client.get_collection(coll.name)
                print(f"  📦 {coll.name}")
                print(f"     Points: {info.points_count:,}")
                print(f"     Vectors: {info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else 'N/A'}")
                print()
        
        await client.close()
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")


async def check_opensearch_indices(url: str = "http://localhost:8905"):
    """List all indices in OpenSearch."""
    print(f"\n{'='*60}")
    print(f"OpenSearch Indices (URL: {url})")
    print(f"{'='*60}")
    
    try:
        client = AsyncOpenSearch(hosts=[url], use_ssl=False)
        
        # Get all indices
        indices = await client.indices.get_alias(index="*")
        
        if not indices:
            print("No indices found.")
        else:
            print(f"\nFound {len(indices)} index(ices):\n")
            for index_name in sorted(indices.keys()):
                # Get index stats
                stats = await client.indices.stats(index=index_name)
                doc_count = stats["indices"][index_name]["total"]["docs"]["count"]
                size = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]
                size_mb = size / (1024 * 1024)
                
                print(f"  📚 {index_name}")
                print(f"     Documents: {doc_count:,}")
                print(f"     Size: {size_mb:.2f} MB")
                print()
        
        await client.close()
    except Exception as e:
        print(f"❌ Error connecting to OpenSearch: {e}")


async def list_qdrant_doc_ids(collection: str, url: str = "http://localhost:8903"):
    """List all unique doc_ids in a Qdrant collection."""
    print(f"\n{'='*60}")
    print(f"Qdrant Collection: {collection}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    try:
        client = AsyncQdrantClient(url=url)
        
        # Check if collection exists
        exists = await client.collection_exists(collection)
        if not exists:
            print(f"❌ Collection '{collection}' does not exist.")
            await client.close()
            return
        
        # Scroll through all points to get doc_ids
        doc_ids = set()
        offset = None
        limit = 1000
        
        while True:
            result = await client.scroll(
                collection,
                limit=limit,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False
            )
            
            points, next_offset = result
            
            if not points:
                break
            
            for point in points:
                if point.payload and "doc_id" in point.payload:
                    doc_ids.add(point.payload["doc_id"])
            
            if next_offset is None:
                break
            offset = next_offset
        
        await client.close()
        
        if not doc_ids:
            print("No doc_ids found in this collection.")
        else:
            print(f"\nFound {len(doc_ids)} unique doc_id(s):\n")
            for doc_id in sorted(doc_ids):
                print(f"  📄 {doc_id}")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def list_opensearch_doc_ids(index: str, url: str = "http://localhost:8905"):
    """List all unique doc_ids in an OpenSearch index."""
    print(f"\n{'='*60}")
    print(f"OpenSearch Index: {index}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    try:
        client = AsyncOpenSearch(hosts=[url], use_ssl=False)
        
        # Check if index exists
        exists = await client.indices.exists(index=index)
        if not exists:
            print(f"❌ Index '{index}' does not exist.")
            await client.close()
            return
        
        # Use aggregation to get unique doc_ids
        doc_ids = set()
        scroll_size = 1000
        
        # Initial search with scroll
        response = await client.search(
            index=index,
            body={
                "size": 0,
                "aggs": {
                    "unique_doc_ids": {
                        "terms": {
                            "field": "doc_id",
                            "size": 10000  # Adjust if you have more than 10k unique doc_ids
                        }
                    }
                }
            }
        )
        
        if "aggregations" in response and "unique_doc_ids" in response["aggregations"]:
            buckets = response["aggregations"]["unique_doc_ids"]["buckets"]
            doc_ids = {bucket["key"] for bucket in buckets}
        
        await client.close()
        
        if not doc_ids:
            print("No doc_ids found in this index.")
        else:
            print(f"\nFound {len(doc_ids)} unique doc_id(s):\n")
            for doc_id in sorted(doc_ids):
                print(f"  📄 {doc_id}")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def show_qdrant_chunk(collection: str, chunk_num: int = 1, url: str = "http://localhost:8903"):
    """Show a specific chunk (by 1-based position) from a Qdrant collection."""
    print(f"\n{'='*60}")
    print(f"Qdrant Chunk #{chunk_num} from: {collection}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    try:
        client = AsyncQdrantClient(url=url)

        exists = await client.collection_exists(collection)
        if not exists:
            print(f"❌ Collection '{collection}' does not exist.")
            await client.close()
            return

        # Scroll to the requested chunk position
        offset = None
        seen = 0
        target = chunk_num  # 1-based

        while True:
            points, next_offset = await client.scroll(
                collection,
                limit=min(target - seen, 100),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                print(f"❌ Only {seen} chunk(s) in collection — requested #{chunk_num}.")
                await client.close()
                return

            seen += len(points)
            if seen >= target:
                point = points[-(seen - target + 1)]
                _print_chunk_payload(point.payload, point_id=point.id)
                await client.close()
                return

            if next_offset is None:
                print(f"❌ Only {seen} chunk(s) in collection — requested #{chunk_num}.")
                await client.close()
                return
            offset = next_offset

    except Exception as e:
        print(f"❌ Error: {e}")


async def show_opensearch_chunk(index: str, chunk_num: int = 1, url: str = "http://localhost:8905"):
    """Show a specific chunk (by 1-based position) from an OpenSearch index."""
    print(f"\n{'='*60}")
    print(f"OpenSearch Chunk #{chunk_num} from: {index}")
    print(f"URL: {url}")
    print(f"{'='*60}")

    try:
        client = AsyncOpenSearch(hosts=[url], use_ssl=False)

        exists = await client.indices.exists(index=index)
        if not exists:
            print(f"❌ Index '{index}' does not exist.")
            await client.close()
            return

        response = await client.search(
            index=index,
            body={
                "from": chunk_num - 1,
                "size": 1,
                "sort": [{"chunk_index": {"order": "asc"}}, "_score"],
            },
        )

        hits = response["hits"]["hits"]
        if not hits:
            total = response["hits"]["total"]["value"]
            print(f"❌ Only {total} chunk(s) in index — requested #{chunk_num}.")
            await client.close()
            return

        hit = hits[0]
        _print_chunk_payload(hit["_source"], point_id=hit["_id"])
        await client.close()

    except Exception as e:
        print(f"❌ Error: {e}")


def _print_chunk_payload(payload: dict, *, point_id=None):
    """Pretty-print a chunk payload."""
    print()
    if point_id is not None:
        print(f"  🆔 point_id:    {point_id}")
    for key in ("chunk_id", "db_id", "doc_id", "chunk_index", "source", "uri", "page", "content_hash"):
        if key in payload:
            print(f"  📎 {key:14s} {payload[key]}")
    text = payload.get("text", "")
    print(f"\n  📝 text ({len(text)} chars):\n")
    print(textwrap.indent(text, "     "))
    print()


async def main():
    """Check both stores or list doc_ids for a specific collection/index."""
    parser = argparse.ArgumentParser(
        description="Check Qdrant collections and OpenSearch indices, or list doc_ids for a specific collection/index."
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Qdrant collection name to list doc_ids for"
    )
    parser.add_argument(
        "--index",
        type=str,
        help="OpenSearch index name to list doc_ids for"
    )
    parser.add_argument(
        "--show-chunk",
        type=int,
        nargs="?",
        const=1,
        default=None,
        metavar="N",
        help="Show chunk at 1-based position N (default: 1). Use with --collection or --index."
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:8903",
        help="Qdrant URL (default: http://localhost:8903)"
    )
    parser.add_argument(
        "--opensearch-url",
        type=str,
        default="http://localhost:8905",
        help="OpenSearch URL (default: http://localhost:8905)"
    )
    
    args = parser.parse_args()
    
    if args.show_chunk is not None:
        if args.collection:
            await show_qdrant_chunk(args.collection, args.show_chunk, args.qdrant_url)
        elif args.index:
            await show_opensearch_chunk(args.index, args.show_chunk, args.opensearch_url)
        else:
            print("❌ --show-chunk requires --collection or --index")
    elif args.collection:
        await list_qdrant_doc_ids(args.collection, args.qdrant_url)
    elif args.index:
        await list_opensearch_doc_ids(args.index, args.opensearch_url)
    else:
        # Default: list all collections and indices
        await check_qdrant_collections(args.qdrant_url)
        await check_opensearch_indices(args.opensearch_url)
        print(f"\n{'='*60}\n")
        print("💡 Tip: Use --collection <name> or --index <name> to list doc_ids for a specific collection/index")
        print("💡 Tip: Use --collection <name> --show-chunk [N] to display a chunk")


if __name__ == "__main__":
    asyncio.run(main())