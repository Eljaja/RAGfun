from __future__ import annotations

import uuid

# Stable namespace UUID for deterministic mapping chunk_id -> UUIDv5
_NS = uuid.UUID("b7bb8d8d-5f2c-4bb3-9c57-20fd9f88f0d8")


def point_id_for_chunk_id(chunk_id: str) -> str:
    """
    Qdrant point IDs must be either an unsigned integer or UUID.
    We keep the business identifier `chunk_id` in payload and use UUIDv5 as point id.
    """
    return str(uuid.uuid5(_NS, chunk_id))




