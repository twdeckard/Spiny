#!/usr/bin/env python3
"""
search_images.py
----------------
Text -> image search against Qdrant using the same CLIP model used for indexing.

This script is compatible with multiple qdrant-client versions:
- Newer: QdrantClient.search(...)
- Older: QdrantClient.search_points(...)
- Fallback: raw REST call to /collections/{collection}/points/search

Usage:
  python3 search_images.py "dog on the beach"
"""

import json
import sys
import urllib.request
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "images"
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 60.0


def _qdrant_rest_search(query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
    """Fallback search via Qdrant REST API."""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION}/points/search"
    payload = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=QDRANT_TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    # Qdrant returns {"result": [...], "status": "ok", ...}
    return data.get("result", [])


def _normalize_hits(hits: Any) -> List[Dict[str, Any]]:
    """
    Normalize qdrant-client response objects or REST JSON into
    a list of dicts: { "score": float, "payload": dict, "id": str }
    """
    out: List[Dict[str, Any]] = []

    # REST path: list of dicts already
    if isinstance(hits, list) and hits and isinstance(hits[0], dict):
        for h in hits:
            out.append(
                {
                    "score": h.get("score"),
                    "payload": h.get("payload") or {},
                    "id": h.get("id"),
                }
            )
        return out

    # qdrant-client path: list of ScoredPoint-like objects
    if isinstance(hits, list):
        for h in hits:
            payload = getattr(h, "payload", None) or {}
            score = getattr(h, "score", None)
            pid = getattr(h, "id", None)
            out.append({"score": score, "payload": payload, "id": pid})
        return out

    return out


def search(query: str, limit: int = 10) -> None:
    model = SentenceTransformer(MODEL_NAME)
    qvec = model.encode(query, normalize_embeddings=True).tolist()

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)

    # Try newest API first
    hits = None
    if hasattr(qc, "search"):
        hits = qc.search(collection_name=COLLECTION, query_vector=qvec, limit=limit, with_payload=True)
    elif hasattr(qc, "search_points"):
        hits = qc.search_points(collection_name=COLLECTION, query_vector=qvec, limit=limit, with_payload=True)
    else:
        hits = _qdrant_rest_search(qvec, limit)

    results = _normalize_hits(hits)

    if not results:
        print("No hits (or could not parse response).")
        return

    for i, r in enumerate(results, 1):
        p = r["payload"] or {}
        print(f"{i:02d} score={r['score']:.4f}  {p.get('path')}")
        if p.get("thumb"):
            print(f"    thumb: {p.get('thumb')}")
        if p.get("taken_ts"):
            print(f"    taken_ts: {p.get('taken_ts')}")
        if p.get("camera_model"):
            print(f"    camera: {p.get('camera_model')}")
        if p.get("sha256"):
            print(f"    sha256: {p.get('sha256')}")
        print()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "a dog on a beach"
    search(query)
