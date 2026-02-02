#!/usr/bin/env python3
import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from fastapi import FastAPI, Query
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "images"
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 60.0

app = FastAPI(title="Spiny API")

model = SentenceTransformer(MODEL_NAME)
qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)


def _rest_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=QDRANT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _qdrant_search(query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
    # Compatible across qdrant-client versions
    if hasattr(qc, "search"):
        hits = qc.search(collection_name=COLLECTION, query_vector=query_vector, limit=limit, with_payload=True)
        return [{"score": getattr(h, "score", None), "payload": getattr(h, "payload", {})} for h in hits]
    if hasattr(qc, "search_points"):
        hits = qc.search_points(collection_name=COLLECTION, query_vector=query_vector, limit=limit, with_payload=True)
        return [{"score": getattr(h, "score", None), "payload": getattr(h, "payload", {})} for h in hits]

    data = _rest_post(f"/collections/{COLLECTION}/points/search", {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    })
    res = data.get("result", [])
    return [{"score": h.get("score"), "payload": h.get("payload") or {}} for h in res]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/search_images")
def search_images(q: str = Query(..., min_length=1), k: int = 10):
    k = max(1, min(int(k), 50))
    qvec = model.encode(q, normalize_embeddings=True).tolist()
    hits = _qdrant_search(qvec, k)

    results = []
    for h in hits:
        p = h.get("payload") or {}
        sha = p.get("sha256")
        if not sha:
            continue
        results.append({
            "score": h.get("score"),
            "sha256": sha,
            "path": p.get("path"),
            "thumb": p.get("thumb"),
        })

    return {"query": q, "k": k, "results": results}
