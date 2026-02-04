#!/usr/bin/env python3
import os
import json
import time
import threading
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ----------------------------
# Config
# ----------------------------
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_TIMEOUT = float(os.environ.get("QDRANT_TIMEOUT", "60.0"))

IMAGES_COLLECTION = os.environ.get("SPINY_IMAGES_COLLECTION", "images")
IMAGES_EMBED_MODEL = os.environ.get("SPINY_IMAGES_EMBED_MODEL", "sentence-transformers/clip-ViT-B-32")

DOCS_COLLECTION = os.environ.get("SPINY_DOCS_COLLECTION", "docs")
DOCS_EMBED_MODEL = os.environ.get("SPINY_DOCS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# App + clients
# ----------------------------
app = FastAPI(title="Spiny API")

qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)

_img_model: Optional[SentenceTransformer] = None
_docs_model: Optional[SentenceTransformer] = None
_img_lock = threading.Lock()
_docs_lock = threading.Lock()

# ----------------------------
# Helpers
# ----------------------------
def _rest_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Raw REST call to Qdrant (kept because it’s proven stable on older clients)."""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=QDRANT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_img_model() -> SentenceTransformer:
    global _img_model
    if _img_model is None:
        with _img_lock:
            if _img_model is None:
                _img_model = SentenceTransformer(IMAGES_EMBED_MODEL)
    return _img_model


def get_docs_model() -> SentenceTransformer:
    global _docs_model
    if _docs_model is None:
        with _docs_lock:
            if _docs_model is None:
                _docs_model = SentenceTransformer(DOCS_EMBED_MODEL)
    return _docs_model


def qdrant_search(collection: str, vector: List[float], k: int):
    """
    Compatibility across qdrant-client versions.
    Returns iterable of hits with .score and .payload.
    """
    if hasattr(qc, "search"):
        return qc.search(collection_name=collection, query_vector=vector, limit=k, with_payload=True)

    if hasattr(qc, "search_points"):
        resp = qc.search_points(collection_name=collection, vector=vector, limit=k, with_payload=True)
        return getattr(resp, "result", resp)

    if hasattr(qc, "query_points"):
        resp = qc.query_points(collection_name=collection, query=vector, limit=k, with_payload=True)
        return getattr(resp, "points", resp)

    raise RuntimeError("Unsupported qdrant-client: no search method found")


def safe_error(where: str, e: Exception, status: int = 500):
    return JSONResponse(
        status_code=status,
        content={"ok": False, "where": where, "error": f"{type(e).__name__}: {e}"},
    )


def _clip_embed_text(q: str) -> List[float]:
    model = get_img_model()
    return model.encode([q], normalize_embeddings=True)[0].tolist()


def _docs_embed_text(q: str) -> List[float]:
    model = get_docs_model()
    return model.encode([q], normalize_embeddings=True)[0].tolist()


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    # Keep this lightweight and *never* model-load here.
    return {"ok": True, "ts": time.time()}


@app.get("/search_images")
def search_images(q: str = Query(...), k: int = 24):
    q = (q or "").strip()
    if not q:
        return {"query": q, "k": k, "results": []}

    try:
        v = _clip_embed_text(q)
        hits = qdrant_search(IMAGES_COLLECTION, v, k)

        results = []
        for h in hits:
            p = getattr(h, "payload", None) or {}
            results.append(
                {
                    "score": float(getattr(h, "score", 0.0)),
                    "path": p.get("path", ""),
                    "thumb": p.get("thumb", ""),
                    "sha256": p.get("sha256", ""),
                }
            )
        return {"query": q, "k": k, "results": results}
    except Exception as e:
        return safe_error("search_images", e)


@app.get("/search_docs")
def search_docs(q: str = Query(...), k: int = 8):
    q = (q or "").strip()
    if not q:
        return {"query": q, "k": k, "results": []}

    try:
        v = _docs_embed_text(q)
        hits = qdrant_search(DOCS_COLLECTION, v, k)

        results = []
        for h in hits:
            p = getattr(h, "payload", None) or {}
            txt = (p.get("text") or "").replace("\n", " ").strip()
            results.append(
                {
                    "score": float(getattr(h, "score", 0.0)),
                    "path": p.get("path", ""),
                    "page": int(p.get("page", 0) or 0),
                    "chunk": int(p.get("chunk", 0) or 0),
                    "snippet": txt[:300] + ("…" if len(txt) > 300 else ""),
                    "doc_sha": p.get("doc_sha", ""),
                }
            )
        return {"query": q, "k": k, "results": results}
    except Exception as e:
        return safe_error("search_docs", e)

