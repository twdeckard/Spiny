#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = os.environ.get("SPINY_DOCS_COLLECTION", "docs")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
EMBED_MODEL = os.environ.get("SPINY_DOCS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def snippet(s: str, n: int = 240) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def qdrant_search(qc: QdrantClient, vector: List[float], k: int):
    """
    Compatibility shim across qdrant-client versions:
      - new: qc.search(...)
      - older: qc.search_points(...).result
      - newer alt: qc.query_points(...).points
    Returns an iterable of hit objects with .score and .payload
    """
    if hasattr(qc, "search"):
        return qc.search(collection_name=COLLECTION, query_vector=vector, limit=k, with_payload=True)

    if hasattr(qc, "search_points"):
        resp = qc.search_points(collection_name=COLLECTION, vector=vector, limit=k, with_payload=True)
        # Some versions return the list directly, some return an object with .result
        return getattr(resp, "result", resp)

    if hasattr(qc, "query_points"):
        resp = qc.query_points(collection_name=COLLECTION, query=vector, limit=k, with_payload=True)
        return getattr(resp, "points", resp)

    raise RuntimeError("Unsupported qdrant-client: no search method found on QdrantClient")


def search(q: str, k: int) -> List[Dict[str, Any]]:
    embed = SentenceTransformer(EMBED_MODEL)
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)
    v = embed.encode([q], normalize_embeddings=True)[0].tolist()

    hits = qdrant_search(qc, vector=v, k=k)

    out = []
    for h in hits:
        p = getattr(h, "payload", None) or {}
        out.append(
            {
                "score": float(getattr(h, "score", 0.0)),
                "path": p.get("path", ""),
                "page": int(p.get("page", 0) or 0),
                "chunk": int(p.get("chunk", 0) or 0),
                "text": p.get("text", ""),
                "doc_sha": p.get("doc_sha", ""),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=8)
    args = ap.parse_args()

    q = " ".join(args.query).strip()
    hits = search(q, k=args.k)

    if not hits:
        print("No results.")
        return

    print(f'Query: "{q}"\n')
    for i, h in enumerate(hits, 1):
        print(f"{i}. score={h['score']:.4f} page={h['page']} chunk={h['chunk']}")
        print(f"   {h['path']}")
        print(f"   {snippet(h['text'])}")
        print("")


if __name__ == "__main__":
    main()

