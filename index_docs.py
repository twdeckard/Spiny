#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer


COLLECTION = os.environ.get("SPINY_DOCS_COLLECTION", "docs")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# Where PDFs live (your requirement)
DOCS_DIR = os.environ.get("SPINY_DOCS_DIR", "/mnt/moltbot/corpus/docs")

# Keep a tiny manifest so we can delete missing docs safely
MANIFEST_PATH = os.environ.get("SPINY_DOCS_MANIFEST", "/ssd/moltbot/state/docs_manifest.json")

# Embedding model (small + solid for English)
EMBED_MODEL = os.environ.get("SPINY_DOCS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking
CHUNK_CHARS = int(os.environ.get("SPINY_DOCS_CHUNK_CHARS", "1100"))
CHUNK_OVERLAP = int(os.environ.get("SPINY_DOCS_CHUNK_OVERLAP", "200"))

# Limits
MAX_PAGES = int(os.environ.get("SPINY_DOCS_MAX_PAGES", "0"))  # 0 = no limit


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_parent_dir(p: str) -> None:
    os.makedirs(os.path.dirname(p), exist_ok=True)


def load_manifest() -> Dict[str, dict]:
    if not os.path.exists(MANIFEST_PATH):
        return {}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(m: Dict[str, dict]) -> None:
    ensure_parent_dir(MANIFEST_PATH)
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, sort_keys=True)
    os.replace(tmp, MANIFEST_PATH)


def is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def list_pdfs_flat(dirpath: str) -> List[str]:
    if not os.path.isdir(dirpath):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(dirpath)):
        p = os.path.join(dirpath, name)
        if os.path.isfile(p) and is_pdf(name):
            out.append(p)
    return out


def normalize_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_chars: int, overlap: int) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_chars - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_chars)

        # Try to end on a sentence boundary if possible
        window = text[start:end]
        cut = max(window.rfind(". "), window.rfind("\n"))
        if cut > int(chunk_chars * 0.6):
            end = start + cut + 1
            window = text[start:end]

        chunks.append(window.strip())
        start += step
    return [c for c in chunks if c]


def stable_point_uuid(doc_sha: str, page: int, chunk_idx: int) -> str:
    # Deterministic UUID so upserts are stable across re-runs
    ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace
    return str(uuid.uuid5(ns, f"{doc_sha}:{page}:{chunk_idx}"))


def ensure_collection(qc: QdrantClient, dim: int) -> None:
    existing = {c.name for c in qc.get_collections().collections}
    if COLLECTION in existing:
        return

    qc.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        optimizers_config=qm.OptimizersConfigDiff(memmap_threshold=20000),
    )


def delete_doc_by_sha(qc: QdrantClient, doc_sha: str) -> None:
    qc.delete(
        collection_name=COLLECTION,
        points_selector=qm.Filter(
            must=[qm.FieldCondition(key="doc_sha", match=qm.MatchValue(value=doc_sha))]
        ),
    )


def extract_pdf_pages(path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(path)
    pages: List[Tuple[int, str]] = []
    n = doc.page_count
    lim = n if MAX_PAGES <= 0 else min(n, MAX_PAGES)
    for i in range(lim):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text = normalize_text(text)
        pages.append((i + 1, text))  # 1-based for humans
    doc.close()
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default=DOCS_DIR, help="Directory containing PDFs (flat)")
    ap.add_argument("--delete-missing", action="store_true", help="Delete docs that are no longer present")
    ap.add_argument("--reindex-all", action="store_true", help="Reindex even if sha unchanged")
    ap.add_argument("--batch", type=int, default=64, help="Upsert batch size")
    args = ap.parse_args()

    docs_dir = args.docs_dir

    pdfs = list_pdfs_flat(docs_dir)
    print(f"[docs] found {len(pdfs)} pdf(s) in {docs_dir}")

    manifest = load_manifest()

    # Optional: delete docs that disappeared from the directory
    if args.delete_missing:
        current_paths = set(pdfs)
        missing_shas = []
        for doc_sha, rec in list(manifest.items()):
            old_path = rec.get("path")
            if old_path and old_path not in current_paths:
                missing_shas.append(doc_sha)
        if missing_shas:
            qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0)
            for doc_sha in missing_shas:
                print(f"[docs] deleting missing doc_sha={doc_sha[:12]}…")
                try:
                    delete_doc_by_sha(qc, doc_sha)
                except Exception as e:
                    print(f"[docs] WARN: delete failed for {doc_sha[:12]}…: {e}")
                manifest.pop(doc_sha, None)
            save_manifest(manifest)

    # Initialize embedder + qdrant
    embed = SentenceTransformer(EMBED_MODEL)
    dim = int(embed.get_sentence_embedding_dimension())
    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120.0)

    ensure_collection(qc, dim=dim)

    total_points = 0
    t0 = time.time()

    for path in pdfs:
        try:
            doc_sha = sha256_file(path)
        except Exception as e:
            print(f"[docs] WARN: cannot hash {path}: {e}")
            continue

        prev = manifest.get(doc_sha)
        if prev and (not args.reindex_all):
            # same doc_sha implies content unchanged (sha is content hash)
            # but path might change; update path
            if prev.get("path") != path:
                prev["path"] = path
                manifest[doc_sha] = prev
                save_manifest(manifest)
            print(f"[docs] skip unchanged {os.path.basename(path)}")
            continue

        # If this doc_sha existed under a different key previously (rare),
        # we just insert new. If same sha reappears, we delete then insert cleanly.
        if doc_sha in manifest:
            print(f"[docs] replacing existing doc_sha={doc_sha[:12]}…")
            try:
                delete_doc_by_sha(qc, doc_sha)
            except Exception as e:
                print(f"[docs] WARN: delete prior points failed: {e}")

        print(f"[docs] indexing {os.path.basename(path)}")
        try:
            pages = extract_pdf_pages(path)
        except Exception as e:
            print(f"[docs] WARN: cannot read {path}: {e}")
            continue

        # Build points
        points: List[qm.PointStruct] = []
        for page_num, page_text in pages:
            if not page_text:
                continue
            chunks = chunk_text(page_text, CHUNK_CHARS, CHUNK_OVERLAP)
            if not chunks:
                continue

            vecs = embed.encode(chunks, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
            for ci, (chunk, v) in enumerate(zip(chunks, vecs)):
                pid = stable_point_uuid(doc_sha, page_num, ci)
                payload = {
                    "path": path,
                    "doc_sha": doc_sha,
                    "page": page_num,
                    "chunk": ci,
                    "text": chunk,
                }
                points.append(qm.PointStruct(id=pid, vector=v.tolist(), payload=payload))

                # Upsert in batches
                if len(points) >= args.batch:
                    qc.upsert(collection_name=COLLECTION, points=points)
                    total_points += len(points)
                    points = []

        if points:
            qc.upsert(collection_name=COLLECTION, points=points)
            total_points += len(points)

        manifest[doc_sha] = {
            "path": path,
            "indexed_at": int(time.time()),
            "pages": len(pages),
            "model": EMBED_MODEL,
            "dim": dim,
        }
        save_manifest(manifest)

    dt = time.time() - t0
    print(f"[docs] done. upserted_points={total_points} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
