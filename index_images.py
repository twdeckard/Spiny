#!/usr/bin/env python3
"""
index_images.py
---------------
Indexes images under /mnt/moltbot/corpus/images into Qdrant for text<->image search.

Key features:
- Uses SentenceTransformers CLIP: sentence-transformers/clip-ViT-B-32
- Text->image and image->image similarity search (same embedding space)
- Stores image vectors in Qdrant with COSINE distance
- Uses deterministic UUID point IDs derived from sha256 (Qdrant requires uint or UUID)
- Stores sha256 + metadata as payload for stable identity, dedupe, and citations
- Generates thumbnails to /ssd/moltbot/thumbs
- Handles common photo extensions; optional HEIC/HEIF support if pillow-heif is installed
- Prints a clear summary at the end

Prereqs:
  pip install qdrant-client sentence-transformers pillow tqdm exifread imagehash
Optional (for HEIC/HEIF):
  pip install pillow-heif
"""

import os
import time
import uuid
import hashlib
import mimetypes
from pathlib import Path

import exifread
import imagehash
from PIL import Image
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
CORPUS_DIR = Path("/mnt/moltbot/corpus/images")
THUMBS_DIR = Path("/ssd/moltbot/thumbs")
COLLECTION = "images"

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 120.0  # Jetson-friendly

MODEL_NAME = "sentence-transformers/clip-ViT-B-32"  # English-only baseline

# Extensions to index (add more if you like)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

# Batch size for Qdrant upserts
BATCH = 32


# -----------------------------
# Optional HEIC/HEIF support
# -----------------------------
def _try_enable_heif():
    try:
        from pillow_heif import register_heif_opener  # type: ignore
        register_heif_opener()
        return True
    except Exception:
        return False


HEIF_ENABLED = _try_enable_heif()


# -----------------------------
# Helpers
# -----------------------------
def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_to_uuid(hex_sha256: str) -> str:
    """
    Deterministic UUID derived from sha256 bytes (first 16 bytes).
    Qdrant accepts UUID or uint IDs. This keeps IDs stable across re-index runs.
    """
    b = bytes.fromhex(hex_sha256)
    return str(uuid.UUID(bytes=b[:16]))


def exif_meta(p: Path) -> dict:
    meta = {}
    try:
        with p.open("rb") as f:
            tags = exifread.process_file(f, details=False)

        dto = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if dto:
            s = str(dto)
            try:
                t = time.strptime(s, "%Y:%m:%d %H:%M:%S")
                meta["taken_ts"] = int(time.mktime(t))
            except Exception:
                pass

        make = tags.get("Image Make")
        model = tags.get("Image Model")
        if make:
            meta["camera_make"] = str(make).strip()
        if model:
            meta["camera_model"] = str(model).strip()

    except Exception:
        pass
    return meta


def make_thumb(img: Image.Image, out_path: Path, max_side: int = 512) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = img.copy()
    im.thumbnail((max_side, max_side))
    im.convert("RGB").save(out_path, "JPEG", quality=85, optimize=True)


def infer_embedding_dim(model: SentenceTransformer) -> int:
    dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
    if dim is None:
        test = model.encode("dimension probe", normalize_embeddings=True)
        try:
            dim = int(test.shape[0])  # numpy
        except Exception:
            dim = int(len(test))
    return int(dim)


def list_image_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for r, _, files in os.walk(root):
        for fn in files:
            p = Path(r) / fn
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
    return paths


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit(f"Corpus directory does not exist: {CORPUS_DIR}")

    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"HEIC/HEIF support enabled: {HEIF_ENABLED}")
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Thumbs: {THUMBS_DIR}")
    print(f"Qdrant: http://{QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Model:  {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    dim = infer_embedding_dim(model)
    print(f"Embedding dim: {dim}")

    qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)

    # Create collection if missing
    existing = {c.name for c in qc.get_collections().collections}
    if COLLECTION not in existing:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            on_disk_payload=True,
        )
        # Useful payload indexes for filtering
        qc.create_payload_index(COLLECTION, "taken_ts", qm.PayloadSchemaType.INTEGER)
        qc.create_payload_index(COLLECTION, "camera_model", qm.PayloadSchemaType.KEYWORD)
        qc.create_payload_index(COLLECTION, "path", qm.PayloadSchemaType.KEYWORD)

    paths = list_image_files(CORPUS_DIR)
    if not paths:
        print(f"No images found under: {CORPUS_DIR}")
        return

    seen = 0
    indexed = 0
    skipped = 0
    upserted_points = 0

    batch_points: list[qm.PointStruct] = []

    for p in tqdm(paths, desc="Indexing images"):
        seen += 1
        try:
            st = p.stat()
            sha = sha256_file(p)
            point_id = sha256_to_uuid(sha)

            mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"

            img = Image.open(p)
            w, hgt = img.size

            # Near-duplicate fingerprint
            ph = str(imagehash.phash(img))  # 16 hex chars (~64-bit)

            # Thumbnail
            thumb_path = THUMBS_DIR / f"{sha}.jpg"
            if not thumb_path.exists():
                make_thumb(img, thumb_path)

            payload = {
                "sha256": sha,
                "path": str(p),
                "mtime": int(st.st_mtime),
                "bytes": int(st.st_size),
                "width": int(w),
                "height": int(hgt),
                "mime": mime,
                "phash64": ph,
                "thumb": str(thumb_path),
                "embedding_model": MODEL_NAME,
                "embedding_dim": dim,
            }
            payload.update(exif_meta(p))

            # Embed image
            emb = model.encode([img], normalize_embeddings=True)[0]

            batch_points.append(
                qm.PointStruct(id=point_id, vector=emb.tolist(), payload=payload)
            )
            indexed += 1

            if len(batch_points) >= BATCH:
                qc.upsert(collection_name=COLLECTION, points=batch_points)
                upserted_points += len(batch_points)
                batch_points = []

        except Exception as e:
            skipped += 1
            if skipped <= 25:
                print(f"SKIP {p}: {e}")

    if batch_points:
        qc.upsert(collection_name=COLLECTION, points=batch_points)
        upserted_points += len(batch_points)

    # Final count
    try:
        total = qc.count(collection_name=COLLECTION, exact=True).count
    except Exception:
        total = None

    print(
        f"Done. Seen={seen} Indexed={indexed} Skipped={skipped} "
        f"Upserted={upserted_points} QdrantCount={total}"
    )


if __name__ == "__main__":
    main()
