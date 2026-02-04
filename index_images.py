#!/usr/bin/env python3
"""
index_images.py (incremental)
-----------------------------
Incrementally indexes images under /mnt/moltbot/corpus/images into Qdrant for text<->image search.

Incremental logic:
- Maintains a manifest at /ssd/moltbot/state/index_manifest.json
- For each file path, if (mtime, size) unchanged since last run -> skip hashing/embedding/upsert
- If changed/new -> process, embed, upsert, update manifest

IDs:
- Qdrant point IDs must be UUID or uint. We use a deterministic UUID derived from sha256 bytes.

Payload:
- Includes sha256, path, mtime, bytes, dimensions, mime, taken_ts, camera info, phash64, thumb path.

Optional:
- HEIC/HEIF support if pillow-heif is installed: pip install pillow-heif
"""

import json
import os
import time
import uuid
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
STATE_DIR = Path("/ssd/moltbot/state")
MANIFEST_PATH = STATE_DIR / "index_manifest.json"

COLLECTION = "images"

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 7200.0  # Jetson-friendly

MODEL_NAME = "sentence-transformers/clip-ViT-B-32"  # English-only baseline

# Extensions to index (add more if you like)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

# Batch size for Qdrant upserts
BATCH = 32

# Optional cleanup: if True, remove manifest entries whose file no longer exists.
# NOTE: This does NOT delete from Qdrant by default (see PRUNE_QDRANT_TOO).
PRUNE_MISSING_MANIFEST = True

# If True, also delete missing files from Qdrant (safe, but do it intentionally).
# For nightly runs, I'd keep this False at first.
PRUNE_QDRANT_TOO = True


# -----------------------------
# Optional HEIC/HEIF support
# -----------------------------
def _try_enable_heif() -> bool:
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


def list_image_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for r, _, files in os.walk(root):
        for fn in files:
            p = Path(r) / fn
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p)
    return paths


def load_manifest() -> Dict[str, Dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_manifest(manifest: Dict[str, Dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    tmp.replace(MANIFEST_PATH)


def safe_resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except Exception:
        return p.absolute()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit(f"Corpus directory does not exist: {CORPUS_DIR}")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"HEIC/HEIF support enabled: {HEIF_ENABLED}")
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Thumbs: {THUMBS_DIR}")
    print(f"State:  {STATE_DIR}")
    print(f"Qdrant: http://{QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Model:  {MODEL_NAME}")

    # Load model + determine dim
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
        # Best-effort payload indexes (ignore if unsupported/exists)
        try:
            qc.create_payload_index(COLLECTION, "taken_ts", qm.PayloadSchemaType.INTEGER)
        except Exception:
            pass
        try:
            qc.create_payload_index(COLLECTION, "camera_model", qm.PayloadSchemaType.KEYWORD)
        except Exception:
            pass
        try:
            qc.create_payload_index(COLLECTION, "path", qm.PayloadSchemaType.KEYWORD)
        except Exception:
            pass

    # Load manifest
    manifest = load_manifest()
    manifest_before = len(manifest)

    # Discover files
    paths = list_image_files(CORPUS_DIR)
    if not paths:
        print(f"No images found under: {CORPUS_DIR}")
        return

    # Optional: prune missing manifest entries
    if PRUNE_MISSING_MANIFEST and manifest:
        existing_paths = set(str(p) for p in paths)
        removed = 0
        missing_items: List[Dict[str, Any]] = []
        for mpath in list(manifest.keys()):
            if mpath not in existing_paths:
                removed += 1
                missing_items.append(manifest[mpath])
                del manifest[mpath]
        if removed:
            print(f"Pruned {removed} missing paths from manifest.")
            # Optional: also delete from Qdrant
            if PRUNE_QDRANT_TOO:
                ids = []
                for it in missing_items:
                    pid = it.get("point_id")
                    if pid:
                        ids.append(pid)
                if ids:
                    print(f"Deleting {len(ids)} points from Qdrant (missing files).")
                    qc.delete(
                        collection_name=COLLECTION,
                        points_selector=qm.PointIdsList(points=ids),
                    )

    seen = 0
    unchanged_skipped = 0
    processed = 0
    upserted_points = 0
    failed = 0

    batch_points: List[qm.PointStruct] = []

    for p in tqdm(paths, desc="Indexing images (incremental)"):
        seen += 1
        try:
            st = p.stat()
            path_key = str(p)

            # Fast skip if unchanged
            old = manifest.get(path_key)
            if old and old.get("mtime") == int(st.st_mtime) and old.get("bytes") == int(st.st_size):
                # Ensure thumbnail still exists; if missing, regenerate without re-embedding
                sha_old = old.get("sha256")
                if sha_old:
                    thumb_path = THUMBS_DIR / f"{sha_old}.jpg"
                    if not thumb_path.exists():
                        try:
                            img = Image.open(p)
                            make_thumb(img, thumb_path)
                        except Exception:
                            pass
                unchanged_skipped += 1
                continue

            # Process new/changed file
            sha = sha256_file(p)
            point_id = sha256_to_uuid(sha)

            mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"

            img = Image.open(p)
            w, hgt = img.size

            ph = str(imagehash.phash(img))  # 16 hex chars (~64-bit)

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

            emb = model.encode([img], normalize_embeddings=True)[0]

            batch_points.append(qm.PointStruct(id=point_id, vector=emb.tolist(), payload=payload))
            processed += 1

            # Update manifest entry only after we successfully queue the point
            manifest[path_key] = {
                "mtime": int(st.st_mtime),
                "bytes": int(st.st_size),
                "sha256": sha,
                "point_id": point_id,
                "thumb": str(thumb_path),
            }

            if len(batch_points) >= BATCH:
                qc.upsert(collection_name=COLLECTION, points=batch_points)
                upserted_points += len(batch_points)
                batch_points = []

        except Exception as e:
            failed += 1
            if failed <= 25:
                print(f"SKIP {p}: {e}")

    if batch_points:
        qc.upsert(collection_name=COLLECTION, points=batch_points)
        upserted_points += len(batch_points)

    # Save manifest at end (atomic)
    save_manifest(manifest)

    # Final count
    try:
        total = qc.count(collection_name=COLLECTION, exact=True).count
    except Exception:
        total = None

    print(
        "Done. "
        f"Seen={seen} "
        f"SkippedUnchanged={unchanged_skipped} "
        f"Processed={processed} "
        f"Upserted={upserted_points} "
        f"Failed={failed} "
        f"ManifestBefore={manifest_before} ManifestAfter={len(manifest)} "
        f"QdrantCount={total}"
    )


if __name__ == "__main__":
    main()
