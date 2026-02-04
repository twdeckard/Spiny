#!/usr/bin/env python3
"""
viewer.py
---------
Minimal local web UI for text->image search over Qdrant using CLIP embeddings.

- Query: text -> CLIP embedding -> Qdrant vector search
- Results: thumbnail grid with score + click-through to original image
- Safe serving: originals must live under CORPUS_DIR
- Works across qdrant-client API variations (search / search_points / REST fallback)
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# -------------------------
# Config
# -------------------------
COLLECTION = "images"
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
QDRANT_TIMEOUT = 60.0

CORPUS_DIR = Path("/mnt/moltbot/corpus/images").resolve()
THUMBS_DIR = Path("/ssd/moltbot/thumbs").resolve()

DEFAULT_K = 24


# -------------------------
# Qdrant compatibility helpers
# -------------------------
def _rest_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=QDRANT_TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _qdrant_search(qc: QdrantClient, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
    """
    Returns list of dict: {id, score, payload}
    """
    hits = None
    if hasattr(qc, "search"):
        hits = qc.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        # hits is list of ScoredPoint-like objects
        out = []
        for h in hits:
            out.append({"id": getattr(h, "id", None), "score": getattr(h, "score", None), "payload": getattr(h, "payload", {})})
        return out

    if hasattr(qc, "search_points"):
        hits = qc.search_points(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        out = []
        for h in hits:
            out.append({"id": getattr(h, "id", None), "score": getattr(h, "score", None), "payload": getattr(h, "payload", {})})
        return out

    # REST fallback
    data = _rest_post(f"/collections/{COLLECTION}/points/search", {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
    })
    res = data.get("result", [])
    out = []
    for h in res:
        out.append({"id": h.get("id"), "score": h.get("score"), "payload": h.get("payload") or {}})
    return out


def _qdrant_find_by_sha256(qc: QdrantClient, sha256: str) -> Optional[Dict[str, Any]]:
    """
    Find a single point by payload.sha256 using scroll (or REST fallback).
    Returns payload dict (or None).
    """
    # Prefer python client scroll if available
    if hasattr(qc, "scroll"):
        points, _ = qc.scroll(
            collection_name=COLLECTION,
            scroll_filter={
                "must": [
                    {"key": "sha256", "match": {"value": sha256}}
                ]
            },
            with_payload=True,
            with_vectors=False,
            limit=1,
        )
        if points:
            p = points[0]
            return getattr(p, "payload", None) or None

    # REST fallback
    data = _rest_post(f"/collections/{COLLECTION}/points/scroll", {
        "filter": {
            "must": [
                {"key": "sha256", "match": {"value": sha256}}
            ]
        },
        "limit": 1,
        "with_payload": True,
        "with_vectors": False
    })
    res = data.get("result", {}).get("points", [])
    if res:
        return res[0].get("payload") or None
    return None


# -------------------------
# Model + client singletons
# -------------------------
app = FastAPI(title="Moltbot Image Search Viewer")

model = SentenceTransformer(MODEL_NAME)
qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)


# -------------------------
# HTML helpers
# -------------------------
def _html_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif; margin: 18px; }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    input[type=text] {{ width: min(700px, 95vw); padding: 10px; font-size: 16px; }}
    input[type=number] {{ width: 90px; padding: 10px; font-size: 16px; }}
    button {{ padding: 10px 14px; font-size: 16px; cursor: pointer; }}
    .muted {{ color: #666; font-size: 13px; }}
    .grid {{ margin-top: 16px; display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; overflow: hidden; background: #fff; }}
    .thumb {{ width: 100%; height: 170px; object-fit: cover; display: block; background: #f6f6f6; }}
    .meta {{ padding: 10px; font-size: 12px; }}
    .path {{ word-break: break-all; color: #333; }}
    .score {{ color: #444; margin-top: 6px; }}
    a {{ color: inherit; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .warn {{ background: #fff7e6; border: 1px solid #ffd699; padding: 10px; border-radius: 10px; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def _render_form(q: str, k: int) -> str:
    q_esc = urllib.parse.quote(q, safe="")
    return f"""
  <h2>Moltbot image search</h2>
  <form method="get" action="/" class="row">
    <input type="text" name="q" value="{q_esc}" placeholder="e.g., dog on the beach" />
    <input type="number" name="k" value="{k}" min="1" max="100" />
    <button type="submit">Search</button>
  </form>
  <div class="muted">Corpus: {CORPUS_DIR} • Thumbs: {THUMBS_DIR} • Collection: {COLLECTION}</div>
"""


def _render_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return '<div class="warn">No results. Try a different query.</div>'

    tiles = []
    for r in results:
        p = r.get("payload") or {}
        sha = p.get("sha256", "")
        score = r.get("score", 0.0)
        thumb_url = f"/thumb/{sha}"
        open_url = f"/img/{sha}"

        short_path = p.get("path", "")
        tiles.append(f"""
        <div class="card">
          <a href="{open_url}" target="_blank" rel="noopener">
            <img class="thumb" src="{thumb_url}" loading="lazy" />
          </a>
          <div class="meta">
            <div class="path">{short_path}</div>
            <div class="score">score: {score:.4f}</div>
          </div>
        </div>
        """)

    return f'<div class="grid">{"".join(tiles)}</div>'


# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(q: str = "", k: int = DEFAULT_K):
    q = q.strip()
    if not q:
        body = _render_form("", DEFAULT_K) + "<p class='muted'>Enter a text query to search your indexed images.</p>"
        return _html_page("Moltbot Image Search", body)

    # Embed query text
    qvec = model.encode(q, normalize_embeddings=True).tolist()

    # Search Qdrant
    results = _qdrant_search(qc, qvec, limit=max(1, min(int(k), 100)))

    body = _render_form(q, k) + _render_results(results)
    return _html_page("Moltbot Image Search", body)


@app.get("/thumb/{sha256}", response_class=FileResponse)
def thumb(sha256: str):
    # thumbs are stored as /ssd/moltbot/thumbs/<sha256>.jpg
    p = (THUMBS_DIR / f"{sha256}.jpg").resolve()
    if not str(p).startswith(str(THUMBS_DIR)):
        raise HTTPException(status_code=400, detail="Invalid thumb path")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.get("/img/{sha256}", response_class=FileResponse)
def original_image(sha256: str):
    # Look up payload by sha256, then serve the original path if it is under CORPUS_DIR.
    payload = _qdrant_find_by_sha256(qc, sha256)
    if not payload:
        raise HTTPException(status_code=404, detail="Image not found in index")

    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=404, detail="No path in payload")

    p = Path(path).resolve()

    # Safety: only serve files inside corpus dir
    if not str(p).startswith(str(CORPUS_DIR)):
        raise HTTPException(status_code=403, detail="File outside corpus scope")
    if not p.exists():
        raise HTTPException(status_code=404, detail="Original file missing")

    # Let the browser infer image type if possible
    return FileResponse(str(p))

# --- Spiny search grid (added) ---
import os, json, urllib.parse, urllib.request
from fastapi import Query
from fastapi.responses import HTMLResponse

SPINY_API_BASE = os.environ.get("SPINY_API_BASE", "http://127.0.0.1:8091")

def _spiny_api_search(q: str, k: int):
    url = f"{SPINY_API_BASE}/search_images?q={urllib.parse.quote(q)}&k={k}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))

@app.get("/search", response_class=HTMLResponse)
def search_grid(q: str = Query(..., min_length=1), k: int = Query(24, ge=1, le=200)):
    payload = _spiny_api_search(q, k)
    results = payload.get("results", [])

    tiles = []
    for r in results:
        sha = r.get("sha256")
        if not sha:
            continue
        score = r.get("score", 0.0)
        path = (r.get("path") or "")
        tiles.append(
            f"<div style='width:180px;margin:8px;'>"
            f"<a href='/img/{sha}'><img src='/thumb/{sha}' style='width:180px;border-radius:10px;'></a>"
            f"<div style='font-size:12px;margin-top:6px;'>score={float(score):.4f}</div>"
            f"<div style='font-size:11px;opacity:0.75;word-break:break-all;'>{path}</div>"
            f"</div>"
        )

    html = (
        "<html><body style='font-family:system-ui;'>"
        f"<h3>Search: “{q}”</h3>"
        "<div style='margin:10px 0;'><a href='/'>Home</a></div>"
        "<div style='display:flex;flex-wrap:wrap;'>"
        + "".join(tiles)
        + "</div></body></html>"
    )
    return HTMLResponse(html)
