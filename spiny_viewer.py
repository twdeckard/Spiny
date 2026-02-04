#!/usr/bin/env python3
import os
import urllib.parse
import urllib.request
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse

API_BASE = os.environ.get("SPINY_API_BASE", "http://127.0.0.1:8091")
PORT = int(os.environ.get("SPINY_VIEWER_PORT", "8090"))
K_DEFAULT = int(os.environ.get("SPINY_VIEWER_K", "24"))

app = FastAPI()

def api_search(q: str, k: int):
    url = f"{API_BASE}/search_images?q={urllib.parse.quote(q)}&k={k}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        import json
        return json.loads(resp.read().decode("utf-8"))

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        "<html><body style='font-family:system-ui;'>"
        "<h2>Spiny Viewer</h2>"
        "<form method='get' action='/search'>"
        "<input name='q' placeholder='search images (english)' style='width:360px;padding:8px;'> "
        "<input type='number' name='k' value='24' min='1' max='200' style='width:90px;padding:8px;'> "
        "<button style='padding:8px 12px;'>Search</button>"
        "</form>"
        "<p>Tip: try <code>airplane</code>, <code>snow at sunset</code></p>"
        "</body></html>"
    )

@app.get("/search", response_class=HTMLResponse)
def search(q: str = Query(..., min_length=1), k: int = Query(K_DEFAULT, ge=1, le=200)):
    payload = api_search(q, k)
    results = payload.get("results", [])
    # results items must have sha256 and score; path optional
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
        "<div style='margin:10px 0;'><a href='/'>New search</a></div>"
        "<div style='display:flex;flex-wrap:wrap;'>"
        + "".join(tiles)
        + "</div></body></html>"
    )
    return HTMLResponse(html)

# Optional: keep backward-compat if someone hits /s?q=...
@app.get("/s")
def s(q: str, k: int = K_DEFAULT):
    return RedirectResponse(url=f"/search?q={urllib.parse.quote(q)}&k={k}")
