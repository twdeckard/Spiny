import json
import os
import urllib.request
from typing import List, Dict, Any, Optional

# Provider switch
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

# Ollama (local)
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "smollm:1.7b")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

def _http_post_json(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def chat(system: str, user: str, max_tokens: int = 256) -> str:
    """
    Unified chat call.
    Today: supports ollama.
    Future: can add openai here without changing callers.
    """
    if LLM_PROVIDER == "ollama":
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":user}],
            "options": {"num_predict": max_tokens},
        }
        out = _http_post_json(f"{OLLAMA_BASE}/api/chat", payload, timeout=180)
        return ((out.get("message") or {}).get("content") or "").strip()

    # Placeholder for future:
    if LLM_PROVIDER == "openai":
        raise RuntimeError("LLM_PROVIDER=openai selected, but OpenAI integration is currently disabled on this device.")

    raise RuntimeError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

def select_images(query: str, candidates: List[Dict[str, Any]], k: int = 6) -> List[str]:
    """
    Ask the model to choose a subset from already-retrieved candidates.
    candidates: list of dicts with sha256, score, path.
    Returns list of sha256 strings.
    """
    # Limit prompt size for Jetson
    cand = candidates[:40]
    # Instruction: do NOT invent; only pick from list; output JSON.
    system = (
        "You are a ranking assistant. You MUST NOT claim you saw images.\n"
        "You only have (sha256, score, path).\n"
        "Choose a diverse set that best matches the query.\n"
        "Return ONLY valid JSON like: {\"selected\": [\"sha\", ...]}.\n"
        "Do not include any other keys or text."
    )
    lines = [f"Query: {query}", f"Pick exactly {k} sha256 values from this list:"]
    for i, r in enumerate(cand, 1):
        lines.append(f"{i}. sha256={r.get('sha256')} score={r.get('score')} path={r.get('path')}")
    user = "\n".join(lines)

    raw = chat(system=system, user=user, max_tokens=200)

    # Robust parse + fallback
    try:
        obj = json.loads(raw)
        sel = obj.get("selected", [])
        sel = [s for s in sel if isinstance(s, str)]
        sel = [s for s in sel if any((r.get("sha256") == s) for r in cand)]
        if len(sel) >= 1:
            return sel[:k]
    except Exception:
        pass

    # Fallback: just take top-k by score
    out = []
    for r in candidates:
        sha = r.get("sha256")
        if isinstance(sha, str):
            out.append(sha)
        if len(out) >= k:
            break
    return out
