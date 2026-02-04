#!/usr/bin/env python3
"""
Spiny Signal Router

- Receives Signal messages via signal-cli (supports both dataMessage and syncMessage sentMessage)
- Routes "photos/images" and natural language photo queries to Spiny API (/search_images)
- Adds "docs" command to query indexed PDFs via Spiny API (/search_docs)
- Optionally uses a local LLM (via llm_provider.py) to curate a small subset from the top-N image results
- Supports session state for photos: more/refine/exclude/only/reset
- Supports thumbnail attachments via Signal: "thumbs [n]"
- Adds a viewer grid link: http://LAN:8090/search?q=...&k=24

Robustness:
- API calls never raise; errors return {"results": [], "error": "..."} and router replies cleanly.
"""

import json
import os
import re
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

# Local LLM abstraction (must be importable; set PYTHONPATH=/ssd/moltbot/app in systemd)
import llm_provider

SIGNAL_CLI = os.environ.get("SIGNAL_CLI", "signal-cli")
SIGNAL_CONFIG = os.environ.get("SIGNAL_CONFIG", "/ssd/moltbot/secrets/signal-cli")

API_BASE = os.environ.get("SPINY_API_BASE", "http://127.0.0.1:8091")

# Viewer
VIEWER_PORT = int(os.environ.get("SPINY_VIEWER_PORT", "8090"))
VIEWER_BASE_ENV = os.environ.get("SPINY_VIEWER_BASE", "").strip()

# Output + behavior
RESULTS_K = int(os.environ.get("SPINY_K", "6"))  # How many image results to show in a message
VIEWER_GRID_K = int(os.environ.get("SPINY_VIEWER_K", "24"))  # How many the web grid should show
MAX_MESSAGE_RESULTS = int(os.environ.get("SPINY_MAX_MESSAGE_RESULTS", "8"))  # hard cap per message
MAX_ATTACH = int(os.environ.get("SPINY_MAX_ATTACH", "3"))  # max thumbs to attach in one command

# Docs output
DOCS_K = int(os.environ.get("SPINY_DOCS_K", "5"))  # How many docs results to show in a message
DOCS_MAX_MESSAGE_RESULTS = int(os.environ.get("SPINY_DOCS_MAX_MESSAGE_RESULTS", "8"))  # hard cap per docs message

# LLM curation (images only)
LLM_CURATE_ENABLED = os.environ.get("SPINY_LLM_CURATE", "1").strip() not in ("0", "false", "False")
LLM_CURATE_K = int(os.environ.get("SPINY_LLM_CURATE_K", str(RESULTS_K)))  # number of curated picks

# Session state
STATE_PATH = os.environ.get("SPINY_STATE_PATH", "/ssd/moltbot/state/spiny_sessions.json")

REPLY_PREFIX = "📷 Spiny results for:"
DOCS_REPLY_PREFIX = "📄 Spiny docs for:"
NL_HINT = "Tip: try 'photos dog', 'docs ford thunderbird fuel pump', 'more', 'refine sunset', 'thumbs 3', 'brain help'"
CMD_HELP_TEXT = (
    "Commands:\n"
    "- photos <query>           (image search)\n"
    "- photos <query> raw       (skip LLM curation)\n"
    "- show me pictures of <x>  (natural language image search)\n"
    "- docs <query>             (PDF search)\n"
    "- more [n]                 (show more image results)\n"
    "- refine <text>            (append to current image query)\n"
    "- exclude <text>           (soft exclude from image query)\n"
    "- only <text>              (replace image query)\n"
    "- thumbs [n]               (send thumb attachments for last image results)\n"
    "- brain <question>         (local assistant)\n"
    "- reset                    (clear session)\n"
)

# --- Natural language triggers (images) ---
CMD_EXPLICIT = re.compile(r"^\s*(photos|images)\s*[: ]\s*(.+)$", re.IGNORECASE)
CMD_NL = re.compile(
    r"^\s*(show|find|search|look)\s+.*\b(photos|pictures|images|pics)\b\s*(of|for)?\s*(.+)$",
    re.IGNORECASE,
)
CMD_DO_YOU_HAVE = re.compile(r"^\s*do you have (any )?(photos|pictures|images)\s*(of)?\s*(.+)$", re.IGNORECASE)

# --- Commands ---
CMD_MORE = re.compile(r"^\s*more(\s+(\d+))?\s*$", re.IGNORECASE)
CMD_REFINE = re.compile(r"^\s*refine\s+(.+)$", re.IGNORECASE)
CMD_ONLY = re.compile(r"^\s*only\s+(.+)$", re.IGNORECASE)
CMD_EXCLUDE = re.compile(r"^\s*exclude\s+(.+)$", re.IGNORECASE)
CMD_RESET = re.compile(r"^\s*reset\s*$", re.IGNORECASE)
CMD_THUMBS = re.compile(r"^\s*thumbs(\s+(\d+))?\s*$", re.IGNORECASE)
CMD_BRAIN = re.compile(r"^\s*brain\s+(.+)$", re.IGNORECASE)
CMD_HELP = re.compile(r"^\s*(help|\?)\s*$", re.IGNORECASE)

# --- Docs command ---
CMD_DOCS = re.compile(r"^\s*docs\s*[: ]\s*(.+)$", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[spiny-signal] {msg}", flush=True)


def ensure_state_dir() -> None:
    d = os.path.dirname(STATE_PATH)
    os.makedirs(d, exist_ok=True)


def load_state() -> Dict[str, Any]:
    ensure_state_dir()
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    ensure_state_dir()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def api_search_images(query: str, k: int) -> Dict[str, Any]:
    """
    Robust API call: never raises.
    If unreachable, returns {"results": [], "error": "..."}.
    """
    url = f"{API_BASE}/search_images?q={urllib.parse.quote(query)}&k={k}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log(f"api_search_images failed url={url!r} err={e!r}")
        return {"results": [], "error": str(e)}


def api_search_docs(query: str, k: int) -> Dict[str, Any]:
    """
    Robust API call: never raises.
    If unreachable, returns {"results": [], "error": "..."}.
    """
    url = f"{API_BASE}/search_docs?q={urllib.parse.quote(query)}&k={k}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log(f"api_search_docs failed url={url!r} err={e!r}")
        return {"results": [], "error": str(e)}


def send_signal(recipient: str, text: str, attachment: Optional[str] = None) -> bool:
    cmd = [SIGNAL_CLI, "--config", SIGNAL_CONFIG, "send", "-m", text, recipient]
    if attachment:
        cmd += ["-a", attachment]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log(f"send failed rc={p.returncode} stdout={p.stdout!r} stderr={p.stderr!r}")
        return False
    return True


def reply_api_unreachable(reply_to: str, op: str, err: str) -> None:
    msg = (
        f"Spiny API unreachable.\n"
        f"Operation: {op}\n"
        f"API_BASE: {API_BASE}\n"
        f"Error: {err}\n\n"
        f"Try on Jetson:\n"
        f"- systemctl status spiny-api --no-pager\n"
        f"- curl -s http://127.0.0.1:8091/health\n"
    )
    send_signal(reply_to, msg)


def detect_lan_ip() -> str:
    """
    Best-effort: discover outward-facing IP (often your LAN interface IP).
    If it fails, fall back to SPINY_LAN_IP or a reasonable default.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return os.environ.get("SPINY_LAN_IP", "192.168.0.117")


def viewer_base() -> str:
    if VIEWER_BASE_ENV:
        return VIEWER_BASE_ENV.rstrip("/")
    return f"http://{detect_lan_ip()}:{VIEWER_PORT}"


def viewer_search_url(query: str, k: int) -> str:
    return f"{viewer_base()}/search?q={urllib.parse.quote(query)}&k={k}"


def extract_text_and_reply_to(evt: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (message_text, reply_to_number, convo_key).
    - convo_key groups messages for state:
      - groupId if present
      - else a phone number
    Handles:
      - envelope.dataMessage.message (incoming messages)
      - envelope.syncMessage.sentMessage.message (your own sent messages, note-to-self, linked devices)
    """
    env = evt.get("envelope") or {}

    # 1) Normal inbound message
    dm = env.get("dataMessage")
    if isinstance(dm, dict):
        msg = (dm.get("message") or "").strip()
        group = dm.get("groupInfo") or {}
        group_id = group.get("groupId") if isinstance(group, dict) else None
        if msg:
            src = env.get("source")
            convo_key = group_id or src
            return msg, src, convo_key

    # 2) Sync of your own sent messages
    sm = env.get("syncMessage")
    if isinstance(sm, dict):
        sent = sm.get("sentMessage")
        if isinstance(sent, dict):
            msg = (sent.get("message") or "").strip()
            if msg:
                dest = (
                    sent.get("destination")
                    or sent.get("destinationNumber")
                    or sent.get("destinationAddress")
                    or env.get("source")
                )
                group = sent.get("groupInfo") or {}
                group_id = group.get("groupId") if isinstance(group, dict) else None
                convo_key = group_id or dest
                return msg, dest, convo_key

    return None, None, None


def parse_query(msg: str) -> Optional[Tuple[str, bool]]:
    """
    Returns (query, raw_mode) where raw_mode means "skip LLM curation".
    Recognizes:
      - "photos dog"
      - "photos dog raw"
      - natural language "show me pictures of dog"
    """
    raw_mode = False
    m = CMD_EXPLICIT.match(msg)
    if m:
        q = m.group(2).strip()
        if q.lower().endswith(" raw"):
            raw_mode = True
            q = q[:-4].strip()
        return q, raw_mode

    m = CMD_NL.match(msg)
    if m:
        q = m.group(4).strip()
        return q, raw_mode

    m = CMD_DO_YOU_HAVE.match(msg)
    if m:
        q = m.group(4).strip()
        return q, raw_mode

    return None


def format_results(query: str, results: List[dict], start: int = 0, curated: bool = False) -> Tuple[str, List[str]]:
    """
    Image results formatter.
    Returns (message_text, shown_shas)
    """
    if not results:
        return (f'No results for: "{query}"\n\n{NL_HINT}', [])

    vb = viewer_base()
    header = f'{REPLY_PREFIX} "{query}"'
    if curated:
        header += " (curated)"

    lines = [
        header,
        f"Open grid: {viewer_search_url(query, k=VIEWER_GRID_K)}",
        ""
    ]

    count = min(MAX_MESSAGE_RESULTS, RESULTS_K)
    shown = results[start:start + count]

    shown_shas: List[str] = []
    for i, r in enumerate(shown, 1 + start):
        sha = r.get("sha256")
        if not sha:
            continue
        shown_shas.append(sha)
        score = r.get("score", 0.0)
        path = (r.get("path") or "")

        lines.append(f"{i}. score={float(score):.4f}")
        lines.append(f"   open:  {vb}/img/{sha}")
        lines.append(f"   thumb: {vb}/thumb/{sha}")
        if path:
            lines.append(f"   path:  {path}")
        lines.append("")

    lines.append("Commands: more | refine <text> | exclude <text> | only <text> | thumbs [n] | brain <q> | reset")
    return ("\n".join(lines).strip(), shown_shas)


def format_docs_results(query: str, results: List[dict]) -> str:
    """
    Docs results formatter.
    Results are expected like:
      {score, path, page, chunk, snippet}
    """
    if not results:
        return f'{DOCS_REPLY_PREFIX} "{query}"\nNo results.\n\n{NL_HINT}'

    lines: List[str] = [
        f'{DOCS_REPLY_PREFIX} "{query}"',
        ""
    ]

    count = min(DOCS_MAX_MESSAGE_RESULTS, DOCS_K)
    shown = results[:count]

    for i, r in enumerate(shown, 1):
        score = r.get("score", 0.0)
        path = (r.get("path") or "")
        page = r.get("page")
        snippet = (r.get("snippet") or "").strip()
        if len(snippet) > 360:
            snippet = snippet[:360].rstrip() + "…"

        # Short doc name
        doc_name = os.path.basename(path) if path else "(unknown)"
        page_str = f"p.{int(page)}" if isinstance(page, (int, float)) else "p.?"
        lines.append(f"{i}. {doc_name} {page_str}  score={float(score):.4f}")
        if snippet:
            lines.append(f"   {snippet}")
        if path:
            lines.append(f"   path: {path}")
        lines.append("")

    lines.append("Tip: docs <query>  (e.g., docs fuel pump priming)")
    return "\n".join(lines).strip()


def get_thumb_path(sha: str) -> Optional[str]:
    """
    Adjust these candidates if your thumbnail directory differs.
    """
    candidates = [
        f"/ssd/moltbot/thumbs/{sha}.jpg",
        f"/ssd/moltbot/corpus/thumbs/{sha}.jpg",
        f"/ssd/moltbot/corpus/thumbs/{sha}.jpeg",
        f"/ssd/moltbot/thumbs/{sha}.jpeg",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def curate_subset(query: str, results_full: List[dict], k: int) -> List[dict]:
    """
    Uses llm_provider.select_images() to choose subset by sha256, then returns
    the subset in the order returned by the model (if possible), otherwise score order.
    Fails safe to top-k by score.
    """
    if not LLM_CURATE_ENABLED:
        return results_full[:k]

    try:
        selected_shas = llm_provider.select_images(query, results_full, k=k)
        if not selected_shas:
            return results_full[:k]

        by_sha = {r.get("sha256"): r for r in results_full if r.get("sha256")}
        subset = [by_sha[s] for s in selected_shas if s in by_sha]

        if len(subset) < k:
            seen = set(selected_shas)
            for r in results_full:
                sha = r.get("sha256")
                if isinstance(sha, str) and sha not in seen:
                    subset.append(r)
                    if len(subset) >= k:
                        break
        return subset[:k]
    except Exception as e:
        log(f"LLM curation failed; falling back to score-ranked. err={e!r}")
        return results_full[:k]


def brain_reply(user_text: str) -> str:
    """
    Small local assistant. Must stay grounded (no image hallucination).
    """
    system = (
        "You are Spiny, a small local assistant running on a Jetson.\n"
        "Be concise, helpful, and practical.\n"
        "If asked about images, you do NOT see images; you only know filenames/paths/scores when provided.\n"
        "When asked for commands, show brief usage.\n"
    )
    if user_text.strip().lower() in ("help", "commands"):
        return CMD_HELP_TEXT
    try:
        out = llm_provider.chat(system=system, user=user_text, max_tokens=220)
        return out[:3500] if out else "No response."
    except Exception as e:
        return f"Local brain unavailable: {e}"


def main() -> None:
    log("starting receive loop")
    state = load_state()

    while True:
        p = subprocess.run(
            [SIGNAL_CLI, "--config", SIGNAL_CONFIG, "-o", "json", "receive", "--timeout", "10"],
            capture_output=True,
            text=True,
        )

        if p.returncode != 0:
            log(f"receive failed rc={p.returncode} stderr={p.stderr!r}")
            time.sleep(1)
            continue

        out = (p.stdout or "").strip()
        if not out:
            continue

        for line in out.splitlines():
            try:
                evt = json.loads(line)
            except Exception:
                continue

            msg, reply_to, convo_key = extract_text_and_reply_to(evt)
            if not msg or not reply_to or not convo_key:
                continue

            msg_stripped = msg.strip()

            # Avoid loops if our own long reply is mirrored back via syncMessage
            if msg_stripped.startswith(REPLY_PREFIX) or msg_stripped.startswith(DOCS_REPLY_PREFIX):
                continue

            log(f"rx convo={convo_key!r} from={reply_to!r} msg={msg_stripped!r}")
            sess = state.get(convo_key) or {}

            # HELP
            if CMD_HELP.match(msg_stripped):
                send_signal(reply_to, CMD_HELP_TEXT)
                continue

            # RESET
            if CMD_RESET.match(msg_stripped):
                state[convo_key] = {}
                save_state(state)
                send_signal(reply_to, "Session reset. " + NL_HINT)
                continue

            # DOCS (PDF search)
            md = CMD_DOCS.match(msg_stripped)
            if md:
                qd = md.group(1).strip()
                if not qd:
                    send_signal(reply_to, "Usage: docs <query>")
                    continue

                payload = api_search_docs(qd, k=DOCS_K)
                if payload.get("error"):
                    reply_api_unreachable(reply_to, op="docs", err=str(payload.get("error")))
                    continue

                results = payload.get("results", []) or []
                txt = format_docs_results(qd, results)
                send_signal(reply_to, txt)
                continue

            # BRAIN (local assistant)
            mb = CMD_BRAIN.match(msg_stripped)
            if mb:
                q = mb.group(1).strip()
                send_signal(reply_to, brain_reply(q))
                continue

            # THUMBS
            mt = CMD_THUMBS.match(msg_stripped)
            if mt:
                n = int(mt.group(2) or "1")
                n = max(1, min(MAX_ATTACH, n))

                shown = sess.get("last_shown_shas") or []
                if not shown:
                    send_signal(reply_to, "No recent results to attach. Search first (e.g., 'photos dog').")
                    continue

                sent_any = False
                for sha in shown[:n]:
                    tp = get_thumb_path(sha)
                    if not tp:
                        continue
                    caption = f"thumb {sha[:10]}…"
                    sent_any = send_signal(reply_to, caption, attachment=tp) or sent_any

                if not sent_any:
                    send_signal(reply_to, "Couldn't find thumbnail files on disk for the current results.")
                continue

            # MORE (images)
            mm = CMD_MORE.match(msg_stripped)
            if mm:
                inc = int(mm.group(2) or str(RESULTS_K))
                inc = max(1, min(50, inc))

                results_full = sess.get("results_full") or []
                if not results_full:
                    send_signal(reply_to, "No active results. Search first (e.g., 'photos dog').")
                    continue

                offset = int(sess.get("offset", 0))
                start = offset
                sess["offset"] = offset + inc

                query = sess.get("query", "")
                curated = bool(sess.get("curated", False))
                txt, shown_shas = format_results(query, results_full, start=start, curated=curated)

                sess["last_shown_shas"] = shown_shas
                state[convo_key] = sess
                save_state(state)
                send_signal(reply_to, txt)
                continue

            # REFINE / ONLY / EXCLUDE (images)
            mr = CMD_REFINE.match(msg_stripped)
            mo = CMD_ONLY.match(msg_stripped)
            me = CMD_EXCLUDE.match(msg_stripped)
            if mr or mo or me:
                base_q = sess.get("query") or ""
                if not base_q and not mo:
                    send_signal(reply_to, "No active query to refine. Search first (e.g., 'photos dog').")
                    continue

                if mr:
                    new_q = f"{base_q}, {mr.group(1).strip()}"
                    op = "refine"
                elif mo:
                    new_q = mo.group(1).strip()
                    op = "only"
                else:
                    new_q = f"{base_q}, without {me.group(1).strip()}"
                    op = "exclude"

                payload = api_search_images(new_q, k=200)
                if payload.get("error"):
                    reply_api_unreachable(reply_to, op=op, err=str(payload.get("error")))
                    continue

                results_full = payload.get("results", []) or []

                curated = False
                if results_full and LLM_CURATE_ENABLED:
                    curated_subset = curate_subset(new_q, results_full, k=LLM_CURATE_K)
                    curated_shas = {r.get("sha256") for r in curated_subset if r.get("sha256")}
                    remainder = [r for r in results_full if r.get("sha256") not in curated_shas]
                    results_full = curated_subset + remainder
                    curated = True

                txt, shown_shas = format_results(new_q, results_full, start=0, curated=curated)
                sess = {
                    "query": new_q,
                    "results_full": results_full,
                    "offset": RESULTS_K,
                    "curated": curated,
                    "last_shown_shas": shown_shas,
                }
                state[convo_key] = sess
                save_state(state)
                send_signal(reply_to, txt)
                continue

            # NEW IMAGE SEARCH QUERY
            pq = parse_query(msg_stripped)
            if pq:
                q, raw_mode = pq
                payload = api_search_images(q, k=200)
                if payload.get("error"):
                    reply_api_unreachable(reply_to, op="search", err=str(payload.get("error")))
                    continue

                results_full = payload.get("results", []) or []

                curated = False
                if results_full and (LLM_CURATE_ENABLED and not raw_mode):
                    curated_subset = curate_subset(q, results_full, k=LLM_CURATE_K)
                    curated_shas = {r.get("sha256") for r in curated_subset if r.get("sha256")}
                    remainder = [r for r in results_full if r.get("sha256") not in curated_shas]
                    results_full = curated_subset + remainder
                    curated = True

                txt, shown_shas = format_results(q, results_full, start=0, curated=curated)
                sess = {
                    "query": q,
                    "results_full": results_full,
                    "offset": RESULTS_K,
                    "curated": curated,
                    "last_shown_shas": shown_shas,
                }
                state[convo_key] = sess
                save_state(state)
                send_signal(reply_to, txt)
                continue

            # IMPLICIT IMAGE SEARCH (simple messages like "dog" or "airplane")
            if len(msg_stripped.split()) <= 8 and any(ch.isalpha() for ch in msg_stripped):
                q = msg_stripped
                payload = api_search_images(q, k=200)
                if payload.get("error"):
                    reply_api_unreachable(reply_to, op="search", err=str(payload.get("error")))
                    continue

                results_full = payload.get("results", []) or []

                curated = False
                if results_full and LLM_CURATE_ENABLED:
                    curated_subset = curate_subset(q, results_full, k=LLM_CURATE_K)
                    curated_shas = {r.get("sha256") for r in curated_subset if r.get("sha256")}
                    remainder = [r for r in results_full if r.get("sha256") not in curated_shas]
                    results_full = curated_subset + remainder
                    curated = True

                txt, shown_shas = format_results(q, results_full, start=0, curated=curated)
                sess = {
                    "query": q,
                    "results_full": results_full,
                    "offset": RESULTS_K,
                    "curated": curated,
                    "last_shown_shas": shown_shas,
                }
                state[convo_key] = sess
                save_state(state)
                send_signal(reply_to, txt)
                continue

            # FALLTHROUGH
            send_signal(reply_to, "I didn't understand. " + NL_HINT)


if __name__ == "__main__":
    main()
