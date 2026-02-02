#!/usr/bin/env python3
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple

SIGNAL_CLI = os.environ.get("SIGNAL_CLI", "signal-cli")
SIGNAL_CONFIG = os.environ.get("SIGNAL_CONFIG", "/ssd/moltbot/secrets/signal-cli")

API_BASE = os.environ.get("SPINY_API_BASE", "http://127.0.0.1:8091")
VIEWER_BASE = os.environ.get("SPINY_VIEWER_BASE", "http://192.168.0.117:8090")
RESULTS_K = int(os.environ.get("SPINY_K", "6"))

CMD_RE = re.compile(r"^\s*(photos|images)\s*[: ]\s*(.+)$", re.IGNORECASE)

# Prevent reply loops (e.g., in Note-to-self). If our own reply comes back via syncMessage, ignore it.
REPLY_PREFIX = "📷 Spiny results for:"

def log(msg: str):
    print(f"[spiny-signal] {msg}", flush=True)

def api_search(query: str, k: int):
    url = f"{API_BASE}/search_images?q={urllib.parse.quote(query)}&k={k}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))

def send_signal(recipient: str, text: str):
    cmd = [SIGNAL_CLI, "--config", SIGNAL_CONFIG, "send", "-m", text, recipient]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log(f"send failed rc={p.returncode} stdout={p.stdout!r} stderr={p.stderr!r}")
        return False
    log(f"sent reply to {recipient}")
    return True

def format_results(query: str, results):
    if not results:
        return f'No results for: "{query}"'
    lines = [f'{REPLY_PREFIX} "{query}"', ""]
    for i, r in enumerate(results[:RESULTS_K], 1):
        sha = r.get("sha256")
        path = r.get("path", "")
        score = r.get("score", 0.0)
        lines.append(f"{i}. score={float(score):.4f}")
        if sha:
            lines.append(f"   open:  {VIEWER_BASE}/img/{sha}")
            lines.append(f"   thumb: {VIEWER_BASE}/thumb/{sha}")
        if path:
            lines.append(f"   path:  {path}")
        lines.append("")
    return "\n".join(lines).strip()

def extract_text_and_reply_to(evt: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (message_text, reply_to_number).
    Handles both:
      - envelope.dataMessage.message (incoming)
      - envelope.syncMessage.sentMessage.message (your own sent messages)
    """
    env = evt.get("envelope") or {}

    # 1) Normal incoming message
    dm = env.get("dataMessage")
    if isinstance(dm, dict):
        msg = (dm.get("message") or "").strip()
        if msg:
            return msg, env.get("source")

    # 2) Sync of your own sent messages (common in Note-to-self tests)
    sm = env.get("syncMessage")
    if isinstance(sm, dict):
        sent = sm.get("sentMessage")
        if isinstance(sent, dict):
            msg = (sent.get("message") or "").strip()
            if msg:
                # destination is who you sent it to (often your own number in Note-to-self)
                dest = (
                    sent.get("destination")
                    or sent.get("destinationNumber")
                    or sent.get("destinationAddress")
                    or env.get("source")
                )
                return msg, dest

    return None, None

def main():
    log("starting receive loop")
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

            msg, reply_to = extract_text_and_reply_to(evt)

            # Log envelope type when message isn't present (helps diagnose receipts/typing/etc)
            if not msg:
                env = evt.get("envelope") or {}
                log(f"rx non-text envelope keys={list(env.keys())}")
                continue

            if msg.startswith(REPLY_PREFIX):
                # avoid loops if our own reply comes back via sync
                continue

            log(f"rx text to handle reply_to={reply_to!r} msg={msg!r}")

            m = CMD_RE.match(msg)
            if not m:
                continue

            query = m.group(2).strip()
            log(f"command matched; query={query!r}")

            try:
                payload = api_search(query, k=max(RESULTS_K, 8))
                results = payload.get("results", [])
                reply = format_results(query, results)
            except Exception as e:
                reply = f"Search failed: {e}"
                log(f"api error: {e}")

            if reply_to:
                send_signal(reply_to, reply)

if __name__ == "__main__":
    main()
