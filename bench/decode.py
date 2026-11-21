"""Time warm, single-stream completions on an OpenAI-compatible endpoint."""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("prompts", type=Path, help='JSONL input, one {"prompt":"..."} per line')
parser.add_argument("--url", default="http://127.0.0.1:8080")
parser.add_argument("--model", default="inkling-2-mini-q4")
parser.add_argument("--tokens", type=int, default=1024)
parser.add_argument("--count", type=int, default=8)
parser.add_argument("--chat", action="store_true")
args = parser.parse_args()
model = args.model
base = args.url.rstrip("/")
raw = args.prompts.read_bytes()
prompts = [json.loads(line)["prompt"] for line in raw.splitlines()[:args.count]]
limit, count, temperature, chat = args.tokens, args.count, 1.0, args.chat
if count < 1 or len(prompts) != count or limit < 2:
    parser.error("Need the requested number of prompts and at least two output tokens")
print("Prompt file SHA-256:", hashlib.sha256(raw).hexdigest(), file=sys.stderr)
headers = {"Content-Type": "application/json"}
if key := os.environ.get("INKLING_API_KEY"):
    headers["Authorization"] = "Bearer " + key


def run(prompt, n):
    body = dict(model=model, prompt=prompt, temperature=temperature, top_p=1.0,
                top_k=-1, min_p=0, seed=42, max_tokens=n, stream=True,
                stream_options={"include_usage": True}, ignore_eos=not chat)
    endpoint = "/v1/completions"
    if chat:
        endpoint = "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": body.pop("prompt")}]
    req = urllib.request.Request(base + endpoint, json.dumps(body).encode(),
                                 headers)
    start = time.monotonic()
    first = None
    chunks = []
    usage = {}
    ended = False
    finish_reason = None
    with urllib.request.urlopen(req, timeout=900) as stream:
        for raw in stream:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                ended = True
                break
            event = json.loads(data)
            if "error" in event:
                raise ValueError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
                if chat:
                    delta = choice.get("delta", {})
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    part = reasoning + (delta.get("content") or "")
                else:
                    part = choice.get("text")
                if part:
                    now = time.monotonic()
                    if first is None:
                        first = now
                    chunks.append(part)
    total = time.monotonic() - start
    tokens = usage.get("completion_tokens", 0)
    stopped_chat = chat and finish_reason == "stop" and 1 <= tokens < n
    if not ended or (tokens != n and not stopped_chat) or not chunks:
        raise ValueError((ended, tokens, n, len(chunks)))
    text = "".join(chunks)
    return dict(tokens=tokens, characters=len(text), finish_reason=finish_reason, seconds=total, ttft=first-start,
                tpot=(total-(first-start))/(tokens-1) if tokens > 1 else None, sample=text[:100])

run("Give a detailed explanation of tides.", 32)
rows = []
for index, prompt in enumerate(prompts[:count]):
    row = run(prompt, limit)
    rows.append(row)
    print(json.dumps(dict(prompt=index, **row)), flush=True)
elapsed = sum(r["seconds"] for r in rows)
intervals = [r["tpot"] for r in rows if r["tpot"] is not None]
mean_tpot = statistics.mean(intervals) if intervals else 0
print(json.dumps(dict(model=model, endpoint="chat" if chat else "completion", decode=1/mean_tpot if mean_tpot > 0 else None,
                      e2e=sum(r["tokens"] for r in rows)/elapsed,
                      characters_per_second=sum(r["characters"] for r in rows)/elapsed)), flush=True)
