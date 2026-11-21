"""Time idle unloading and the next request on a standalone Inkling Q4 server."""
import json
import sys
import time
import urllib.request

base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
model = "inkling-2-mini-q4"


def state():
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as stream:
        entries = json.load(stream)["data"]
    entry, = [item for item in entries if item["id"] == model]
    return entry["status"]["value"]


def complete():
    payload = dict(model=model, prompt="Two plus two equals", max_tokens=4, temperature=0)
    request = urllib.request.Request(base + "/v1/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as stream:
        result = json.load(stream)
    if result["usage"]["completion_tokens"] <= 0 or state() != "loaded":
        raise RuntimeError("Expected a completed request and a resident model")


complete()
start = time.monotonic()
last_loaded = 0
while True:
    time.sleep(5)
    status = state()
    elapsed = time.monotonic() - start
    if status == "unloaded":
        break
    if elapsed > 630 or status not in {"loaded", "unloading"}:
        raise SystemExit(f"Expected idle unload after 600 seconds, saw {status} at {elapsed:.1f}s")
    if status == "loaded":
        last_loaded = elapsed
if elapsed < 595 or last_loaded > 610:
    raise SystemExit(f"Idle unload outside the expected interval: {last_loaded:.1f}s to {elapsed:.1f}s")
complete()
print(f"Unloaded between {last_loaded:.1f}s and {elapsed:.1f}s; the next completion reloaded the model.")
