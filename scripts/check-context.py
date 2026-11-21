"""Check the advertised context and retrieval from a 65,504-token prompt."""
import json
import sys
import time
import urllib.request

base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
model = "inkling-2-mini-q4"


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def encode(text):
    return request("/tokenize", {"content": text, "add_special": False})["tokens"]


request("/v1/completions", dict(model=model, prompt="Start.", temperature=0, max_tokens=1))
context = request("/props")["default_generation_settings"]["n_ctx"]
if context != 65536:
    raise SystemExit(f"Expected a 65,536-token context, got {context}")
needle = encode(" The archive code is LANTERN.")
filler = encode(" Field note: four finches visited the maple tree beside the lake.")
question = encode(" What is the archive code? The archive code is")
count = context - 32 - len(needle) - len(question)
prompt = needle + (filler * (count // len(filler) + 1))[:count] + question
start = time.monotonic()
result = request("/completion", dict(model=model, prompt=prompt, n_predict=12, temperature=0,
                                     ignore_eos=True, cache_prompt=False))
elapsed = time.monotonic() - start
timings = result["timings"]
if timings["prompt_n"] != len(prompt) or timings["predicted_n"] != 12 or "LANTERN" not in result["content"]:
    raise SystemExit("Near-limit retrieval failed")
print(f"{len(prompt)} input tokens and 12 output tokens in {elapsed:.2f} seconds: {result['content']!r}")
