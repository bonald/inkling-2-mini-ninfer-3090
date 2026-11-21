"""Check a text and an image turn against the standalone Inkling Q4 server."""
import base64
import json
import struct
import sys
import urllib.request
import zlib

base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
headers = {"Content-Type": "application/json"}


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

def solid_green():
    side = 160
    pixels = (b"\0" + b"\x00\xff\x00" * side) * side
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()

model = "inkling-2-mini-q4"
tests = [
    ("What is 17 multiplied by 6? Answer only with the number.", "102"),
    ([{"type": "image_url", "image_url": {"url": solid_green()}},
      {"type": "text", "text": "Name the color filling this image. Answer in English with one word."}], "green")
]
for content, expected in tests:
    body = dict(model=model, messages=[dict(role="user", content=content)], max_tokens=256,
                temperature=0, chat_template_kwargs={"reasoning_effort": "none"})
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=900) as res:
        result = json.load(res)
    answer = result["choices"][0]["message"].get("content") or ""
    print(model, repr(answer), result["usage"]["completion_tokens"], flush=True)
    if answer.strip().casefold().strip(".* ") != expected:
        raise ValueError(f"expected {expected!r}")
