#!/usr/bin/env python3
"""Download or verify the pinned Inkling Q4 model, drafter, projector and tokenizer."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess

BASE = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/ed40e7da625f594fc9c4990f5f66be24873c22d7"
GGUF = "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/ea0c79b2ecd73669e774c8519b79bfabfddbc027"
VISION = "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f"
TARGET = "Inkling-2-Mini-UD-IQ4_XS.gguf"
FILES = [
    (TARGET, f"{GGUF}/{TARGET}", "a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1"),
    ("dspark-BF16.gguf", f"{GGUF}/dspark/dspark-Inkling-2-Mini-BF16.gguf", "b8985695c949a4e3270666b44b65959a8c3a599c6e599e0e3e5028f8c217f7b2"),
    ("mmproj-BF16.gguf", f"{VISION}/mmproj-BF16.gguf", "2c165c78790e78a3f45a2d1e9e931083e52ee69d45ff2a0dc9c281fdf1a666e7"),
    ("frontend/config.json", f"{BASE}/config.json", "a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2"),
    ("frontend/tokenizer.json", f"{BASE}/tokenizer.json", "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6"),
    ("frontend/tokenizer_config.json", f"{BASE}/tokenizer_config.json", "bf09342d285139fd37ce10c70878af1d52cb390e9901e10248654293a80ee636"),
    ("frontend/chat_template.jinja", f"{BASE}/chat_template.jinja", "53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc"),
    ("frontend/generation_config.json", f"{BASE}/generation_config.json", "a08737c07a65a8fdf69f7cf086576210e5aad4b241437090b26d7b0c4f7905f7"),
]


def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(root, item, verify_only):
    name, url, expected = item
    path = root / name
    if path.exists() or verify_only:
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"Missing or invalid SHA-256 for {path}. Move a damaged file aside before retrying.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        # A previous interruption may have completed the transfer before promotion.
        if not partial.exists() or digest(partial) != expected:
            subprocess.run(["curl", "--fail", "--location", "--show-error", "--silent",
                            "--retry", "5", "--continue-at", "-", "--output", str(partial), url], check=True)
            if digest(partial) != expected:
                partial.unlink()
                raise SystemExit(f"SHA-256 mismatch for {name}. Discarded incomplete download.")
        partial.replace(path)
    print(f"Verified {name}  {expected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("models"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        args.directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if args.verify_only else fcntl.LOCK_EX)
        for item in FILES:
            fetch(args.directory, item, args.verify_only)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
