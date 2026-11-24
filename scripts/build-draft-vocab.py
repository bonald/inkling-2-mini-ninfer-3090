#!/usr/bin/env python3
"""Build a sorted int32 draft-vocabulary subset binary for INKLING_DRAFT_VOCAB.

Tokenizes a local directory of text/code files with the Inkling 2 tokenizer,
ranks tokens by corpus frequency, always retains the 256 raw byte tokens and
special tokens (199998..vocab_size-1), and writes the top-N token IDs in
strictly increasing int32 order.

Usage:
    python3 scripts/build-draft-vocab.py --tokenizer models/frontend/tokenizer.json \
        --corpus /path/to/code-and-docs --size 32768 --out assets/draft-vocab-32k.bin
"""
import argparse
import collections
import os
import struct
from tokenizers import Tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Path to Inkling 2 tokenizer.json")
    ap.add_argument("--corpus", required=True, nargs="+", help="Directory or file paths to scan")
    ap.add_argument("--size", type=int, default=32768, help="Draft vocab size (default: 32768)")
    ap.add_argument("--max-files", type=int, default=2000)
    ap.add_argument("--max-bytes-per-file", type=int, default=131072)
    ap.add_argument("--out", required=True, help="Output .bin path")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    exts = {".py", ".rs", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp",
            ".js", ".ts", ".tsx", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".sh"}
    counts: collections.Counter[int] = collections.Counter()
    files_read = 0
    total_tokens = 0

    for root_path in args.corpus:
        if os.path.isfile(root_path):
            paths = [root_path]
        else:
            paths = []
            for dp, dn, fn in os.walk(root_path):
                dn[:] = [d for d in dn if d not in {".git", "build", "target", "node_modules", "__pycache__"}]
                for f in fn:
                    if os.path.splitext(f)[1].lower() in exts:
                        paths.append(os.path.join(dp, f))
        for p in paths:
            if files_read >= args.max_files:
                break
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read(args.max_bytes_per_file)
            except OSError:
                continue
            if not text:
                continue
            ids = tok.encode(text, add_special_tokens=False).ids
            counts.update(ids)
            total_tokens += len(ids)
            files_read += 1

    # Always include raw byte tokens (0..255) and special tokens (199998..vocab_size-1)
    selected = set(range(min(256, vocab_size)))
    if vocab_size > 199998:
        selected.update(range(199998, vocab_size))
    for tid, _ in counts.most_common():
        if len(selected) >= args.size:
            break
        if 0 <= tid < vocab_size:
            selected.add(tid)
    # Pad deterministically if corpus was very small
    for tid in range(vocab_size):
        if len(selected) >= args.size:
            break
        selected.add(tid)

    ids_sorted = sorted(selected)
    assert len(ids_sorted) == args.size
    covered = sum(c for tid, c in counts.items() if tid in selected)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as out:
        out.write(struct.pack(f"<{len(ids_sorted)}i", *ids_sorted))
    cov_pct = (100.0 * covered / total_tokens) if total_tokens else 0.0
    print(f"Wrote {args.out}: {len(ids_sorted)} IDs from {files_read} files / {total_tokens:,} tokens ({cov_pct:.2f}% coverage)")


if __name__ == "__main__":
    main()
