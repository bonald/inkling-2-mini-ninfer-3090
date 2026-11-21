"""Compare final answers on the same 32 GSM8K and 32 ARC-Challenge test questions."""
import argparse
import hashlib
import io
import json
import re
import urllib.request
from decimal import Decimal
from pathlib import Path

RELEASES = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main",
              "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge",
            "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9"),
}
NUMBER = r"[+-]?(?:\d+(?:,\d{3})*)(?:\.\d+)?"


def normalize(text, suite):
    lines = text.strip().splitlines()
    if not lines:
        return None
    last = lines[-1].strip().replace("**", "").strip(" $")
    last = re.sub(r"^(?:(?:the )?(?:final )?answer(?: is|:)\s*|####\s*)", "", last, flags=re.I)
    last = re.sub(r"^\\boxed\{([^{}]+)\}\.?$", r"\1", last).strip(" $()").removesuffix(".")
    if suite == "arc":
        return last.upper() if re.fullmatch(r"[A-Ea-e1-5]", last) else None
    last = re.sub(r"^([+-])\$", r"\1", last)
    if re.fullmatch(NUMBER, last):
        return format(Decimal(last.replace(",", "")).normalize(), "f")
    return None


def questions(suite):
    import pyarrow.parquet as pq
    repo, revision, config, sha256 = RELEASES[suite]
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{config}/test-00000-of-00001.parquet"
    with urllib.request.urlopen(url, timeout=90) as stream:
        blob = stream.read()
    if hashlib.sha256(blob).hexdigest() != sha256:
        raise ValueError(f"{repo} test split checksum mismatch")
    records = pq.read_table(io.BytesIO(blob)).to_pylist()
    def rank(i):
        return hashlib.sha256(f"Inkling-Q4-comparison-v1/{suite}/{i}".encode()).digest()
    for i in sorted(range(len(records)), key=rank)[:32]:
        row = records[i]
        if suite == "arc":
            choices = list(zip(row["choices"]["label"], row["choices"]["text"], strict=True))
            prompt = row["question"] + "\n" + "\n".join(f"{label}. {text}" for label, text in choices)
            prompt += "\nReply with only the label of the correct choice."
            expected = row["answerKey"]
        else:
            prompt = row["question"] + "\nReply with only the final numerical answer."
            expected = row["answer"].rsplit("#### ", 1)[1]
        expected = normalize(expected, suite)
        if expected is None:
            raise ValueError(f"Unrecognized answer key for {suite} row {i}")
        yield i, prompt, expected


def capture(args):
    headers = {"Content-Type": "application/json"}
    if args.key_file:
        headers["Authorization"] = "Bearer " + args.key_file.read_text().strip()
    for suite in RELEASES:
        for i, prompt, expected in questions(suite):
            body = dict(model=args.model, messages=[dict(role="user", content=prompt)], temperature=0,
                        top_k=0, top_p=1, min_p=0, seed=117, max_tokens=2048, reasoning_effort="none",
                        chat_template_kwargs={"reasoning_effort": "none"})
            req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                         json.dumps(body).encode(), headers)
            with urllib.request.urlopen(req, timeout=300) as stream:
                result = json.load(stream)
            reply = result["choices"][0]
            text = reply["message"].get("content") or ""
            print(json.dumps(dict(suite=suite, row=i, expected=expected, answer=normalize(text, suite),
                                  text=text, tokens=result["usage"]["completion_tokens"],
                                  finish=reply["finish_reason"], model=args.model)), flush=True)


def summarize(reference, candidate):
    def load(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["answer"] = normalize(row["text"], row["suite"])
        keyed = {(r["suite"], r["row"]): r for r in rows}
        if len(rows) != 64 or any(sum(k[0] == suite for k in keyed) != 32 for suite in RELEASES):
            raise ValueError(f"Incomplete or duplicate questions in {path}")
        return keyed

    base, other = load(reference), load(candidate)
    if base.keys() != other.keys():
        raise ValueError("Test questions differ")
    for suite in RELEASES:
        keys = [k for k in base if k[0] == suite]
        if any(base[k]["expected"] != other[k]["expected"] for k in keys):
            raise ValueError("Reference labels differ")
        left = {k for k in keys if base[k]["expected"] == base[k]["answer"]}
        right = {k for k in keys if other[k]["expected"] == other[k]["answer"]}
        print(json.dumps(dict(suite=suite, total=len(keys), reference_correct=len(left),
                              candidate_correct=len(right), regressions=sorted(k[1] for k in left - right),
                              improvements=sorted(k[1] for k in right - left),
                              same_nonempty_answer=sum(base[k]["answer"] is not None and
                                                       base[k]["answer"] == other[k]["answer"] for k in keys),
                              reference_length_stops=sum(base[k]["finish"] == "length" for k in keys),
                              candidate_length_stops=sum(other[k]["finish"] == "length" for k in keys))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    collect = modes.add_parser("capture")
    collect.add_argument("url")
    collect.add_argument("model")
    collect.add_argument("--key-file", type=Path)
    compare = modes.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        capture(args)
    else:
        summarize(args.reference, args.candidate)
