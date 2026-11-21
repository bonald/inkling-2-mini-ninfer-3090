"""Compare complete next-token distributions exported by the Inkling quality tools."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def open_scores(path):
    with path.open("rb") as stream:
        vocab = int.from_bytes(stream.read(4), "little")
    if vocab <= 0 or path.suffix not in {".bf16", ".f32"}:
        raise ValueError(f"Invalid score file {path}")
    scalar = "<u2" if path.suffix == ".bf16" else "<f4"
    dtype = np.dtype([("target", "<i4"), ("logits", scalar, (vocab,))])
    if (path.stat().st_size - 4) % dtype.itemsize:
        raise ValueError(f"Incomplete output record in {path}")
    return np.memmap(path, mode="r", offset=4, dtype=dtype), vocab


def log_softmax(scores):
    if scores.dtype == np.uint16:
        scores = (scores.astype("uint32") << 16).view("float32")
    x = scores.astype("float64")
    maximum = x.max(axis=1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ValueError("Invalid next-token logits")
    x -= maximum
    x -= np.log(np.exp(x).sum(axis=1, keepdims=True))
    return x


def compare(reference, candidate, window_tokens):
    left, vocab = open_scores(reference)
    right, other_vocab = open_scores(candidate)
    if (not len(left) or vocab != other_vocab or
            not np.array_equal(left["target"], right["target"])):
        raise ValueError("Scores must use the same vocabulary and literal target token IDs")
    targets = left["target"]
    if targets.min() < 0 or targets.max() >= vocab:
        raise ValueError("Target token ID outside the vocabulary")
    if window_tokens < 2 or len(left) % (window_tokens - 1):
        raise ValueError("Expected complete, disjoint scoring windows")

    matched = runner_up = top_five_disagreements = 0
    divergence = 0.0
    tv, ref_loss, trial_loss = [], [], []
    for start in range(0, len(left), 12):
        end = min(start + 12, len(left))
        lp = log_softmax(left["logits"][start:end])
        lq = log_softmax(right["logits"][start:end])
        p, q = np.exp(lp), np.exp(lq)
        choices = lq.argmax(1)
        agree = lp.argmax(1) == choices
        matched += int(agree.sum())
        # Tied reference scores share their best rank.
        selected = lp[np.arange(end - start), choices]
        ranks = 1 + (lp[~agree] > selected[~agree, None]).sum(axis=1)
        runner_up += int((ranks == 2).sum())
        top_five_disagreements += int((ranks <= 5).sum())
        with np.errstate(invalid="ignore"):
            divergence += np.where(p > 0, p * (lp - lq), 0).sum()
        tv.extend(0.5 * np.abs(p - q).sum(axis=1))
        rows, ids = np.arange(end - start), targets[start:end]
        ref_loss.extend(-lp[rows, ids])
        trial_loss.extend(-lq[rows, ids])

    changes = np.subtract(trial_loss, ref_loss).reshape(-1, window_tokens - 1).mean(axis=1)
    return dict(scored_tokens=len(left), vocabulary=vocab, top1_matches=matched,
                disagreements_at_rank_2=runner_up, disagreements_in_top_5=top_five_disagreements,
                reference_perplexity=math.exp(np.mean(ref_loss)),
                candidate_perplexity=math.exp(np.mean(trial_loss)),
                mean_kl_nats=float(divergence / len(left)), mean_total_variation=float(np.mean(tv)),
                p95_total_variation=float(np.quantile(tv, 0.95)), window_mean_nll_changes=changes.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--window-tokens", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.candidate, args.window_tokens), allow_nan=False))
