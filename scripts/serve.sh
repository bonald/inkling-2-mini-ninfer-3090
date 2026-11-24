#!/usr/bin/env bash
set -euo pipefail
export INKLING_STABLE_PREFIX=1 INKLING_FULL_SAMPLE=1 INKLING_RESIDUAL_DRAW=1 INKLING_WIDTH_POLICY=1
export INKLING_Q8_GROUP=1 INKLING_PACK_EXPERTS=1 INKLING_DRAFT_TAIL=1 INKLING_MMA_PREFILL=1
export INKLING_SHARED_VERIFY_ATTN="${INKLING_SHARED_VERIFY_ATTN:-1}"
export INKLING_VERIFY_MMA="${INKLING_VERIFY_MMA:-2}"
export INKLING_FAST_DOWN="${INKLING_FAST_DOWN:-1}"
export INKLING_PROPOSAL_HEAD=/models/native/proposal-q4.q4
if [[ -z "${INKLING_DRAFT_VOCAB+x}" && -r /usr/local/share/inkling/draft-vocab-32k.bin ]]; then
    export INKLING_DRAFT_VOCAB=/usr/local/share/inkling/draft-vocab-32k.bin
fi
model=/models/native/Inkling-q4.ninfer
if [[ ! -r "$model" || ! -r "$INKLING_PROPOSAL_HEAD" ]]; then
    echo 'Run docker compose run --rm prepare first to prepare the Inkling Q4 files.' >&2
    exit 1
fi
exec ninfer-serve "$model" --model-id inkling-2-mini-q4 --host 0.0.0.0 --port 8080 \
    --unload-idle-seconds 600 --max-context "${INKLING_MAX_CONTEXT:-131072}" \
    --max-concurrency "${INKLING_MAX_CONCURRENCY:-1}" --kv-dtype "${INKLING_KV_DTYPE:-mxfp8}" \
    --vision --structured-output --host-cache-mib 2048 \
    --temperature 1 --top-k 0 --top-p 1 --min-p 0 --spec dspark \
    --draft-tokens "${INKLING_DRAFT_TOKENS:-7}" --adaptive-mtp --draft-min-p 0
