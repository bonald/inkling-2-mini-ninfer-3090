#!/usr/bin/env bash
set -euo pipefail
export INKLING_STABLE_PREFIX=1 INKLING_FULL_SAMPLE=1 INKLING_RESIDUAL_DRAW=1 INKLING_WIDTH_POLICY=1
export INKLING_Q8_GROUP=1 INKLING_PACK_EXPERTS=1 INKLING_DRAFT_TAIL=1 INKLING_MMA_PREFILL=1
export INKLING_PROPOSAL_HEAD=/models/native/proposal-q4.q4
model=/models/native/Inkling-q4.ninfer
if [[ ! -r "$model" || ! -r "$INKLING_PROPOSAL_HEAD" ]]; then
    echo 'Run docker compose run --rm prepare first to prepare the Inkling Q4 files.' >&2
    exit 1
fi
exec ninfer-serve "$model" --model-id inkling-2-mini-q4 --host 0.0.0.0 --port 8080 \
    --unload-idle-seconds 600 --max-context 65536 --max-concurrency 1 --vision --structured-output --host-cache-mib 2048 \
    --temperature 1 --top-k 0 --top-p 1 --min-p 0 --spec dspark --draft-tokens 2 --draft-min-p 0
