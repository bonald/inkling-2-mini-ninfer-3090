#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
models=${1:-/models}
mkdir -p "$models"
exec 9>"$models/.prepare.lock"
flock 9
if [[ -d "$models/native" ]]; then
    exec python "$here/verify-native.py" "$models/native"
fi
python "$here/download-model.py" "$models"
stage=$(mktemp -d "$models/.convert.XXXXXXXX")
trap 'find "$stage" -depth -delete' EXIT
out="$stage/native"
mkdir -m 755 "$out"
python "$here/prepare-draft.py" "$models/dspark-BF16.gguf" "$stage/draft-small.gguf"
echo "40b53936fb1f53cf1ac7edde238acddf6ef25fc95c613060ee8699ce2332136d  $stage/draft-small.gguf" | sha256sum -c -
name=Inkling-2-Mini-UD-IQ4_XS
python "$here/prepare-head.py" "$models/$name.gguf" "$out/proposal-q4.q4"
python -m tools.convert --model "$models/frontend" --recipe inkling2_gguf \
    --source "gguf=$models/$name.gguf" --source "vision=$models/mmproj-BF16.gguf" \
    --source "dspark=$stage/draft-small.gguf" --components text,vision,dspark \
    --name "$name" --out "$out/Inkling-q4.ninfer" --device cpu
python "$here/verify-native.py" "$out"
mv "$out/Inkling-q4.ninfer.conversion.json" "$stage/"
chmod 644 "$out/Inkling-q4.ninfer" "$out/proposal-q4.q4"
mv -T "$out" "$models/native"
