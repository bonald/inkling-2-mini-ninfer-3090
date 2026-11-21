#!/usr/bin/env bash
set -euo pipefail
out=${1:?Usage: fetch-engine.sh DIRECTORY}
revision=395ce8b85627bcb2d53a1a81623088eab755443c
mkdir -p "$out"
curl -fsSL --retry 4 "https://codeload.github.com/iamwavecut/ninfer-all/tar.gz/$revision" -o "$out/source.tgz"
echo "9dcc9ca9728a9fcefa302ff1ab08253d3048cb7a98e2b575707d1bd47552ea91  $out/source.tgz" | sha256sum -c -
tar -xzf "$out/source.tgz" --strip-components=1 -C "$out"
