#!/usr/bin/env bash
set -euo pipefail
cd /src
mkdir -p /notices/{upstream,sources}
cp -a /project /notices/build-recipe
cp LICENSE CONTRIBUTORS.md /notices/upstream/
cp source.tgz third_party/media-sources/*.tar.xz /notices/sources/
cp /build/third_party/media/sources.json /notices/upstream/media-sources.json
find third_party -type f \( -iname 'license*' -o -iname 'notice*' -o -iname 'copying*' \) \
    -exec cp --parents '{}' /notices/upstream/ \;
sed -n '1,/^ \*\//p' third_party/xgrammar/3rdparty/picojson/picojson.h > /notices/upstream/picojson-license.txt
sed -n '1,/^ \*\//p' third_party/spdlog/include/spdlog/fmt/bundled/format.h > /notices/upstream/fmt-license.txt
tar -xOf third_party/media-sources/ffmpeg-9.0.2.tar.xz ffmpeg-9.0.2/COPYING.LGPLv2.1 > /notices/upstream/FFmpeg-LGPL-2.1.txt
tar -xOf third_party/media-sources/dav1d-1.5.4.tar.xz dav1d-1.5.4/COPYING > /notices/upstream/dav1d-BSD.txt
