#!/usr/bin/env python3
"""Verify the native weights and inference metadata against the tested Q4 conversion."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

METADATA_SHA = "27f47ebd2accd5da7a95f9195619c1a2f9d18415b31eb70b1e76ce4ddcb44140"
PAYLOAD_SHA = "f52f91d93e47e9c6c910a665e5ac16edefcf0af3f4f42b062ca9bcbab32c1a26"
HEAD_SHA = "19bcba746f7001f22625db496a5fbc33035daf641d860cc6c05ce38044c230ad"


def verify(root):
    with (root / "Inkling-q4.ninfer").open("rb") as stream:
        magic, size, _artifact_uuid = struct.unpack("<8sQ16s", stream.read(32))
        if magic != b"NINFER\0\x03" or not 0 < size < 4 * 1024 * 1024:
            raise ValueError("Invalid NInfer v3 header")
        metadata = json.loads(stream.read(size))
        # The converter stores input paths and a random file UUID for each build.
        metadata["provenance"]["sources"] = sorted(metadata["provenance"]["sources"])
        canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        if hashlib.sha256(canonical).hexdigest() != METADATA_SHA:
            raise ValueError("Inkling Q4 configuration, tensor directory or bindings differ")
        stream.seek((32 + size + 4095) // 4096 * 4096)
        if hashlib.file_digest(stream, "sha256").hexdigest() != PAYLOAD_SHA:
            raise ValueError("Inkling Q4 tensor/resource payload differs")
    with (root / "proposal-q4.q4").open("rb") as head:
        if hashlib.file_digest(head, "sha256").hexdigest() != HEAD_SHA:
            raise ValueError("DSpark output projection differs")
    print("Verified Q4 native metadata, tensor/resource bytes and DSpark output projection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
