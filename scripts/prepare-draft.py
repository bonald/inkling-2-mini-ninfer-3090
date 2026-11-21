"""Convert only the 27 DSpark backbone projections from publisher BF16 to Q4_0."""
import argparse
import os
import struct
import tempfile
from pathlib import Path

from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize, quantize

Q = GGMLQuantizationType


def convert(source, destination):
    bf16 = GGUFReader(str(source))
    backbone = ("attn_k", "attn_k_ctx", "attn_output", "attn_q", "attn_v", "attn_v_ctx", "ffn_down", "ffn_gate", "ffn_up")
    projections = {f"blk.{layer}.{name}.weight" for layer in range(3) for name in backbone}
    if len(bf16.tensors) != 54 or not projections.issubset(t.name for t in bf16.tensors):
        raise ValueError("Expected the pinned BF16 Inkling 2 Mini DSpark tensor layout")
    with source.open("rb") as file:
        header = bytearray(file.read(bf16.data_offset))
    file_type = bf16.fields["general.file_type"]
    value_offset = file_type.offset + sum(part.nbytes for part in file_type.parts[:file_type.data[0]])
    struct.pack_into("<I", header, value_offset, 2)  # GGUF MOSTLY_Q4_0
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", mode="wb+") as output:
        output.write(header)
        offset = 0
        for tensor in sorted(bf16.tensors, key=lambda t: t.data_offset):
            field = tensor.field
            type_offset = field.offset + sum(part.nbytes for part in field.parts[:4])
            struct.pack_into("<Q", header, type_offset + 4, offset)
            output.seek(bf16.data_offset + offset)
            if tensor.name in projections:
                if tensor.tensor_type != Q.BF16:
                    raise ValueError(f"Expected BF16 for {tensor.name}")
                struct.pack_into("<I", header, type_offset, int(Q.Q4_0))
                for start in range(0, tensor.data.shape[0], 256):
                    rows = dequantize(tensor.data[start:start + 256], Q.BF16)
                    output.write(quantize(rows, Q.Q4_0).tobytes())
            else:
                output.write(tensor.data.tobytes())
            length = output.tell() - bf16.data_offset
            offset = (length + bf16.alignment - 1) // bf16.alignment * bf16.alignment
        output.truncate(bf16.data_offset + offset)
        output.seek(0)
        output.write(header)
        output.flush()
        os.fchmod(output.fileno(), 0o644)
        os.link(output.name, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_bf16", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    convert(args.source_bf16, args.destination)
