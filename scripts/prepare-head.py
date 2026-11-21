"""Create the drafter-only Q4_0 output projection from a target GGUF."""
import argparse
import os
import tempfile
from pathlib import Path
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize, quantize

def convert(model, destination):
    if destination.exists():
        raise FileExistsError(destination)
    tensor, = [t for t in GGUFReader(str(model)).tensors if t.name == "output.weight"]
    with tempfile.NamedTemporaryFile(dir=destination.parent, mode="wb") as f:
        for start in range(0, tensor.data.shape[0], 256):
            weights = dequantize(tensor.data[start:start + 256], tensor.tensor_type)
            f.write(quantize(weights, GGMLQuantizationType.Q4_0).tobytes())
        f.flush()
        os.fchmod(f.fileno(), 0o644)
        os.link(f.name, destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.model, args.destination)
