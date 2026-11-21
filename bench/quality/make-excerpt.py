"""Recreate the 4,210-token excerpt from the pinned WikiText-2 raw test split."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from tokenizers import Tokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("archive", type=Path)
parser.add_argument("tokenizer", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
with args.archive.open("rb") as stream:
    if hashlib.file_digest(stream, "sha256").hexdigest() != "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11":
        parser.error("Expected the pinned WikiText-2 raw v1 archive")
with zipfile.ZipFile(args.archive) as archive:
    text = archive.read("wikitext-2-raw/wiki.test.raw").decode("utf-8")[:40000]
tokens = Tokenizer.from_file(str(args.tokenizer)).encode(text, add_special_tokens=False)
excerpt = text[:tokens.offsets[4210][0]].encode("utf-8")
if hashlib.sha256(excerpt).hexdigest() != "772a3960f9ba6936c1624f7c61fb8eb68076adcf331bf93bd144622c7af908ce":
    parser.error("Excerpt differs from the scored text")
args.output.write_bytes(excerpt)
