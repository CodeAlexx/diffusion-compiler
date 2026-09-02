#!/usr/bin/env python3
"""Write a diffusion-compiler oracle fixture manifest (v1) beside a
SafeTensors payload that a per-model oracle script produced.

The manifest states provenance the compiler tools consume (difbisect
validate-oracle / --oracle-manifest): creator repository and revision, model
name, input hashes, dtype, semantic boundary, fixture version, payload path and
SHA-256, and the ordered boundary list with shapes and dtypes read from the
payload header. No tensor is interpreted; names are carried verbatim.

Development-only helper. Pure Python, no third-party imports.
"""

import argparse
import hashlib
import json
import os
import struct
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header(path):
    with open(path, "rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


DTYPE_NAMES = {"F32": "f32", "BF16": "bf16", "F16": "f16", "I8": "i8", "I32": "i32", "BOOL": "bool"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", required=True, help="SafeTensors payload written by the oracle script")
    parser.add_argument("--creator-repository", required=True)
    parser.add_argument("--creator-revision", required=True)
    parser.add_argument("--model", required=True, help="model name as the frontend knows it")
    parser.add_argument("--semantic-boundary", required=True, help="what the boundaries are, e.g. 'denoiser block outputs after text fusion'")
    parser.add_argument("--dtype", required=True, help="declared numerical class of the fixture, e.g. bf16")
    parser.add_argument("--fixture-version", required=True)
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH", help="input file the oracle consumed; hashed")
    parser.add_argument("--boundary", action="append", default=[], metavar="NAME[=TENSOR]", help="ordered boundary; TENSOR defaults to NAME")
    parser.add_argument("--note", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    header = safetensors_header(args.payload)
    boundaries = []
    names = args.boundary or list(header.keys())
    for item in names:
        name, _, tensor = item.partition("=")
        tensor = tensor or name
        if tensor not in header:
            sys.exit(f"payload has no tensor named {tensor}")
        entry = header[tensor]
        boundaries.append({
            "name": name,
            "tensor": tensor,
            "shape": entry["shape"],
            "dtype": DTYPE_NAMES.get(entry["dtype"], entry["dtype"].lower()),
        })
    inputs = []
    for item in args.input:
        name, _, path = item.partition("=")
        if not path:
            sys.exit("--input expects NAME=PATH")
        inputs.append({"name": name, "path": os.path.abspath(path), "sha256": sha256_file(path)})
    manifest = {
        "kind": "diffusion-compiler-oracle-fixture",
        "version": 1,
        "creator": {"repository": args.creator_repository, "revision": args.creator_revision},
        "model": {"name": args.model},
        "semantic_boundary": args.semantic_boundary,
        "dtype": args.dtype,
        "fixture_version": args.fixture_version,
        "inputs": inputs,
        "payload": {
            "path": os.path.relpath(os.path.abspath(args.payload), os.path.dirname(os.path.abspath(args.out)) or "."),
            "sha256": sha256_file(args.payload),
            "bytes": os.path.getsize(args.payload),
        },
        "boundaries": boundaries,
        "note": args.note,
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"ORACLE_FIXTURE_MANIFEST {args.out} boundaries={len(boundaries)} inputs={len(inputs)}")


if __name__ == "__main__":
    main()
