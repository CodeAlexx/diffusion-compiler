#!/usr/bin/env python3
"""CPU-only source-oracle gate for H3 media mechanics, not neural quality.

Executes selected *actual source methods* without importing the model runtime
or creating randomly initialized neural networks. Encoder callbacks return
native saved boundary fixtures. This isolates temporal concatenation, spatial
seams, posterior draw/round/normalize, and control row ordering. It neither
validates the learned encoder nor admits ControlNet/video generation quality.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import torch


def tensor(path):
    data = path.read_bytes()
    if data[:8] != b"DIFTNS01":
        raise ValueError(f"native tensor magic mismatch: {path}")
    if hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError(f"native tensor checksum mismatch: {path}")
    version, dtype, rank = struct.unpack_from("<III", data, 8)
    if version != 1 or dtype != 1:
        raise ValueError("oracle fixtures must be version-one F32 tensors")
    dims = struct.unpack_from(f"<{rank}Q", data, 20)
    size = struct.unpack_from("<Q", data, 20 + 8 * rank)[0]
    start = 28 + 8 * rank
    if size != math.prod(dims) * 4 or start + size + 32 != len(data):
        raise ValueError("native tensor payload does not match shape")
    return torch.frombuffer(bytearray(data[start:start + size]), dtype=torch.float32).reshape(dims)


def source_class(path, name, methods):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    selected = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if {n.name for n in selected} != set(methods):
        raise ValueError(f"source method contract changed: {path}")
    definition = ast.ClassDef(name="SourceOracle", bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace = {"torch": torch, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"LATENTS_MEAN", "LATENTS_STD"}:
                    constants[target.id] = ast.literal_eval(node.value)
    identity = {"path": str(path), "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "methods": {n.name: [n.lineno, n.end_lineno] for n in selected}}
    return namespace["SourceOracle"], constants, identity


def source_patchify(path):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    names = {"patchify_video", "pad_to_patch"}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("source control packing contract changed")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, {"path": str(path), "sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "methods": {n.name: [n.lineno, n.end_lineno] for n in selected}}


def compare(name, expected, actual, exact=True):
    if expected.shape != actual.shape:
        raise AssertionError(f"{name}: shape mismatch {expected.shape} != {actual.shape}")
    a, b = expected.double().flatten(), actual.double().flatten()
    error = b - a
    denom = max(float(torch.linalg.vector_norm(a)), 1e-30)
    bit_mismatches = int(torch.count_nonzero(expected.contiguous().view(torch.int32) != actual.contiguous().view(torch.int32)))
    cosine = float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))
    result = {"boundary": name, "shape": list(actual.shape), "cosine": cosine,
              "relative_l2": float(torch.linalg.vector_norm(error)) / denom,
              "max_abs": float(error.abs().max()),
              "norm_ratio": float(torch.linalg.vector_norm(b)) / denom,
              "nonfinite": int(torch.count_nonzero(~torch.isfinite(actual))),
              "bit_mismatches": bit_mismatches,
              "reference_sha256": hashlib.sha256(expected.contiguous().numpy().tobytes()).hexdigest(),
              "native_sha256": hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest()}
    # Fixed bars before execution: copies, packing and rounded CPU posterior
    # must be byte-exact. No looser fallback threshold is used on failure.
    result["bar"] = "bit_exact" if exact else "relative_l2<=1e-7 and max_abs<=1e-6 and cosine>=0.999999999999"
    result["passed"] = result["nonfinite"] == 0 and (bit_mismatches == 0 if exact else
        result["relative_l2"] <= 1e-7 and result["max_abs"] <= 1e-6 and cosine >= 0.999999999999)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True, type=Path,
                        help="fresh output from dif_h3_encode_tests DIR")
    parser.add_argument("--serenityflow", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.serenityflow / "serenityflow/models/minimax_h3"
    cls, constants, vae_identity = source_class(root / "vae/comfy_impl/video_vae.py", "MiniMaxH3VideoVAE",
        {"encode_temporal", "split_tiles", "blend", "tiled_encode", "encode"})
    packing, packing_identity = source_patchify(root / "timeline.py")
    control_cls, _, control_identity = source_class(root / "control.py", "H3FunControlNet",
                                                  {"init_stream", "_video_positions"})
    control_cls.init_stream.__globals__["T"] = SimpleNamespace(**packing)
    load = lambda name: tensor(args.fixtures / f"{name}.diftensor")
    results = []

    # Source temporal code sees a real distinct-frame sequence. Only the neural
    # encode call is replaced with the saved exact moments boundary.
    temporal = cls()
    temporal.clip_length, temporal.token_drop = 17, 3
    temporal._normalize_pixels = lambda x: x
    clips = [load(f"temporal-clip-{i}") for i in range(8)]
    seen = []
    def encoded_clip(pixels):
        index = len(seen)
        wanted = torch.arange(index * 17, index * 17 + 17).clamp_max(123).float()
        if not torch.equal(pixels[0, 0, :, 0, 0], wanted):
            raise AssertionError("source temporal fixture does not preserve actual frame order/final hold")
        seen.append(index)
        return clips[index].clone()
    temporal._adaptive_encode = encoded_clip
    pixels = torch.arange(124).float().view(1, 1, 124, 1, 1).expand(1, 3, 124, 2, 2)
    results.append(compare("temporal_concat_then_drop_124_to_37", temporal.encode_temporal(pixels, "cpu"), load("temporal-joined")))

    spatial = cls()
    spatial.tile_size, spatial.tile_overlap_min, spatial.vae_ratio = 64, 32, 16
    tiles = iter([load(name) for name in ("spatial-left", "spatial-right", "spatial-bottom-left", "spatial-bottom-right")])
    spatial._encode_moments = lambda _: next(tiles).clone()
    results.append(compare("spatial_raw_neighbor_corner_and_time", spatial.tiled_encode(torch.zeros(1, 3, 17, 96, 96)), load("spatial-stitched")))

    posterior = cls()
    posterior.latents_mean = torch.tensor(constants["LATENTS_MEAN"], dtype=torch.float32)
    posterior.latents_std = torch.tensor(constants["LATENTS_STD"], dtype=torch.float32)
    posterior.encode_temporal = lambda x, device: load("posterior-moments")
    for sample in (False, True):
        actual = posterior.encode(torch.zeros(1, 3, 22, 2, 2), device="cpu", sample_posterior=sample,
                                  generator=torch.Generator(device="cpu").manual_seed(42), round_posterior=sample)
        name = "posterior-sample" if sample else "posterior-mean"
        results.append(compare(name, actual, load(name)))

    for channels in (24, 49):
        latent = load("posterior-mean") if channels == 24 else load("control-49-latent")
        expected = load(f"control-{channels}-rows")
        control = control_cls()
        control.spec = SimpleNamespace(patch_size=(1, 2, 2))
        control.streamer = None
        control.control_blocks = [SimpleNamespace(
            adaln_proj=SimpleNamespace(linear=SimpleNamespace(in_features=4)), before_proj=lambda x: x)]
        captured = []
        def projection_boundary(rows):
            captured.append(rows.clone())
            return torch.zeros(rows.shape[0], 4)
        projection_boundary.weight = torch.zeros(4, 196)
        control.control_proj_in = projection_boundary
        plan = SimpleNamespace(video_is_target=torch.ones(expected.shape[0], dtype=torch.bool),
            segments=[SimpleNamespace(stream="video", start=1, stop=expected.shape[0] + 1)])
        # Execute actual init_stream up to and including layout/insertion; only
        # learned projections are identity/zero boundary callbacks.
        control.init_stream(torch.zeros(expected.shape[0] + 2, 4), latent, plan, torch.zeros(1, 4))
        results.append(compare(f"control_{channels}_channel_rows", captured[0], expected))
    report = {"status": "PASS" if all(x["passed"] for x in results) else "FAIL",
              "scope": "CPU media mechanics at saved neural boundaries; no learned encoder, ControlNet forward, GPU, PTF or decoded-quality claim",
              "sources": [vae_identity, packing_identity, control_identity], "torch": torch.__version__, "results": results}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
