#!/usr/bin/env python3
"""Frozen CPU F32 fixtures from the pinned MiniMax-H3 creator audio encoder.

This is an offline source oracle, never a native runtime dependency. Generation
imports the actual Diffusers checkout only after argument parsing, constructs a
small but complete encoder, and captures submodule outputs during its full
``encode(...).latent_dist.mode()``. It does not reproduce network equations.

The two mono batch items represent asymmetric stereo channels. The 47 samples
exercise creator right-padding to 50; strides 2 and 5 exercise both even and odd
stride padding. Four heads of width 8 exercise head averaging and integral 8->4
adaptive pooling. Raw weight_g/weight_v and the frozen zero key bias are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


PINNED_REVISION = "e1b518dfd5e390e7ba09a79a1d39fe1c6cb52dc1"
SOURCE_FILE = "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py"
SOURCE_MODULE = "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio"
ENCODER_PREFIXES = ("encoder.", "pre_block.", "mean_proj.", "logs_proj.")
CONFIG = {
    "encoder_dim": 4,
    "encoder_rates": [2, 5],
    "latent_dim": 32,
    "latent_channels": 4,
    "num_attention_heads": 4,
    "eps": 1e-5,
    "batch": 2,
    "samples": 47,
}
EXPECTED_SHAPES = {
    "in.waveform": [2, 1, 47],
    "out.trunk": [2, 32, 5],
    "out.pre_block": [2, 4, 5],
    "out.mean": [2, 4, 5],
}
# Frozen before the first native measurement; no CLI threshold overrides.
F32_BARS = {
    "cosine_min": 0.999999,
    "relative_l2_max": 1e-5,
    "max_abs_max": 1e-4,
    "norm_ratio_abs_error_max": 1e-5,
    "nonfinite_max": 0,
    "bit_mismatches": "reported_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    # Exclusive creation preserves earlier gate evidence on accidental reruns.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def validate_source(root: Path) -> dict:
    source = root / SOURCE_FILE
    if not source.is_file():
        raise ValueError(f"creator source is missing: {source}")
    revision = git_output(root, "rev-parse", "HEAD")
    if revision != PINNED_REVISION:
        raise ValueError(f"creator revision must be {PINNED_REVISION}, got {revision}")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no", "--", "src/diffusers")
    if dirty:
        raise ValueError(f"creator Diffusers source has tracked changes:\n{dirty}")
    return {"root": str(root), "revision": revision, "file": SOURCE_FILE,
            "sha256": sha256_file(source)}


def tensor_record(tensor) -> dict:
    array = tensor.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": "F32",
            "sha256": hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()}


def imported_source_hashes(root: Path) -> dict:
    files = set()
    for name, module in tuple(sys.modules.items()):
        if name != "diffusers" and not name.startswith("diffusers."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path)))
        if not path.is_relative_to(root / "src"):
            raise ValueError(f"Diffusers module was imported outside pinned source: {name}: {path}")
        if path.is_file():
            files.add(path)
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(files)}


def generate(args: argparse.Namespace) -> int:
    root = args.source_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"output directory already exists; choose a fresh evidence path: {output}")
    source = validate_source(root)

    # Set before importing any neural libraries. No downloads, accelerator
    # selection, BF16 cast, attention replacement, or creator patch is involved.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_ATTN_BACKEND"] = "native"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.path.insert(0, str(root / "src"))
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    creator = importlib.import_module(SOURCE_MODULE)
    if Path(creator.__file__).resolve() != root / SOURCE_FILE:
        raise ValueError(f"wrong creator module imported: {creator.__file__}")

    # Tiny decoder construction is required by the source class constructor;
    # the decoder is never evaluated or included in the encoder weight file.
    model_kwargs = {name: CONFIG[name] for name in (
        "encoder_dim", "encoder_rates", "latent_dim", "latent_channels", "num_attention_heads"
    )}
    model_kwargs.update(decoder_dim=8, decoder_rates=[5, 2], decoder_kernel_sizes=[9, 4],
                        resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]],
                        sampling_rate=32000)
    with torch.device("cpu"):
        model = creator.AutoencoderKLMiniMaxH3Audio(**model_kwargs).float().eval()
    for name, module in model.pre_block.named_modules():
        if isinstance(module, torch.nn.LayerNorm) and module.eps != CONFIG["eps"]:
            raise ValueError(f"source LayerNorm epsilon differs at pre_block.{name}: {module.eps}")

    randomized = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not name.startswith(ENCODER_PREFIXES):
                continue
            values = torch.randn_like(parameter) * 0.25
            if name.endswith("alpha"):
                values = values.abs() + 0.3
            elif name.endswith("weight_g"):
                values = values.abs() + 0.5
            elif ".norm" in name and name.endswith("weight"):
                values = values + 0.9
            parameter.copy_(values)
            randomized.append(name)
        if not torch.equal(model.pre_block.attn.zero_k_bias,
                           torch.zeros_like(model.pre_block.attn.zero_k_bias)):
            raise ValueError("creator zero_k_bias must remain a frozen zero buffer")

        wave = torch.randn((2, 1, CONFIG["samples"]), dtype=torch.float32, device="cpu") * 0.2
        ramp = torch.linspace(-0.3, 0.3, CONFIG["samples"], dtype=torch.float32, device="cpu")
        wave[0, 0] += ramp
        wave[1, 0] = wave[1, 0] * 0.6 - ramp * 0.5 + 0.1
        wave[0, 0, 0] += 0.7
        wave[1, 0, -1] -= 0.5
    captures = {}

    def capture_trunk(_module, _inputs, result):
        captures["out.trunk"] = result.detach().clone().contiguous()

    def capture_pre_block(_module, _inputs, result):
        captures["out.pre_block"] = result.detach().transpose(1, 2).contiguous().clone()

    hooks = [model.encoder.register_forward_hook(capture_trunk),
             model.pre_block.register_forward_hook(capture_pre_block)]
    try:
        with torch.inference_mode():
            mean = model.encode(wave).latent_dist.mode()
    finally:
        for hook in hooks:
            hook.remove()
    references = {"in.waveform": wave.contiguous(), **captures,
                  "out.mean": mean.detach().contiguous()}
    if set(references) != set(EXPECTED_SHAPES):
        raise ValueError(f"incomplete source captures: {sorted(references)}")
    for name, tensor in references.items():
        if (list(tensor.shape) != EXPECTED_SHAPES[name] or tensor.dtype != torch.float32
                or tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"invalid CPU F32 reference tensor: {name}: {tensor.shape}, {tensor.dtype}")
    if torch.equal(references["out.mean"][0], references["out.mean"][1]):
        raise ValueError("source mean did not distinguish the asymmetric stereo batch items")
    weights = {name: tensor.detach().cpu().float().contiguous()
               for name, tensor in model.state_dict().items() if name.startswith(ENCODER_PREFIXES)}
    if not any(name.endswith("weight_g") for name in weights) or not any(
        name.endswith("weight_v") for name in weights
    ):
        raise ValueError("source must preserve raw weight-normalization parameters")
    for name, tensor in weights.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"nonfinite source weight: {name}")
    if validate_source(root) != source:
        raise ValueError("creator source changed during fixture generation")

    manifest = {
        "schema": "dif.h3_audio_encoder_fixture.v1",
        "oracle": "AutoencoderKLMiniMaxH3Audio.encode(...).latent_dist.mode()",
        "source": {**source, "imported_files_sha256": imported_source_hashes(root)},
        "generator": {"file": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
        "seed": args.seed,
        "runtime": {"torch": torch.__version__, "diffusers": importlib.import_module("diffusers").__version__,
                    "device": "cpu", "dtype": "F32", "attention_backend": "native",
                    "threads": 1, "deterministic_algorithms": True},
        "constructor": model_kwargs,
        "config": CONFIG,
        "geometry": {"padded_samples": 50, "frames": 5, "head_dim": 8, "pool_output_dim": 4},
        "scope": "bounded encoder tensor parity; not released dimensions or decoded audio quality",
        "capture": "forward hooks during the single full creator encode call",
        "randomized_parameters": randomized,
        "frozen_zero_buffer": "pre_block.attn.zero_k_bias",
        "weight_format": "encoder-only creator state_dict; raw weight_g/weight_v retained",
        "logs_proj": "saved because full creator encode evaluates it; native posterior-mode graph need not bind it",
        "reference_tensors": {name: tensor_record(tensor) for name, tensor in references.items()},
        "weight_tensors": {name: tensor_record(tensor) for name, tensor in weights.items()},
        "bars": {name: dict(F32_BARS) for name in EXPECTED_SHAPES if name.startswith("out.")},
        "native_comparison": "not_run",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", CONFIG)
    save_file(weights, str(output / "weights.safetensors"))
    save_file(references, str(output / "reference.safetensors"))
    manifest["files_sha256"] = {name: sha256_file(output / name) for name in (
        "config.json", "weights.safetensors", "reference.safetensors"
    )}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": "fixture_written", "output_dir": str(output),
                      "reference_shapes": EXPECTED_SHAPES, "weight_tensors": len(weights),
                      "source_revision": source["revision"], "native_comparison": "not_run"}, indent=2))
    return 0


def compare(args: argparse.Namespace) -> int:
    """Read native snapshots without loading Torch or executing either model."""
    import numpy as np
    from safetensors.numpy import load_file

    fixture = args.output_dir.resolve()
    native_path = args.compare.resolve()
    if args.report is not None and args.report.exists():
        raise ValueError(f"comparison report already exists: {args.report}")
    with (fixture / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != "dif.h3_audio_encoder_fixture.v1":
        raise ValueError("unsupported fixture manifest schema")
    expected_bars = {name: dict(F32_BARS) for name in EXPECTED_SHAPES if name.startswith("out.")}
    if manifest.get("bars") != expected_bars:
        raise ValueError("fixture bars do not match the frozen F32 gate")
    if manifest.get("config") != CONFIG or manifest.get("source", {}).get("revision") != PINNED_REVISION:
        raise ValueError("fixture configuration or pinned source revision differs")
    for name in ("config.json", "weights.safetensors", "reference.safetensors"):
        if sha256_file(fixture / name) != manifest.get("files_sha256", {}).get(name):
            raise ValueError(f"fixture file hash mismatch: {name}")
    reference = load_file(str(fixture / "reference.safetensors"))
    native = load_file(str(native_path))
    for label, tensors in (("reference", reference), ("native", native)):
        for name, shape in EXPECTED_SHAPES.items():
            if name not in tensors:
                raise ValueError(f"missing {label} tensor: {name}")
            array = tensors[name]
            if list(array.shape) != shape or array.dtype != np.dtype("float32"):
                raise ValueError(f"invalid {label} tensor {name}: expected F32 {shape}, "
                                 f"got {array.dtype} {list(array.shape)}")
    for name, expected in reference.items():
        if name not in EXPECTED_SHAPES:
            raise ValueError(f"unexpected reference tensor: {name}")
        digest = hashlib.sha256(expected.astype("<f4", copy=False).tobytes()).hexdigest()
        if digest != manifest.get("reference_tensors", {}).get(name, {}).get("sha256"):
            raise ValueError(f"reference tensor hash mismatch: {name}")
    input_bytes = native["in.waveform"].astype("<f4", copy=False).tobytes()
    if input_bytes != reference["in.waveform"].astype("<f4", copy=False).tobytes():
        raise ValueError("native in.waveform is not byte-identical to the frozen oracle input")
    results = {}
    first_divergent = None
    last_good = "in.waveform"
    for name in expected_bars:
        expected = reference[name]
        actual = native[name]
        nonfinite = int(np.count_nonzero(~np.isfinite(expected)) + np.count_nonzero(~np.isfinite(actual)))
        metrics = {
            "cosine": None, "relative_l2": None, "max_abs": None, "norm_ratio": None,
            "nonfinite": nonfinite,
            "bit_mismatches": int(np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32))),
        }
        if nonfinite == 0:
            ref64 = expected.astype(np.float64).ravel()
            got64 = actual.astype(np.float64).ravel()
            ref_norm = float(np.linalg.norm(ref64))
            got_norm = float(np.linalg.norm(got64))
            diff = got64 - ref64
            error_norm = float(np.linalg.norm(diff))
            both_zero = ref_norm == 0 and got_norm == 0
            cosine = (float(np.dot(ref64, got64)) / (ref_norm * got_norm)
                      if ref_norm and got_norm else float(both_zero))
            metrics.update(
                cosine=min(1.0, max(-1.0, cosine)),
                relative_l2=error_norm / ref_norm if ref_norm else (0.0 if both_zero else None),
                max_abs=float(np.max(np.abs(diff))),
                norm_ratio=got_norm / ref_norm if ref_norm else (1.0 if both_zero else None),
            )
        bars = expected_bars[name]
        passed = (
            metrics["nonfinite"] <= bars["nonfinite_max"]
            and metrics["cosine"] is not None and metrics["cosine"] >= bars["cosine_min"]
            and metrics["relative_l2"] is not None and metrics["relative_l2"] <= bars["relative_l2_max"]
            and metrics["max_abs"] is not None and metrics["max_abs"] <= bars["max_abs_max"]
            and metrics["norm_ratio"] is not None
            and abs(metrics["norm_ratio"] - 1.0) <= bars["norm_ratio_abs_error_max"]
        )
        results[name] = {"passed": passed, "metrics": metrics, "bars": bars,
                         "native_sha256": hashlib.sha256(actual.astype("<f4", copy=False).tobytes()).hexdigest()}
        if first_divergent is None:
            if passed:
                last_good = name
            else:
                first_divergent = name
    report = {
        "schema": "dif.h3_audio_encoder_comparison.v1",
        "passed": first_divergent is None,
        "scope": manifest["scope"],
        "fixture": str(fixture),
        "manifest_sha256": sha256_file(fixture / "manifest.json"),
        "native_file": str(native_path),
        "native_file_sha256": sha256_file(native_path),
        "input_byte_identical": True,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "source_revision": PINNED_REVISION,
        "first_divergent_captured_boundary": first_divergent,
        "last_good_captured_boundary": last_good,
        "boundaries": results,
    }
    if args.report is not None:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="fresh directory to generate, or existing fixture directory with --compare")
    parser.add_argument("--source-root", type=Path,
                        help="pinned Diffusers checkout containing src/diffusers; required to generate")
    parser.add_argument("--seed", type=lambda value: int(value, 0),
                        help="explicit deterministic seed; decimal or 0x-prefixed integer")
    parser.add_argument("--compare", type=Path,
                        help="native safetensors containing in.waveform and all three out.* captures")
    parser.add_argument("--report", type=Path,
                        help="optional fresh comparison JSON path; existing reports are never overwritten")
    args = parser.parse_args(argv)
    if args.compare is None and (args.source_root is None or args.seed is None):
        parser.error("generation requires both --source-root and --seed")
    if args.compare is not None and (args.source_root is not None or args.seed is not None):
        parser.error("--compare reads the frozen fixture; omit --source-root and --seed")
    if args.compare is None and args.report is not None:
        parser.error("--report requires --compare")
    if args.seed is not None and not 0 <= args.seed <= 2**63 - 1:
        parser.error("--seed must be in [0, 2**63 - 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return compare(args) if args.compare is not None else generate(args)
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        print(f"gate_h3_audio_encoder: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
