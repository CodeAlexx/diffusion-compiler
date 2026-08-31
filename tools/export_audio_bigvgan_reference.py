#!/usr/bin/env python3
"""Torch reference exporter for the BigVGAN native audio decode gates
(docs/BIGVGAN_DECODE_PLAN.md gates 2 and 3).

Subcommands:

  fold-dump OUT.safetensors
      Gate-2 reference: the decoder-side state dict with torch's own
      weight_norm removed (torch.nn.utils.remove_weight_norm semantics via
      per-module reconstruction), filters and passthrough tensors unchanged.
      Encoder-side tensors are dropped.

  compare-fold REF.safetensors OURS.safetensors
      Gate-2 comparator: folded conv weights vs torch reference at
      relative <= 1e-7 (float64 fold); upsample filters must equal exactly
      2x the reference filter (bit-exact: x2 is an exponent increment);
      everything else byte-identical. Census check on both sides.

  stage-dump ROWS.diftensor OUT_DIR [--stages N]
      Gate-3 reference: run the pinned-lineage vendor decoder (the
      checkpoint's own dac_bigvgan.py package — the module lineage the
      diffusers reference mirrors and the Mojo port was gated against) on
      the recorded [2T,32] audio state rows, dumping F32 boundaries:
      after dec_in_proj+conv_pre ("pre"), after each upsample stage's
      block-average ("stage0".."stage6"), and the final waveform ("tail",
      clamped). Latent denormalization (config.json latents_mean/std) is
      applied to the rows first, mirroring the pipeline.

Checkpoint paths are pinned to the accepted run's audio VAE. All math runs
on CPU in F32 with TF32 disabled.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path

import torch

torch.manual_seed(0)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

VAE_DIR = Path("/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/audio_vae")


def write_diftensor(path: Path, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    assert value.dtype == torch.float32
    raw = value.numpy().tobytes(order="C")
    shape = tuple(int(d) for d in value.shape)
    payload = bytearray(b"DIFTNS01")
    payload += struct.pack("<III", 1, 1, len(shape))
    for dimension in shape:
        payload += struct.pack("<Q", dimension)
    payload += struct.pack("<Q", len(raw))
    payload += raw
    payload += hashlib.sha256(payload).digest()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_diftensor(path: Path) -> torch.Tensor:
    blob = path.read_bytes()
    assert blob[:8] == b"DIFTNS01"
    _, dtype_code, rank = struct.unpack_from("<III", blob, 8)
    assert dtype_code == 1
    offset = 20
    dims = []
    for _ in range(rank):
        (dim,) = struct.unpack_from("<Q", blob, offset)
        dims.append(dim)
        offset += 8
    (nbytes,) = struct.unpack_from("<Q", blob, offset)
    offset += 8
    import numpy

    array = numpy.frombuffer(blob, dtype=numpy.float32, count=nbytes // 4,
                             offset=offset).reshape(dims).copy()
    return torch.from_numpy(array)


def load_state() -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(VAE_DIR / "model.safetensors", device="cpu")


def decoder_names(state) -> list[str]:
    return sorted(n for n in state
                  if n.startswith(("dec_in_proj.", "decoder.")))


def fold_dump(out_path: Path) -> None:
    """Writes TWO references: OUT (torch F32-norm remove_weight_norm
    semantics) and OUT with suffix .f64.safetensors (the float64-exact fold:
    sum of squares in float64, scale = g / float32(sqrt), w = v * scale in
    F32 — the exact arithmetic the native importer and the accepted Serenity
    decode run, so the native fold is expected BIT-EXACT against it)."""
    state = load_state()
    folded: dict[str, torch.Tensor] = {}
    folded_f64: dict[str, torch.Tensor] = {}
    names = decoder_names(state)
    count_folded = 0
    for name in names:
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            v = state[name]
            g = state[base + ".weight_g"]
            norm = v.norm(2, dim=tuple(range(1, v.dim())), keepdim=True)
            folded[base + ".weight"] = (v * (g / norm)).contiguous()
            sum_squares = v.double().pow(2).sum(
                dim=tuple(range(1, v.dim())), keepdim=True)
            scale = (g / sum_squares.sqrt().float()).float()
            folded_f64[base + ".weight"] = (v * scale).contiguous()
            count_folded += 1
        else:
            folded[name] = state[name].contiguous()
            folded_f64[name] = state[name].contiguous()
    from safetensors.torch import save_file

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(folded, str(out_path))
    f64_path = out_path.with_suffix(".f64.safetensors")
    save_file(folded_f64, str(f64_path))
    print(f"FOLD_DUMP PASS input_census={len(names)} "
          f"output_census={len(folded)} folded={count_folded} "
          f"f64_reference={f64_path}")


def compare_fold(reference_path: Path, ours_path: Path) -> None:
    from safetensors.torch import load_file

    reference = load_file(str(reference_path), device="cpu")
    ours = load_file(str(ours_path), device="cpu")
    if set(reference) != set(ours):
        missing = set(reference) ^ set(ours)
        raise SystemExit(f"FOLD_COMPARE FAIL census mismatch: {sorted(missing)[:6]}")
    worst_name, worst_rel = "", 0.0
    exact = 0
    for name, ref in reference.items():
        mine = ours[name]
        if mine.shape != ref.shape and not name.endswith(
                (".upsample.filter", ".downsample.lowpass.filter")):
            raise SystemExit(f"FOLD_COMPARE FAIL shape {name}")
        if name.endswith((".upsample.filter", ".downsample.lowpass.filter")):
            # Ours is channel-expanded [C,1,K]; reference is the shared
            # [1,1,K] tap. Upsample additionally absorbs the exact x2 ratio.
            factor = 2.0 if name.endswith(".upsample.filter") else 1.0
            expanded = (ref * factor).expand(mine.shape[0], -1, -1)
            if not torch.equal(mine, expanded):
                raise SystemExit(f"FOLD_COMPARE FAIL expanded filter not "
                                 f"bit-exact: {name}")
            exact += 1
            continue
        if name.endswith(".weight") and (name + "_g") not in reference and \
                not name.endswith(("dec_in_proj.weight",)) and \
                name.replace(".weight", ".weight_v") not in reference:
            pass
        if torch.equal(mine, ref):
            exact += 1
            continue
        denominator = ref.double().norm().item()
        rel = ((mine.double() - ref.double()).norm().item() /
               (denominator if denominator > 0 else 1.0))
        if rel > worst_rel:
            worst_rel, worst_name = rel, name
    # Against the float64-exact reference the native fold reproduces the
    # arithmetic verbatim, so the bar is bit-exactness of every folded
    # weight; the torch-F32-norm reference drifts at its own rounding
    # (measured 1.25e-7 worst on the 14336-wide conv_pre fold) and is
    # reported informationally by a second run against OUT.
    exact_mode = reference_path.name.endswith(".f64.safetensors")
    if exact_mode:
        verdict = "PASS" if worst_rel == 0.0 else "FAIL"
        print(f"FOLD_COMPARE {verdict} mode=f64-exact "
              f"tensors={len(reference)} bit_exact={exact} "
              f"worst_rel={worst_rel:.3e} worst={worst_name!r} "
              f"bar=bit-exact")
    else:
        bar = 5e-7
        verdict = "PASS" if worst_rel <= bar else "FAIL"
        print(f"FOLD_COMPARE {verdict} mode=torch-f32-norm-info "
              f"tensors={len(reference)} bit_exact={exact} "
              f"worst_rel={worst_rel:.3e} worst={worst_name!r} "
              f"bar={bar:.0e}")
    if verdict == "FAIL":
        raise SystemExit(1)


def load_vendor_decoder():
    package = types.ModuleType("audio_vae_pkg")
    package.__path__ = [str(VAE_DIR)]
    sys.modules["audio_vae_pkg"] = package

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"audio_vae_pkg.{name}", VAE_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"audio_vae_pkg.{name}"] = module
        spec.loader.exec_module(module)
        return module

    for name in ["dac_utils", "dac_activations", "dac_alias_free_filter",
                 "dac_alias_free_resample", "dac_alias_free_act",
                 "dac_attn_proj", "dac_bigvgan"]:
        load(name)
    dac_audio_vae = load("dac_audio_vae")
    metadata = json.loads((VAE_DIR / "metadata.json").read_text())
    kwargs = metadata["metadata"]["kwargs"]
    import yaml

    audio_config = yaml.safe_load((VAE_DIR / "config.yaml").read_text())
    model = dac_audio_vae.DacAudioVAE(
        encoder_rates=kwargs["encoder_rates"],
        decoder_rates=kwargs["decoder_rates"],
        attn_proj=kwargs["attn_proj"],
        decoder_type=kwargs["decoder_type"],
        decoder_dim=audio_config["model_config"]["decoder_dim"],
        vae_latent_channels=audio_config["model_config"]["vae_latent_channels"],
        sample_rate=kwargs["sample_rate"],
    )
    model.load_state_dict(load_state(), strict=True)
    return model.eval()


def stage_dump(rows_path: Path, out_dir: Path, stages: int) -> None:
    config = json.loads((VAE_DIR.parent / "audio_vae" / "config.json").read_text())
    mean = torch.tensor(config["latents_mean"], dtype=torch.float32)
    std = torch.tensor(config["latents_std"], dtype=torch.float32)

    rows = read_diftensor(rows_path)  # [2T, 32]
    assert rows.dim() == 2 and rows.shape[1] == 32
    frames = rows.shape[0] // 2
    latents = torch.empty(2, 32, frames, dtype=torch.float32)
    for stereo in range(2):  # rearrange.mojo minimax_h3_unpack_audio
        block = rows[stereo * frames:(stereo + 1) * frames]  # [T, 32]
        latents[stereo] = block.transpose(0, 1)
    latents = latents * std.view(1, 32, 1) + mean.view(1, 32, 1)

    model = load_vendor_decoder()
    decoder = model.decoder
    dec_in_proj = model.dec_in_proj if hasattr(model, "dec_in_proj") else None
    if dec_in_proj is None:
        raise SystemExit("vendor model has no dec_in_proj")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with torch.no_grad():
        hidden = decoder.conv_pre(dec_in_proj(latents))
        manifest["pre"] = {
            "shape": list(hidden.shape),
            "sha256": write_diftensor(out_dir / "pre.diftensor", hidden)}
        num_kernels = decoder.num_kernels
        for stage in range(min(stages, decoder.num_upsamples)):
            hidden = decoder.ups[stage][0](hidden)
            residual = None
            for j in range(num_kernels):
                block = decoder.resblocks[stage * num_kernels + j](hidden)
                residual = block if residual is None else residual + block
            hidden = residual / num_kernels
            manifest[f"stage{stage}"] = {
                "shape": list(hidden.shape),
                "sha256": write_diftensor(
                    out_dir / f"stage{stage}.diftensor", hidden)}
        if stages >= decoder.num_upsamples:
            hidden = decoder.activation_post(hidden)
            hidden = decoder.conv_post(hidden)
            hidden = torch.clamp(hidden, min=-1.0, max=1.0)
            manifest["tail"] = {
                "shape": list(hidden.shape),
                "sha256": write_diftensor(out_dir / "tail.diftensor", hidden)}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"STAGE_DUMP PASS boundaries={len(manifest)} frames={frames}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    command = sys.argv[1]
    if command == "fold-dump" and len(sys.argv) == 3:
        fold_dump(Path(sys.argv[2]))
    elif command == "compare-fold" and len(sys.argv) == 4:
        compare_fold(Path(sys.argv[2]), Path(sys.argv[3]))
    elif command == "stage-dump" and len(sys.argv) >= 4:
        stages = 7
        argv = sys.argv[2:]
        if "--stages" in argv:
            index = argv.index("--stages")
            stages = int(argv[index + 1])
            argv = argv[:index] + argv[index + 2:]
        stage_dump(Path(argv[0]), Path(argv[1]), stages)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
