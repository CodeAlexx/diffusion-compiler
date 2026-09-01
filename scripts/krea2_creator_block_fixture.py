#!/usr/bin/env python3
"""Generate the real-dimension Krea 2 block-0 creator parity fixture.

PyTorch is used only as the development oracle. The accepted native path
loads the raw checkpoint and executes the same boundaries through DiffIR.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_COMMIT = "db3984fbc6e13b34c0064990fc2d95ac64d00058"
FEATURES = 6144
HEADS = 48
KV_HEADS = 12
HEAD_DIM = 128
MLP_DIM = 16384
TEXT_TOKENS = 512
IMAGE_TOKENS = 4096
SEQUENCE = TEXT_TOKENS + IMAGE_TOKENS
VALID_TEXT_TOKENS = 128
SEED = 20260831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--input-fixture", type=Path)
    parser.add_argument("--final-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.creator))
    from mmdit import (  # pylint: disable=import-error,import-outside-toplevel
        SingleStreamBlock,
        _mask,
        attention,
        rope,
        ropeapply,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Krea 2 creator fixture requires CUDA")
    if args.block < 0 or args.block >= 28:
        raise ValueError("--block must be in [0,27]")

    started = time.perf_counter()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False

    block = SingleStreamBlock(
        features=FEATURES,
        heads=HEADS,
        multiplier=4,
        bias=False,
        kvheads=KV_HEADS,
    )
    prefix = f"blocks.{args.block}."
    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        names = sorted(name for name in checkpoint.keys() if name.startswith(prefix))
        state = {name[len(prefix) :]: checkpoint.get_tensor(name) for name in names}
    missing, unexpected = block.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    del state
    block = block.to(device="cuda", dtype=torch.bfloat16).eval()

    if args.input_fixture:
        with safe_open(args.input_fixture, framework="pt", device="cpu") as previous:
            names = set(previous.keys())
            if "final_output" in names:
                sequence_cpu = previous.get_tensor("final_output")
            elif "sequence_input" in names:
                sequence_cpu = previous.get_tensor("sequence_input")
            elif {"context_input", "projected_image"} <= names:
                sequence_cpu = torch.cat(
                    (
                        previous.get_tensor("context_input"),
                        previous.get_tensor("projected_image"),
                    ),
                    dim=1,
                )
            else:
                raise ValueError("input fixture has no usable block sequence")
            modulation_name = (
                "modulation_input" if "modulation_input" in names else "modulation_output"
            )
            modulation_cpu = previous.get_tensor(modulation_name).reshape(
                1, 6 * FEATURES
            )
            positions_cpu = previous.get_tensor("positions")
            validity_cpu = previous.get_tensor("validity_mask")
        sequence = sequence_cpu.cuda()
        modulation = modulation_cpu.cuda()
        positions = positions_cpu.cuda()
        validity = validity_cpu.cuda()
    else:
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        sequence_f32 = torch.randn((1, SEQUENCE, FEATURES), generator=generator)
        sequence = sequence_f32.to(torch.bfloat16).cuda()
        del sequence_f32
        modulation_f32 = torch.randn((1, 6 * FEATURES), generator=generator)
        modulation = modulation_f32.to(torch.bfloat16).cuda()
        del modulation_f32
        positions = torch.zeros((1, SEQUENCE, 3), dtype=torch.float32, device="cuda")
        image_y = torch.arange(64, dtype=torch.float32, device="cuda")[:, None]
        image_x = torch.arange(64, dtype=torch.float32, device="cuda")[None, :]
        positions[0, TEXT_TOKENS:, 1] = image_y.expand(64, 64).reshape(-1)
        positions[0, TEXT_TOKENS:, 2] = image_x.expand(64, 64).reshape(-1)
        validity = torch.zeros((1, SEQUENCE), dtype=torch.bool, device="cuda")
        validity[:, :VALID_TEXT_TOKENS] = True
        validity[:, TEXT_TOKENS:] = True
    expanded_mask = _mask(validity)
    freqs = torch.cat(
        [rope(positions[..., axis], dim, 1000.0) for axis, dim in enumerate([32, 48, 48])],
        dim=-3,
    )

    def native_heads(value: torch.Tensor) -> torch.Tensor:
        return value.permute(0, 2, 1, 3).contiguous()

    captures: dict[str, torch.Tensor] = {
        "sequence_input": sequence.cpu(),
        "modulation_input": modulation.cpu(),
        "positions": positions.cpu(),
        "validity_mask": validity.cpu(),
    }
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    compute_started = time.perf_counter()
    with torch.inference_mode():
        modulated = modulation + block.mod.lin
        prescale, preshift, pregate, postscale, postshift, postgate = modulated.chunk(6, dim=-1)
        input_normalized = block.prenorm(sequence)
        attention_input = (1 + prescale[:, None, :]) * input_normalized + preshift[:, None, :]

        query = rearrange(block.attn.wq(attention_input), "B L (H D) -> B H L D", H=HEADS)
        key = rearrange(block.attn.wk(attention_input), "B L (H D) -> B H L D", H=KV_HEADS)
        value = rearrange(block.attn.wv(attention_input), "B L (H D) -> B H L D", H=KV_HEADS)
        query, key, value = block.attn.qknorm(query, key, value)
        rotary_query, rotary_key = ropeapply(query, key, freqs)
        attention_output = attention(
            rotary_query,
            rotary_key,
            value,
            mask=expanded_mask,
            gqa=True,
        )
        attention_gate = torch.sigmoid(block.attn.gate(attention_input))
        output_projection = block.attn.wo(attention_output * attention_gate)
        attention_residual = sequence + pregate[:, None, :] * output_projection

        post_normalized = block.postnorm(attention_residual)
        mlp_input = (1 + postscale[:, None, :]) * post_normalized + postshift[:, None, :]
        mlp_gate = block.mlp.gate(mlp_input)
        mlp_gate_activated = torch.nn.functional.silu(mlp_gate)
        mlp_up = block.mlp.up(mlp_input)
        mlp_activation = mlp_gate_activated * mlp_up
        mlp_output = block.mlp.down(mlp_activation)
        final_output = attention_residual + postgate[:, None, :] * mlp_output
    torch.cuda.synchronize()
    compute_seconds = time.perf_counter() - compute_started

    if args.final_only:
        captures.update({"final_output": final_output.cpu()})
    else:
        captures.update({
            "modulated_parameters": modulated.cpu(),
            "input_normalized": input_normalized.cpu(),
            "attention_input": attention_input.cpu(),
            "query": native_heads(query).cpu(),
            "key": native_heads(key).cpu(),
            "value": native_heads(value).cpu(),
            "rotary_query": native_heads(rotary_query).cpu(),
            "rotary_key": native_heads(rotary_key).cpu(),
            "attention_output": attention_output.reshape(SEQUENCE, FEATURES).cpu(),
            "attention_gate": attention_gate.reshape(SEQUENCE, FEATURES).cpu(),
            "output_projection": output_projection.reshape(SEQUENCE, FEATURES).cpu(),
            "attention_residual": attention_residual.cpu(),
            "mlp_input": mlp_input.cpu(),
            "mlp_gate": mlp_gate.reshape(SEQUENCE, MLP_DIM).cpu(),
            "mlp_up": mlp_up.reshape(SEQUENCE, MLP_DIM).cpu(),
            "mlp_gate_activated": mlp_gate_activated.reshape(
                SEQUENCE, MLP_DIM).cpu(),
            "mlp_activation": mlp_activation.reshape(SEQUENCE, MLP_DIM).cpu(),
            "mlp_output": mlp_output.reshape(SEQUENCE, FEATURES).cpu(),
            "final_output": final_output.cpu(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        captures,
        args.output,
        metadata={
            "creator_commit": SOURCE_COMMIT,
            "checkpoint": str(args.checkpoint),
            "block": str(args.block),
            "seed": str(SEED),
            "geometry": "B1_L4608_D6144_text512_image4096",
            "dtype": "BF16",
        },
    )
    report = {
        "creator_commit": SOURCE_COMMIT,
        "checkpoint": str(args.checkpoint),
        "block": args.block,
        "seed": SEED,
        "sequence": SEQUENCE,
        "valid_text_tokens": VALID_TEXT_TOKENS,
        "image_tokens": IMAGE_TOKENS,
        "dtype": "BF16",
        "compute_seconds": compute_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "fixture": str(args.output),
        "input_fixture": str(args.input_fixture) if args.input_fixture else None,
        "final_only": args.final_only,
        "tensors": {name: list(tensor.shape) for name, tensor in captures.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
