#!/usr/bin/env python3
"""Development-only FLUX.2 Qwen first-divergence oracle.

This script is not part of the native product path. It instantiates official
Transformers Qwen3 decoder modules, loads one layer at a time from the pinned
BF16 oracle checkpoint, and writes depth and first-layer boundary tensors for
comparison with DiffIR. Production prompt conditioning remains C++.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class ShardReader:
    def __init__(self, directory: Path):
        index_path = directory / "model.safetensors.index.json"
        self.directory = directory
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self.files: dict[Path, object] = {}

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.directory / self.weight_map[name]
        handle = self.files.get(shard)
        if handle is None:
            handle = safe_open(shard, framework="pt", device="cpu")
            self.files[shard] = handle
        return handle.get_tensor(name)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    text_encoder = args.checkpoint / "text_encoder"
    config = Qwen3Config.from_json_file(text_encoder / "config.json")
    config._attn_implementation = "sdpa"
    if args.depth <= 0 or args.depth > config.num_hidden_layers:
        raise SystemExit("--depth is outside the Qwen3 depth")

    reader = ShardReader(text_encoder)
    fixture = load_file(args.inputs)
    input_ids = fixture["input_ids"].to(args.device, dtype=torch.long)
    attention_mask = fixture["attention_mask"].to(args.device, dtype=torch.long)
    embedding = reader.tensor("model.embed_tokens.weight").to(args.device)
    hidden = torch.nn.functional.embedding(input_ids, embedding)
    del embedding

    position_ids = torch.arange(
        hidden.shape[1], device=args.device, dtype=torch.long
    ).unsqueeze(0)
    cache_position = position_ids[0]
    rotary = Qwen3RotaryEmbedding(config, device=args.device)
    position_embeddings = rotary(hidden, position_ids)
    causal_mask = create_causal_mask(
        config=config,
        input_embeds=hidden,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    captures: dict[str, torch.Tensor] = {
        "embedding": hidden.detach(),
        "rope_cos": position_embeddings[0].detach(),
        "rope_sin": position_embeddings[1].detach(),
    }

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captures[name] = value.detach()

        return hook

    def capture_attention_input(_module, inputs):
        captures["attention"] = inputs[0].detach()

    for layer_index in range(args.depth):
        prefix = f"model.layers.{layer_index}."
        with torch.device("meta"):
            layer = Qwen3DecoderLayer(config, layer_index)
        layer.to_empty(device=args.device)
        layer.to(dtype=torch.bfloat16)
        state = {
            name.removeprefix(prefix): reader.tensor(name).to(args.device)
            for name in reader.weight_map
            if name.startswith(prefix)
        }
        layer.load_state_dict(state, strict=True)
        del state
        layer.eval()

        handles = []
        layer_input = hidden
        if layer_index == 0:
            modules = {
                "input_norm": layer.input_layernorm,
                "q_proj": layer.self_attn.q_proj,
                "k_proj": layer.self_attn.k_proj,
                "v_proj": layer.self_attn.v_proj,
                "q_norm": layer.self_attn.q_norm,
                "k_norm": layer.self_attn.k_norm,
                "attention_projected": layer.self_attn.o_proj,
                "post_attention_norm": layer.post_attention_layernorm,
                "mlp_gate": layer.mlp.gate_proj,
                "mlp_up": layer.mlp.up_proj,
                "mlp_down": layer.mlp.down_proj,
            }
            handles = [
                module.register_forward_hook(capture(name))
                for name, module in modules.items()
            ]
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    capture_attention_input
                )
            )

        with torch.no_grad():
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        for handle in handles:
            handle.remove()

        if layer_index == 0:
            rotated_q, rotated_k = apply_rotary_pos_emb(
                captures["q_norm"].transpose(1, 2),
                captures["k_norm"].transpose(1, 2),
                position_embeddings[0],
                position_embeddings[1],
            )
            captures["rotated_q"] = rotated_q.transpose(1, 2).detach()
            captures["rotated_k"] = rotated_k.transpose(1, 2).detach()
            captures["attention_residual"] = (
                layer_input + captures["attention_projected"]
            ).detach()
            captures["mlp_gate_activated"] = layer.mlp.act_fn(
                captures["mlp_gate"]
            ).detach()
            captures["mlp_gated"] = (
                captures["mlp_gate_activated"] * captures["mlp_up"]
            ).detach()
            captures["layer_output"] = hidden.detach()
        captures[f"hidden_{layer_index + 1:02d}"] = hidden.detach()
        del layer

    selected = [
        captures[f"hidden_{index:02d}"]
        for index in (9, 18, 27)
        if index <= args.depth
    ]
    if selected:
        captures["conditioning"] = torch.cat(selected, dim=-1).detach()

    serialized = {
        name: value.contiguous().cpu() for name, value in captures.items()
    }
    save_file(
        serialized,
        args.output,
        metadata={
            "oracle": "transformers.Qwen3DecoderLayer",
            "checkpoint": str(args.checkpoint.resolve()),
            "depth": str(args.depth),
            "attention": "sdpa",
            "torch": torch.__version__,
        },
    )
    print(
        f"FLUX2_QWEN_ORACLE output={args.output} "
        f"shape={list(hidden.shape)} dtype={hidden.dtype} captures={len(serialized)}"
    )


if __name__ == "__main__":
    main()
