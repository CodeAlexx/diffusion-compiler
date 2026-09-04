#!/usr/bin/env python3
"""Creator oracle for the SDXL text conditioning: the reference sampler's
own tokenizer and CLIP-L / OpenCLIP-G towers (F32 through its manual-cast
ops) on a prompt, with the token ids, per-tower boundaries, the concatenated
2048-wide context and the projected pooled vector."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sdxl_reference_common import (
    CLIP_G_PREFIX, CLIP_L_PREFIX, Captures, add_reference_source, cpu, refuse_overwrite, save)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    refuse_overwrite(args.output)
    commit = add_reference_source(args.reference_source)
    import comfy.sd  # noqa: E402
    import comfy.model_management  # noqa: E402

    # The creator's own single-file loader: it recognizes the SDXL dual
    # tower and converts the OpenCLIP-G layout the way the sampler does.
    clip = comfy.sd.load_checkpoint_guess_config_clip_only(str(args.checkpoint))
    if not hasattr(clip.cond_stage_model, "clip_g"):
        raise SystemExit("the reference did not build the SDXL dual tower")
    tokens = clip.tokenize(args.prompt)
    l_ids = [t for t, _w in tokens["l"][0]]
    g_ids = [t for t, _w in tokens["g"][0]]
    if len(tokens["l"]) != 1 or len(tokens["g"]) != 1:
        raise SystemExit("prompt spans several 77-token chunks; keep the oracle to one")
    valid = 1 + sum(1 for t in g_ids if t != 0)  # BOS ... EOS then zero pads (G)
    g_eos = g_ids.index(49407)

    comfy.model_management.load_model_gpu(clip.patcher)
    towers = clip.cond_stage_model
    captures = Captures()
    for tag, tower in (("l", towers.clip_l), ("g", towers.clip_g)):
        text_model = tower.transformer.text_model
        layers = text_model.encoder.layers
        captures.output(layers[0], f"{tag}_layer_0")
        # hidden_states[-2]: the raw residual after len(layers) - 1 layers
        captures.output(layers[len(layers) - 2], f"{tag}_hidden")
        captures.output(text_model.final_layer_norm, f"{tag}_final_norm")
        captures.input(layers[0], f"{tag}_embeddings")
    with torch.no_grad():
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        l_out, _l_pooled = towers.clip_l.encode_token_weights(tokens["l"])
        g_out, g_pooled = towers.clip_g.encode_token_weights(tokens["g"])
    captures.release()
    values = {
        "l_token_ids": torch.tensor(l_ids, dtype=torch.int32),
        "g_token_ids": torch.tensor(g_ids, dtype=torch.int32),
        "g_pooled_row": torch.tensor([g_eos], dtype=torch.int32),
        "l_out": cpu(l_out), "g_out": cpu(g_out), "context": cpu(cond),
        "pooled": cpu(pooled), "g_pooled": cpu(g_pooled),
    }
    values.update(captures.values)
    save(args.output, values, {
        "creator_commit": commit, "checkpoint": args.checkpoint.name, "prompt": args.prompt,
        "valid_tokens": str(valid),
    })


if __name__ == "__main__":
    main()
