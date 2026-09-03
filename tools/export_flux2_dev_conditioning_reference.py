#!/usr/bin/env python3
"""Emit the FLUX.2 [dev] conditioning reference for one prompt (offline oracle).

Reproduces the official embedder path with transformers: system message +
user prompt through the Mistral Small 3.1 chat template
(add_generation_prompt=False, padding="max_length", truncation, 512 tokens,
the tokenizer's own padding side), Mistral3ForConditionalGeneration with
output_hidden_states=True, hidden states 10/20/30 stacked to [512, 15360].

Writes <out>.safetensors with:
  input_ids      I32 [512]
  attention_mask I32 [512]
  conditioning   BF16 [512, 15360]   (torch.stack(...).rearrange b c l d -> b l (c d))
  hidden_10 / hidden_20 / hidden_30   BF16 [512, 5120]
and prints the padding side, valid token count and tensor stats.

Usage:
  export_flux2_dev_conditioning_reference.py --model-dir SNAPSHOT_DIR
      --prompt "A cat holding a sign that says hello world" --out ref
      [--device cpu|cuda] [--dtype bfloat16]
The snapshot dir is the HF FLUX.2-dev snapshot holding text_encoder/ and
tokenizer/. 48 GB of BF16 weights: on cpu this needs the host RAM; on cuda
it needs device_map offload.
"""
import argparse, json, os, sys, time

import torch
from safetensors.torch import save_file

SYSTEM_MESSAGE = ("You are an AI that reasons about image descriptions. You give "
                  "structured responses focusing on object relationships, object "
                  "attribution and actions without speculation.")
OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
MAX_LENGTH = 512

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--fixture-out", default="",
                help="also write the gate fixture: valid-token rows of the conditioning "
                     "plus input_ids/attention_mask/position_ids (see tools/gate_flux2_dev_conditioning.py)")
a = ap.parse_args()

from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

tok = AutoTokenizer.from_pretrained(os.path.join(a.model_dir, "tokenizer"))
print("padding_side:", tok.padding_side, "pad_token_id:", tok.pad_token_id, "bos:", tok.bos_token_id)
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
    {"role": "user", "content": [{"type": "text", "text": a.prompt.replace("[IMG]", "")}]},
]
enc = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True,
                              return_dict=True, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=MAX_LENGTH)
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]
valid = int(attention_mask.sum())
print("valid tokens:", valid, "first ids:", input_ids[0, :8].tolist(), "last ids:", input_ids[0, -8:].tolist())
print("rendered:", tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[:200])

dtype = getattr(torch, a.dtype)
t0 = time.time()
model = Mistral3ForConditionalGeneration.from_pretrained(
    os.path.join(a.model_dir, "text_encoder"), torch_dtype=dtype,
    device_map=a.device if a.device == "cpu" else "auto", low_cpu_mem_usage=True)
model.eval()
print(f"loaded in {time.time() - t0:.1f} s")
with torch.no_grad():
    out = model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device),
                output_hidden_states=True, use_cache=False)
hs = out.hidden_states
stacked = torch.stack([hs[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)  # [b, 3, l, d]
b, c, l, d = stacked.shape
conditioning = stacked.permute(0, 2, 1, 3).reshape(b, l, c * d)  # b l (c d)
print("conditioning:", tuple(conditioning.shape), "dtype", conditioning.dtype,
      "mean %.5f std %.5f" % (conditioning.float().mean().item(), conditioning.float().std().item()))
save_file({
    "input_ids": input_ids[0].to(torch.int32).contiguous(),
    "attention_mask": attention_mask[0].to(torch.int32).contiguous(),
    # transformers derives position_ids from cache_position = arange(seq) when
    # the caller passes only input_ids + attention_mask (the official embedder
    # does), so left-padded rows occupy positions 0..P-1 and real tokens start
    # at P. Recorded explicitly so the compiler side binds the same positions.
    "position_ids": torch.arange(MAX_LENGTH, dtype=torch.int32).contiguous(),
    "conditioning": conditioning[0].to(torch.bfloat16).cpu().contiguous(),
    "hidden_10": hs[10][0].to(torch.bfloat16).cpu().contiguous(),
    "hidden_20": hs[20][0].to(torch.bfloat16).cpu().contiguous(),
    "hidden_30": hs[30][0].to(torch.bfloat16).cpu().contiguous(),
}, a.out + ".safetensors", metadata={"prompt": a.prompt, "padding_side": tok.padding_side,
                                     "valid_tokens": str(valid), "layers": json.dumps(OUTPUT_LAYERS_MISTRAL)})
print("wrote", a.out + ".safetensors")
if a.fixture_out:
    valid = attention_mask[0].bool()
    save_file({
        "input_ids": input_ids[0].to(torch.int32).contiguous(),
        "attention_mask": attention_mask[0].to(torch.int32).contiguous(),
        "position_ids": torch.arange(MAX_LENGTH, dtype=torch.int32).contiguous(),
        "conditioning_valid_rows": conditioning[0][valid].to(torch.bfloat16).cpu().contiguous(),
    }, a.fixture_out, metadata={"prompt": a.prompt, "padding_side": tok.padding_side,
                                "valid_tokens": str(int(valid.sum())), "layers": json.dumps(OUTPUT_LAYERS_MISTRAL),
                                "system_message": SYSTEM_MESSAGE})
    print("wrote fixture", a.fixture_out)
