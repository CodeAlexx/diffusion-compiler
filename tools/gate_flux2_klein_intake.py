#!/usr/bin/env python3
"""Development-only BF16 Klein 4B intake gate against pinned creator forwards.

Never used by the native product. Replays exact native noise/conditioning/
schedule through creator Flux2.forward + denoise_cfg and AutoEncoder.decode.
The local BF16 Qwen checkpoint is an explicitly named alternative to BFL's
FP8 default, not a claim of FP8-checkpoint parity.
"""
import argparse
import gc
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

PIN = "50fe5162777813d869182b139e83b10743caef15"


def metrics(candidate, reference):
    a, b = candidate.cpu().double().flatten(), reference.cpu().double().flatten()
    error = a - b
    return {"cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            "relative_l2": (error.norm() / b.norm().clamp_min(1e-20)).item(),
            "max_abs": error.abs().max().item(),
            "nonfinite": int((~torch.isfinite(a)).sum() + (~torch.isfinite(b)).sum())}


def vae_alias(name):
    aliases = {"decoder.post_quant_conv.": "post_quant_conv.",
               "decoder.norm_out.": "decoder.conv_norm_out.",
               "decoder.mid.block_1.": "decoder.mid_block.resnets.0.",
               "decoder.mid.block_2.": "decoder.mid_block.resnets.1."}
    for old, new in {"norm": "group_norm", "q": "to_q", "k": "to_k", "v": "to_v", "proj_out": "to_out.0"}.items():
        aliases[f"decoder.mid.attn_1.{old}."] = f"decoder.mid_block.attentions.0.{new}."
    for i in range(4):
        aliases[f"decoder.up.{i}.block."] = f"decoder.up_blocks.{3-i}.resnets."
        aliases[f"decoder.up.{i}.upsample.conv."] = f"decoder.up_blocks.{3-i}.upsamplers.0.conv."
    for old, new in aliases.items():
        if name.startswith(old):
            return (new + name[len(old):]).replace(".nin_shortcut.", ".conv_shortcut.")
    return name


def main():
    ap = argparse.ArgumentParser()
    for name in ["creator-source", "native-state", "native-report", "checkpoint", "text-encoder", "vae-checkpoint", "output-dir"]:
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    assert subprocess.check_output(["git", "-C", str(args.creator_source), "rev-parse", "HEAD"], text=True).strip() == PIN
    sys.path.insert(0, str(args.creator_source / "src"))
    from flux2.model import Flux2, Klein4BParams
    from flux2.sampling import denoise_cfg
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    native = load_file(args.native_state)
    config = json.loads(args.native_report.read_text())
    assert config["flux2_model"] == "klein4b", "this bounded full-resident gate is for 4B"
    torch.set_grad_enabled(False)
    report = {"creator_commit": PIN, "model": "Klein Base 4B", "checkpoint": str(args.checkpoint),
              "text_encoder": str(args.text_encoder), "text_dtype": "BF16 local checkpoint, not creator-default FP8",
              "native_state": str(args.native_state), "native_report": str(args.native_report),
              "bars": {"conditioning_cosine_min": 0.999, "conditioning_rel_l2_max": 0.01,
                       "latent_cosine_min": 0.995, "latent_rel_l2_max": 0.1,
                       "same_latent_decode_psnr_min_db": 50, "trajectory_png_psnr_min_db": 30}}
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder, local_files_only=True)
    encoder = AutoModelForCausalLM.from_pretrained(args.text_encoder, torch_dtype=torch.bfloat16,
                                                  device_map="cuda", local_files_only=True).eval()
    conditioning_captures = {}
    for label, prompt in [("positive", config["prompt"]), ("negative", "")]:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to("cuda")
        result = encoder(**tokens, output_hidden_states=True, use_cache=False)
        expected = torch.cat([result.hidden_states[i] for i in [9, 18, 27]], dim=-1)[0].cpu()
        conditioning_captures[label + "_conditioning"] = expected.contiguous()
        report[label + "_conditioning"] = metrics(native[label + "_conditioning"], expected)
        print(label, report[label + "_conditioning"], flush=True)
        del result, expected, tokens
    del encoder
    gc.collect(); torch.cuda.empty_cache()

    with torch.device("meta"):
        model = Flux2(Klein4BParams())
    model.load_state_dict(load_file(args.checkpoint, device="cuda"), strict=True, assign=True)
    model.eval()
    height, width = config["height"] // 16, config["width"] // 16
    count = height * width
    image_ids = torch.zeros((1, count, 4), dtype=torch.float32, device="cuda")
    image_ids[0, :, 1] = torch.arange(count, device="cuda") // width
    image_ids[0, :, 2] = torch.arange(count, device="cuda") % width
    text_ids = torch.zeros((2, 512, 4), dtype=torch.float32, device="cuda")
    text_ids[:, :, 3] = torch.arange(512, device="cuda")
    latent = denoise_cfg(model, native["initial_image_tokens"][None].cuda(), image_ids,
                         torch.stack([native["negative_conditioning"], native["positive_conditioning"]]).cuda(),
                         text_ids, native["timesteps"].float().tolist(), config["guidance"])[0].cpu()
    report["final_latent"] = metrics(native["final_image_tokens"], latent)
    del model
    gc.collect(); torch.cuda.empty_cache()

    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())
    # Only decode is exercised. Remove the unused encoder, strictly load every
    # remaining creator decoder parameter and BN buffer by its storage alias.
    del ae.encoder
    raw = load_file(args.vae_checkpoint)
    state = {}
    for name, value in ae.state_dict().items():
        tensor = raw[vae_alias(name)]
        if tensor.shape != value.shape:
            assert name.startswith("decoder.mid.attn_1.") and tensor.ndim == 2
            assert value.ndim == 4 and value.shape[2:] == (1, 1) and tensor.shape == value.shape[:2]
            tensor = tensor.reshape(value.shape)
        assert tensor.dtype == value.dtype
        state[name] = tensor.cuda()
    ae.load_state_dict(state, strict=True, assign=True)
    ae.eval()
    def decode(tokens):
        return ae.decode(tokens.T.reshape(1, 128, height, width).cuda()).clamp(-1, 1).cpu()
    same = decode(native["final_image_tokens"])
    pixels = decode(latent)
    def psnr(a, b):
        mse = (a.float() - b.float()).square().mean().item()
        return 10 * math.log10(4 / max(mse, 1e-30))
    report["same_latent_decode_psnr_db"] = psnr(native["clamped_output"], same)
    report["trajectory_png_psnr_db"] = psnr(native["clamped_output"], pixels)
    array = ((pixels[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(args.output_dir / "creator.png")
    save_file({"final_image_tokens": latent, "clamped_output": pixels,
               "same_latent_decode": same, **conditioning_captures}, args.output_dir / "creator.safetensors")
    report["passed"] = all(report[k]["cosine"] >= 0.999 and report[k]["relative_l2"] <= 0.01 and report[k]["nonfinite"] == 0
                           for k in ["positive_conditioning", "negative_conditioning"])
    report["passed"] &= (report["final_latent"]["cosine"] >= 0.995 and report["final_latent"]["relative_l2"] <= 0.1
                          and report["final_latent"]["nonfinite"] == 0
                          and report["same_latent_decode_psnr_db"] >= 50 and report["trajectory_png_psnr_db"] >= 30)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
