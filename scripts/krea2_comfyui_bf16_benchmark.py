#!/usr/bin/env python3
"""Matched Krea 2 Turbo BF16 denoiser benchmark through ComfyUI.

This is a development comparator, not a production dependency.  It imports the
existing ComfyUI runtime, loads the official BF16 Turbo checkpoint, and reuses
the frozen Diffusion Compiler conditioning/start/schedule fixture.  Guidance is
disabled: each Euler step performs exactly one conditional denoiser evaluation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import resource
import statistics
import sys
import threading
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--tokenizer-inputs", type=Path)
    parser.add_argument("--source-faithful-prefused", action="store_true")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--valid-text-tokens", type=int, default=512)
    parser.add_argument("--reserve-vram-gb", type=float, default=1.0)
    return parser.parse_args()


def tokens_to_latent(tokens):
    """Inverse of Krea's 2x2 channel-last patchification."""
    batch, count, features = tokens.shape
    if (batch, count, features) != (1, 4096, 64):
        raise ValueError(f"unexpected initial token shape {tuple(tokens.shape)}")
    return (
        tokens.reshape(batch, 64, 64, 16, 2, 2)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, 16, 128, 128)
        .contiguous()
    )


def latent_to_tokens(latent):
    batch, channels, height, width = latent.shape
    if (batch, channels, height, width) != (1, 16, 128, 128):
        raise ValueError(f"unexpected latent shape {tuple(latent.shape)}")
    return (
        latent.reshape(batch, channels, 64, 2, 64, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, 4096, 64)
        .contiguous()
    )


def load_comfy_context(path: Path, valid_text_tokens: int):
    """Load the frozen pre-TextFusion 12-layer Qwen stack for ComfyUI."""
    import torch  # pylint: disable=import-outside-toplevel
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel

    with safe_open(path, framework="pt", device="cpu") as source:
        keys = list(source.keys())
        taps = [source.get_tensor(f"tap_{index:02d}") for index in range(12)]
    if any(tuple(tap.shape) != (512, 2560) for tap in taps):
        raise ValueError(f"unexpected Qwen tap shapes in {path}: {keys}")
    if valid_text_tokens <= 0 or valid_text_tokens > 512:
        raise ValueError("--valid-text-tokens must be in [1,512]")
    return (
        torch.stack(taps, dim=1)
        .unsqueeze(0)[:, :valid_text_tokens]
        .reshape(1, valid_text_tokens, 12 * 2560)
        .contiguous()
    )


def load_prefused_context(path: Path):
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel

    with safe_open(path, framework="pt", device="cpu") as source:
        context = source.get_tensor("conditioning_output").contiguous()
    if tuple(context.shape) != (1, 512, 6144):
        raise ValueError(f"unexpected pre-fused context shape {tuple(context.shape)}")
    return context


def load_creator_text_mask(path: Path):
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel

    with safe_open(path, framework="pt", device="cpu") as source:
        mask = source.get_tensor("attention_mask")[:, 34:].bool().contiguous()
    if tuple(mask.shape) != (1, 512) or int(mask.sum()) != 109:
        raise ValueError(f"unexpected Krea creator text mask {tuple(mask.shape)}")
    return mask


def prepare_prefused_static(model, context, text_mask):
    import torch  # pylint: disable=import-outside-toplevel

    image_positions = torch.zeros(64, 64, 3, device="cuda", dtype=torch.float32)
    image_positions[..., 1] = torch.arange(64, device="cuda")[:, None]
    image_positions[..., 2] = torch.arange(64, device="cuda")[None, :]
    text_positions = torch.zeros(1, 512, 3, device="cuda", dtype=torch.float32)
    positions = torch.cat(
        (text_positions, image_positions.reshape(1, 4096, 3)), dim=1
    )
    frequencies = model.pe_embedder(positions)
    valid = torch.cat(
        (text_mask.cuda(), torch.ones(1, 4096, device="cuda", dtype=torch.bool)),
        dim=1,
    )
    attention_mask = valid[:, None, :, None] & valid[:, None, None, :]
    return context.cuda().to(torch.bfloat16), frequencies, attention_mask


def prefused_velocity(model, latent, context, frequencies, attention_mask):
    """Creator mmdit.py:387-415 with static TextFusion already hoisted."""
    import torch  # pylint: disable=import-outside-toplevel
    from comfy.ldm.flux.layers import (  # pylint: disable=import-outside-toplevel
        timestep_embedding,
    )

    image = latent_to_tokens(latent)
    image = model.first(image)
    timestep = prefused_velocity.timestep
    time_vector = model.tmlp(
        timestep_embedding(timestep, model.tdim).unsqueeze(1).to(image.dtype)
    )
    modulation = model.tproj(time_vector)
    combined = torch.cat((context, image), dim=1)
    transformer_options = {
        "prefetch_dynamic_vbars": True,
        "total_blocks": len(model.blocks),
        "block_type": "single",
        "img_slice": [512, 4608],
    }
    for index, block in enumerate(model.blocks):
        transformer_options["block_index"] = index
        combined = block(
            combined,
            modulation,
            frequencies,
            attention_mask,
            timestep_zero_index=None,
            transformer_options=transformer_options,
        )
    output = model.last(combined, time_vector)[:, 512:4608]
    return tokens_to_latent(output)


prefused_velocity.timestep = None


class NvmlPeakSampler:
    def __init__(self) -> None:
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None
        self._library = None

    def __enter__(self):
        try:
            class MemoryInfo(ctypes.Structure):
                _fields_ = [
                    ("total", ctypes.c_ulonglong),
                    ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong),
                ]

            library = ctypes.CDLL("libnvidia-ml.so.1")
            if library.nvmlInit_v2() != 0:
                raise RuntimeError("nvmlInit_v2 failed")
            handle = ctypes.c_void_p()
            if library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 failed")
            self._library = library

            def used_bytes() -> int:
                memory = MemoryInfo()
                if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                    return 0
                return int(memory.used)

            self.baseline_bytes = used_bytes()
            self.peak_bytes = self.baseline_bytes

            def poll() -> None:
                while not self._stop.wait(0.02):
                    self.peak_bytes = max(self.peak_bytes, used_bytes())

            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
        except Exception:  # NVML is supplementary; torch metrics remain.
            self._library = None
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._library is not None:
            self._library.nvmlShutdown()


def tensor_metrics(actual, expected) -> dict[str, float | int]:
    import torch  # pylint: disable=import-outside-toplevel

    a = actual.float().reshape(-1)
    b = expected.float().reshape(-1)
    delta = a - b
    norm_b = torch.linalg.vector_norm(b)
    norm_a = torch.linalg.vector_norm(a)
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "relative_l2": float(torch.linalg.vector_norm(delta) / norm_b),
        "max_absolute": float(delta.abs().max()),
        "norm_ratio": float(norm_a / norm_b),
        "nonfinite": int((~torch.isfinite(a)).sum()),
        "bit_mismatches": int((actual.view(torch.int16) != expected.view(torch.int16)).sum()),
        "elements": int(a.numel()),
    }


def main() -> None:
    args = parse_args()
    for path in (args.comfyui, args.checkpoint, args.conditioning, args.trajectory):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing benchmark artifact")
    if args.source_faithful_prefused and args.tokenizer_inputs is None:
        raise ValueError("--source-faithful-prefused requires --tokenizer-inputs")

    # Comfy normally parses its own process command line.  This standalone
    # comparator sets only the relevant production knobs before model-management
    # modules are imported.
    sys.path.insert(0, str(args.comfyui.resolve()))
    import comfy.cli_args  # pylint: disable=import-outside-toplevel

    comfy.cli_args.args.bf16_unet = True
    comfy.cli_args.args.lowvram = True
    comfy.cli_args.args.reserve_vram = args.reserve_vram_gb
    comfy.cli_args.args.preview_method = comfy.cli_args.LatentPreviewMethod.NoPreviews
    if args.source_faithful_prefused:
        comfy.cli_args.args.use_pytorch_cross_attention = True

    import comfy_aimdo.control  # pylint: disable=import-outside-toplevel

    headroom_bytes = int(args.reserve_vram_gb * 1024**3)
    try:
        comfy_aimdo.control.init(
            simple_vram_headroom=headroom_bytes,
            nvml_pressure=not comfy.cli_args.args.disable_nvml_pressure,
        )
    except TypeError:
        try:
            comfy_aimdo.control.init(simple_vram_headroom=headroom_bytes)
        except TypeError:
            comfy_aimdo.control.init()

    import torch  # pylint: disable=import-outside-toplevel
    from safetensors import safe_open  # pylint: disable=import-outside-toplevel
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel
    import comfy.memory_management  # pylint: disable=import-outside-toplevel
    import comfy.model_management as model_management  # pylint: disable=import-outside-toplevel
    import comfy.model_patcher  # pylint: disable=import-outside-toplevel
    import comfy.sd  # pylint: disable=import-outside-toplevel
    import comfy.utils  # pylint: disable=import-outside-toplevel

    if not torch.cuda.is_available():
        raise RuntimeError("ComfyUI BF16 benchmark requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False

    try:
        aimdo_initialized = comfy_aimdo.control.init_devices(
            (device.index, 0) for device in model_management.get_all_torch_devices()
        )
    except TypeError:
        aimdo_initialized = comfy_aimdo.control.init_devices(
            device.index for device in model_management.get_all_torch_devices()
        )
    if aimdo_initialized:
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True

    if args.source_faithful_prefused:
        context_cpu = load_prefused_context(args.conditioning)
        text_mask_cpu = load_creator_text_mask(args.tokenizer_inputs)
    else:
        context_cpu = load_comfy_context(args.conditioning, args.valid_text_tokens)
        text_mask_cpu = None
    with safe_open(args.trajectory, framework="pt", device="cpu") as source:
        initial_tokens = source.get_tensor("initial_image_tokens").contiguous()
        schedule = source.get_tensor("timesteps").float().contiguous()
        expected_final = source.get_tensor("final_image_tokens").contiguous()
        expected_steps = {
            index: {
                "velocity": source.get_tensor(f"step_{index}_velocity").contiguous(),
                "image": source.get_tensor(
                    f"step_{index}_image_tokens"
                ).contiguous(),
            }
            for index in range(1, int(schedule.numel()))
        }

    total_steps = int(schedule.numel() - 1)
    execute_steps = total_steps if args.stop_after == 0 else args.stop_after
    if execute_steps <= 0 or execute_steps > total_steps:
        raise ValueError(f"--stop-after must be zero or in [1,{total_steps}]")
    expected_context_shape = (
        (1, 512, 6144)
        if args.source_faithful_prefused
        else (1, args.valid_text_tokens, 12 * 2560)
    )
    if tuple(context_cpu.shape) != expected_context_shape:
        raise ValueError(f"unexpected conditioning shape {tuple(context_cpu.shape)}")

    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    report: dict[str, object] = {
        "runtime": "ComfyUI/PyTorch",
        "comparator_mode": (
            "source-faithful-prefused-masked"
            if args.source_faithful_prefused
            else "stock-krea2-frontend"
        ),
        "comfyui_commit": None,
        "checkpoint": str(args.checkpoint.resolve()),
        "conditioning": str(args.conditioning.resolve()),
        "trajectory": str(args.trajectory.resolve()),
        "dtype": "BF16",
        "geometry": (
            f"1024x1024_B1_text{args.valid_text_tokens}_image4096_patch2"
        ),
        "steps": total_steps,
        "executed_steps": execute_steps,
        "creator_guidance": 0.0,
        "comfy_cfg": 1.0,
        "unconditional_evaluations": 0,
        "mu_shift": 1.15,
        "dynamic_vram_initialized": bool(aimdo_initialized),
        "attention_mask_note": (
            "creator 512-token boolean outer mask supplied to every block"
            if args.source_faithful_prefused
            else "ComfyUI Krea2 does not expose the creator padding mask through "
            "BaseModel.extra_conds; 512 is the same-shape timing arm, while 109 "
            "is the normal valid-token-only ComfyUI semantic arm"
        ),
        "steps_detail": [],
    }
    try:
        import subprocess  # pylint: disable=import-outside-toplevel

        report["comfyui_commit"] = subprocess.check_output(
            ["git", "-C", str(args.comfyui), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        pass

    with NvmlPeakSampler() as nvml:
        load_started = time.perf_counter()
        state_dict, metadata = comfy.utils.load_safetensors(str(args.checkpoint.resolve()))
        patcher = comfy.sd.load_diffusion_model_state_dict(
            state_dict,
            model_options={"dtype": torch.bfloat16},
            metadata=metadata,
        )
        if patcher is None:
            raise RuntimeError("ComfyUI could not detect the Krea 2 BF16 checkpoint")
        report["checkpoint_load_seconds"] = time.perf_counter() - load_started

        stage_started = time.perf_counter()
        model_management.load_model_gpu(patcher)
        torch.cuda.synchronize()
        report["initial_stage_seconds"] = time.perf_counter() - stage_started
        report["dynamic_model"] = bool(patcher.is_dynamic())
        report["model_loaded_vram_bytes"] = int(patcher.loaded_size())

        latent = tokens_to_latent(initial_tokens).cuda()
        if args.source_faithful_prefused:
            context, frequencies, attention_mask = prepare_prefused_static(
                patcher.model.diffusion_model, context_cpu, text_mask_cpu
            )
        else:
            latent = latent.float()
            context = context_cpu.cuda().to(torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()

        captures = {"initial_image_tokens": initial_tokens}
        step_wall: list[float] = []
        step_device_ms: list[float] = []
        with torch.inference_mode():
            for index in range(execute_steps):
                sigma_value = float(schedule[index])
                next_value = float(schedule[index + 1])
                sigma = torch.full((1,), sigma_value, device="cuda", dtype=torch.float32)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                wall_started = time.perf_counter()
                begin.record()
                if args.source_faithful_prefused:
                    prefused_velocity.timestep = torch.full(
                        (1,), sigma_value, device="cuda", dtype=torch.bfloat16
                    )
                    velocity = prefused_velocity(
                        patcher.model.diffusion_model,
                        latent,
                        context,
                        frequencies,
                        attention_mask,
                    )
                    latent = (
                        latent + (next_value - sigma_value) * velocity
                    ).to(torch.bfloat16)
                else:
                    denoised = patcher.model.apply_model(
                        latent,
                        sigma,
                        c_crossattn=context,
                    )
                    velocity = (latent - denoised) / sigma_value
                    latent = latent + (next_value - sigma_value) * velocity
                end.record()
                torch.cuda.synchronize()
                wall_seconds = time.perf_counter() - wall_started
                device_ms = float(begin.elapsed_time(end))
                step_wall.append(wall_seconds)
                step_device_ms.append(device_ms)
                tokens = latent_to_tokens(latent.to(torch.bfloat16)).cpu()
                velocity_tokens = latent_to_tokens(velocity.to(torch.bfloat16)).cpu()
                captures[f"step_{index + 1}_image_tokens"] = tokens
                captures[f"step_{index + 1}_velocity"] = velocity_tokens
                expected_velocity = expected_steps[index + 1]["velocity"]
                expected_image = expected_steps[index + 1]["image"]
                report["steps_detail"].append(
                    {
                        "step": index + 1,
                        "conditional_wall_seconds": wall_seconds,
                        "conditional_device_ms": device_ms,
                        "unconditional_seconds": 0.0,
                        "velocity_metrics": tensor_metrics(
                            velocity_tokens, expected_velocity
                        ),
                        "image_metrics": tensor_metrics(tokens, expected_image),
                    }
                )
                print(
                    f"KREA2_COMFYUI_BF16 step={index + 1}/{execute_steps} "
                    f"wall_s={wall_seconds:.3f} device_ms={device_ms:.3f} cfg=off",
                    flush=True,
                )

        final_tokens = latent_to_tokens(latent.to(torch.bfloat16)).cpu()
        captures["final_image_tokens"] = final_tokens
        if execute_steps == total_steps:
            report["final_metrics_vs_creator"] = tensor_metrics(final_tokens, expected_final)

        report["denoiser_wall_seconds"] = sum(step_wall)
        report["per_step_wall_mean_seconds"] = sum(step_wall) / len(step_wall)
        report["per_step_wall_median_seconds"] = statistics.median(step_wall)
        report["per_step_wall_min_seconds"] = min(step_wall)
        report["per_step_wall_max_seconds"] = max(step_wall)
        report["device_total_ms"] = sum(step_device_ms)
        report["device_per_step_mean_ms"] = sum(step_device_ms) / len(step_device_ms)
        report["torch_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        report["torch_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        report["nvml_baseline_used_bytes"] = nvml.baseline_bytes
        report["nvml_peak_used_bytes"] = nvml.peak_bytes

    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    report["wall_seconds"] = time.perf_counter() - started
    report["peak_host_rss_bytes"] = int(usage_after.ru_maxrss * 1024)
    report["minor_page_faults"] = int(usage_after.ru_minflt - usage_before.ru_minflt)
    report["major_page_faults"] = int(usage_after.ru_majflt - usage_before.ru_majflt)
    report["output"] = str(args.output.resolve())
    save_file({name: value.contiguous() for name, value in captures.items()}, args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
