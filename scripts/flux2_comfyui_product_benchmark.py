#!/usr/bin/env python3
"""Run the frozen FLUX.2 Base 9B prompt-to-PNG workload through ComfyUI."""

import argparse
import hashlib
import json
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def process_status(pid):
    if pid is None:
        return {}
    values = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
    return values


def process_io(pid):
    if pid is None:
        return {}
    values = {}
    for line in Path(f"/proc/{pid}/io").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value)
    return values


def gpu_sample():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory_mib, utilization, temperature, power_w = (part.strip() for part in output.split(","))
    return {
        "memory_used_mib": int(memory_mib),
        "utilization_percent": int(utilization),
        "temperature_c": int(temperature),
        "power_w": float(power_w),
    }


def workflow(prefix):
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "flux-2-klein-base-9b.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "A cat holding a sign that says hello world", "clip": ["2", 0]},
        },
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "6": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 4.0},
        },
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": 20260901}},
        "8": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "9": {"class_type": "Flux2Scheduler", "inputs": {"steps": 50, "width": 1024, "height": 1024}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["6", 0],
                "sampler": ["10", 0],
                "sigmas": ["9", 0],
                "latent_image": ["8", 0],
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    args = parser.parse_args()

    prefix = "flux2-klein-base-9b-matched-5080/comfyui-bf16-full50"
    prompt = workflow(prefix)
    system_stats = request_json(f"{args.server}/system_stats")
    prompt_sha256 = hashlib.sha256(json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    gpu_samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                gpu_samples.append(gpu_sample())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            stop.wait(args.poll_seconds)

    io_before = process_io(args.server_pid)
    rss_before = process_status(args.server_pid)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started_ns = time.time_ns()
    started = time.perf_counter()
    queued = request_json(
        f"{args.server}/prompt",
        {"prompt": prompt, "client_id": str(uuid.uuid4())},
    )
    prompt_id = queued["prompt_id"]
    history = None
    while history is None:
        time.sleep(0.2)
        response = request_json(f"{args.server}/history/{prompt_id}")
        candidate = response.get(prompt_id)
        if candidate is not None and candidate.get("status", {}).get("completed"):
            history = candidate
    ended = time.perf_counter()
    ended_ns = time.time_ns()
    stop.set()
    monitor_thread.join()
    io_after = process_io(args.server_pid)
    rss_after = process_status(args.server_pid)

    messages = history["status"].get("messages", [])
    server_start_ns = None
    server_end_ns = None
    for message, data in messages:
        if message == "execution_start":
            server_start_ns = data.get("timestamp")
        elif message == "execution_success":
            server_end_ns = data.get("timestamp")

    image = history["outputs"]["13"]["images"][0]
    output_path = args.comfyui / image["type"] / image.get("subfolder", "") / image["filename"]
    image_bytes = output_path.read_bytes()
    report = {
        "runtime": "stock ComfyUI/PyTorch",
        "comfyui_commit": subprocess.check_output(
            ["git", "-C", str(args.comfyui), "rev-parse", "HEAD"], text=True
        ).strip(),
        "workload": {
            "model": "FLUX.2-klein-base-9B",
            "transformer": "flux-2-klein-base-9b.safetensors",
            "transformer_weight_dtype": "default/BF16",
            "text_encoder": "qwen_3_8b_fp8mixed.safetensors",
            "vae": "flux2-vae.safetensors",
            "prompt": "A cat holding a sign that says hello world",
            "negative_prompt": "",
            "seed": 20260901,
            "width": 1024,
            "height": 1024,
            "steps": 50,
            "guidance": 4.0,
            "scheduler": "Flux2Scheduler generalized-time/SNR shift",
            "sampler": "euler",
            "product_boundary": "queued literal prompt through saved PNG",
        },
        "prompt_graph_sha256": prompt_sha256,
        "prompt_id": prompt_id,
        "timing_ms": {
            "client_queue_to_history_complete": (ended - started) * 1000.0,
            "client_start_unix_ns": started_ns,
            "client_end_unix_ns": ended_ns,
            "server_execution": None
            if server_start_ns is None or server_end_ns is None
            else float(server_end_ns - server_start_ns),
        },
        "output": {
            "path": str(output_path),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "bytes": len(image_bytes),
        },
        "gpu_monitor": {
            "samples": len(gpu_samples),
            "peak_memory_used_mib": max((sample["memory_used_mib"] for sample in gpu_samples), default=None),
            "peak_utilization_percent": max((sample["utilization_percent"] for sample in gpu_samples), default=None),
            "peak_temperature_c": max((sample["temperature_c"] for sample in gpu_samples), default=None),
            "peak_power_w": max((sample["power_w"] for sample in gpu_samples), default=None),
        },
        "server_process": {
            "pid": args.server_pid,
            "rss_before": rss_before,
            "rss_after": rss_after,
            "io_delta": {
                key: io_after.get(key, 0) - io_before.get(key, 0)
                for key in set(io_before) | set(io_after)
            },
        },
        "system_stats": system_stats,
        "history_status": history["status"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
