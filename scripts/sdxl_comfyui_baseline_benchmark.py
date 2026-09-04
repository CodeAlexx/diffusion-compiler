#!/usr/bin/env python3
"""Run the frozen SDXL prompt-to-PNG workload through a running ComfyUI server.

The baseline for the SDXL speed work: stock ComfyUI/PyTorch, single-file
sd_xl_base_1.0.safetensors (UNet + CLIP-L + OpenCLIP-G + VAE), the standard
CheckpointLoaderSimple -> CLIPTextEncode x2 -> EmptyLatentImage -> KSampler ->
VAEDecode -> SaveImage graph. Measures the product boundary: queued literal
prompt through the saved PNG (client wall), plus the server's own execution
timestamps, GPU samples, server RSS and I/O deltas.

Usage:
  sdxl_comfyui_baseline_benchmark.py --comfyui /home/alex/SwarmUI/dlbackend/ComfyUI \
      --server http://127.0.0.1:8189 --server-pid PID --report out.json
      [--steps 25 --cfg 5.0 --sampler euler --scheduler normal --seed 20260901]
"""
import argparse
import hashlib
import json
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

PROMPT = "A cat holding a sign that says hello world"
CHECKPOINT = "sd_xl_base_1.0.safetensors"


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
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
         "--format=csv,noheader,nounits"], text=True).strip()
    memory_mib, utilization, temperature, power_w = (part.strip() for part in output.split(","))
    return {"memory_used_mib": int(memory_mib), "utilization_percent": int(utilization),
            "temperature_c": int(temperature), "power_w": float(power_w)}


def workflow(args, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": args.prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": args.negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": args.width, "height": args.height, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": args.seed, "steps": args.steps,
                         "cfg": args.cfg, "sampler_name": args.sampler,
                         "scheduler": args.scheduler, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8189")
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--tag", default="run")
    parser.add_argument("--prompt", default=PROMPT,
                        help="positive prompt (default: the frozen workload prompt); a different "
                             "text on a warm-up run keeps ComfyUI's node cache from serving the "
                             "CLIP encodes on the measured run")
    parser.add_argument("--negative", default="")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="normal")
    args = parser.parse_args()
    prefix = f"sdxl-base-baseline/comfyui-{args.tag}"
    prompt = workflow(args, prefix)
    system_stats = request_json(f"{args.server}/system_stats")
    prompt_sha256 = hashlib.sha256(
        json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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
    queued = request_json(f"{args.server}/prompt", {"prompt": prompt, "client_id": str(uuid.uuid4())})
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
    server_start_ns = server_end_ns = None
    for message, data in history["status"].get("messages", []):
        if message == "execution_start":
            server_start_ns = data.get("timestamp")
        elif message == "execution_success":
            server_end_ns = data.get("timestamp")
    image = history["outputs"]["7"]["images"][0]
    output_path = args.comfyui / image["type"] / image.get("subfolder", "") / image["filename"]
    image_bytes = output_path.read_bytes()
    report = {
        "runtime": "stock ComfyUI/PyTorch",
        "comfyui_commit": subprocess.check_output(
            ["git", "-C", str(args.comfyui), "rev-parse", "HEAD"], text=True).strip(),
        "workload": {
            "model": "SDXL base 1.0 (single file: UNet + CLIP-L + OpenCLIP-G + VAE, fp16)",
            "checkpoint": CHECKPOINT,
            "prompt": args.prompt, "negative_prompt": args.negative, "seed": args.seed,
            "width": args.width, "height": args.height, "steps": args.steps, "cfg": args.cfg,
            "sampler": args.sampler, "scheduler": args.scheduler,
            "product_boundary": "queued literal prompt through saved PNG",
        },
        "prompt_graph_sha256": prompt_sha256,
        "prompt_id": prompt_id,
        "timing_ms": {
            "client_queue_to_history_complete": (ended - started) * 1000.0,
            "client_start_unix_ns": started_ns,
            "client_end_unix_ns": ended_ns,
            "server_execution": None if server_start_ns is None or server_end_ns is None
            else float(server_end_ns - server_start_ns),
        },
        "output": {"path": str(output_path), "sha256": hashlib.sha256(image_bytes).hexdigest(),
                   "bytes": len(image_bytes)},
        "gpu_monitor": {
            "samples": len(gpu_samples),
            "peak_memory_used_mib": max((s["memory_used_mib"] for s in gpu_samples), default=None),
            "peak_utilization_percent": max((s["utilization_percent"] for s in gpu_samples), default=None),
            "peak_temperature_c": max((s["temperature_c"] for s in gpu_samples), default=None),
            "peak_power_w": max((s["power_w"] for s in gpu_samples), default=None),
        },
        "server_process": {
            "pid": args.server_pid, "rss_before": rss_before, "rss_after": rss_after,
            "io_delta": {key: io_after.get(key, 0) - io_before.get(key, 0)
                         for key in set(io_before) | set(io_after)},
        },
        "system_stats": system_stats,
        "history_status": history["status"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("timing_ms", "output", "gpu_monitor")}, indent=2))


if __name__ == "__main__":
    main()
