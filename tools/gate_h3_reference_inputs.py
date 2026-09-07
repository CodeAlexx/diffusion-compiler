#!/usr/bin/env python3
"""Small native CLI integration checks; no neural runtime or oracle imports."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def tensor(path, dtype, shape, payload):
    data = b"DIFTNS01" + struct.pack("<III", 1, dtype, len(shape))
    data += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<Q", len(payload)) + payload
    path.write_bytes(data + hashlib.sha256(data).digest())


def read(path):
    data = path.read_bytes()
    assert data[:8] == b"DIFTNS01" and hashlib.sha256(data[:-32]).digest() == data[-32:]
    version, dtype, rank = struct.unpack_from("<III", data, 8)
    assert version == 1
    shape = struct.unpack_from(f"<{rank}Q", data, 20)
    size = struct.unpack_from("<Q", data, 20 + rank * 8)[0]
    assert len(data) == 28 + rank * 8 + size + 32
    return dtype, shape, data[28 + rank * 8:-32]


def safetensors(path, values):
    header, payload = {}, b""
    for name, items in values.items():
        raw = struct.pack(f"<{len(items)}i", *items)
        header[name] = {"dtype": "I32", "shape": [len(items)],
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload += raw
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_safe(path, name):
    data = path.read_bytes()
    size = struct.unpack_from("<Q", data)[0]
    item = json.loads(data[8:8 + size])[name]
    lo, hi = item["data_offsets"]
    return list(struct.unpack(f"<{(hi-lo)//4}i", data[8 + size + lo:8 + size + hi]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--processor", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    def run(tool, *args, ok=True):
        result = subprocess.run([str(a.build / tool), *map(str, args)], capture_output=True, text=True)
        if ok and result.returncode:
            raise AssertionError(result.stderr)
        if not ok:
            assert result.returncode != 0, result.stdout
        return result

    parts = []
    expected_ids, expected_tags, expected_map, expected_positions = [], [], [], []
    visual_count = 0
    # Actual-source order: per-modality labels, interleaved references, prompt last.
    for index, kind in enumerate(["audio", "image", "audio", "image", "prompt"]):
        if kind != "image":
            label = "<Audio 1>: " if index == 0 else "<Audio 2>: " if index == 2 else "A person talks.\n"
            path = a.output / f"text-{index}.diftensor"
            run("diftokenize", "--processor", a.processor, "--prompt", label,
                "--diftensor-out", path, "--quiet")
            dtype, shape, raw = read(path)
            assert dtype == 5
            ids = list(struct.unpack(f"<{shape[0]}i", raw))
            parts += ["--text-input", path]
            expected_ids += ids
            expected_tags += [1] * len(ids)
            expected_map += [-1] * len(ids)
        else:
            # The learned vision path is unchanged; isolate destination-offset glue.
            ids = [100 + index, 151652, 151655, 151653]
            path = a.output / f"image-{index}.safetensors"
            safetensors(path, {"input_ids": ids, "token_tags": [1, 0, 0, 0],
                              "vision_destination_map": [-1, -1, 0, -1], "visual_positions": [2]})
            parts += ["--vision-input", path]
            expected_positions += [len(expected_ids) + 2]
            expected_ids += ids
            expected_tags += [1, 0, 0, 0]
            expected_map += [-1, -1, visual_count, -1]
            visual_count += 1
    combined = a.output / "combined.safetensors"
    run("difh3vision", "combine-inputs", *parts, "--output", combined,
        "--ids-out", a.output / "ids.diftensor", "--tags-out", a.output / "tags.diftensor")
    for name, expected in [("input_ids", expected_ids), ("token_tags", expected_tags),
                           ("vision_destination_map", expected_map), ("visual_positions", expected_positions)]:
        assert read_safe(combined, name) == expected, name

    # Clean audio is a bit copy: include -0 and a subnormal to catch needless math.
    clean1 = struct.pack("<32I", 0x80000000, 1, *([0x3e800000] * 30))
    clean2 = struct.pack("<64f", *[i / 128 for i in range(64)])
    target = struct.pack("<32f", *[-i / 64 for i in range(32)])
    c1, c2, noise = [a.output / f"{name}.diftensor" for name in ("clean1", "clean2", "noise")]
    tensor(c1, 1, [1, 32], clean1)
    tensor(c2, 1, [2, 32], clean2)
    tensor(noise, 1, [1, 32], target)
    state = a.output / "audio-state.diftensor"
    args = ["--clean-conditions", "--condition", c1, "--condition", c2, "--target-noise", noise]
    run("difh3state", *args, "--output", state)
    assert read(state) == (1, (4, 32), clean1 + clean2 + target)
    run("difh3state", *args, "--condition-timestep", "1", "--output", a.output / "invalid-t.diftensor", ok=False)
    run("difh3state", *args, "--condition-noise", noise, "--output", a.output / "invalid-noise.diftensor", ok=False)
    run("difh3state", *args, "--output", state, ok=False)
    run("difh3vision", "combine-inputs", "--text-input", a.output / "text-0.diftensor",
        "--output", a.output / "audio-only.safetensors", "--ids-out", a.output / "audio-only-ids.diftensor",
        "--tags-out", a.output / "audio-only-tags.diftensor", ok=False)
    # Stop deliberately at program open, after the real inference CLI computes
    # modality ranges. No learned model is needed to catch its former invalid
    # assumption that reference audio is contiguous with target audio.
    layout_cases = [
        ([], "condition_video=[] audio=[2,4) target_video=[4,5)"),
        (["audio:0:0:0:1", "image:1:2:2:0"],
         "condition_video=[4,5) audio=[2,4),[5,7) target_video=[7,8)"),
        (["image:1:2:2:0", "audio:0:0:0:1"],
         "condition_video=[2,3) audio=[3,7) target_video=[7,8)"),
        (["audio:0:0:0:1", "image:1:2:2:0", "audio:0:0:0:1", "image:1:2:2:0"],
         "condition_video=[4,5),[7,8) audio=[2,4),[5,7),[8,10) target_video=[10,11)"),
    ]
    for index, (geometries, expected) in enumerate(layout_cases):
        refs = [item for geometry in geometries for item in ("--reference-geometry", geometry)]
        result = run("difh3infer", "--backend", "cpu", "--denoise-only",
                     "--denoiser-program", a.output / "missing-layout-check.difir",
                     "--denoiser-bundle", a.output / "unused.difbind",
                     "--all-text-tokens", 2, "--text", noise, "--video", noise, "--audio", noise,
                     "--latent-t", 1, "--latent-h", 2, "--latent-w", 2, "--audio-latents", 1,
                     "--simple-steps", 1, *refs, "--output-latent", a.output / f"unused-video-{index}",
                     "--output-audio", a.output / f"unused-audio-{index}", ok=False)
        assert expected in result.stdout, (index, result.stdout, result.stderr)
        assert "missing-layout-check.difir" in result.stderr, result.stderr
    print(json.dumps({"passed": 13, "scope": "native label/map, clean-row and inference-layout integration",
                      "neural_execution": False, "output": str(a.output)}))


if __name__ == "__main__":
    main()
