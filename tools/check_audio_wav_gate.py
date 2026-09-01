#!/usr/bin/env python3
"""Artifact gate for the native BigVGAN audio decode: compare a decoded WAV
against the accepted artifact's audio.wav.

Bars (from the plan's honest-bar reasoning — byte-identity of the file is
NOT the bar; different executor, different reduction order):
  - 44-byte header byte-identical
  - per-sample int16 |delta| <= 1
  - differing samples <= 1%
  - SNR vs the recorded wav >= 60 dB
"""

import sys
import wave
from pathlib import Path

import numpy


def read_wav(path: Path):
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(params.nframes)
    return params, numpy.frombuffer(frames, dtype=numpy.int16)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: check_audio_wav_gate.py ACCEPTED.wav OURS.wav")
    accepted_path, ours_path = Path(sys.argv[1]), Path(sys.argv[2])
    accepted_header = accepted_path.read_bytes()[:44]
    ours_header = ours_path.read_bytes()[:44]
    header_ok = accepted_header == ours_header

    _, accepted = read_wav(accepted_path)
    _, ours = read_wav(ours_path)
    if accepted.shape != ours.shape:
        raise SystemExit(f"AUDIO_WAV_GATE FAIL shape {accepted.shape} vs {ours.shape}")
    delta = ours.astype(numpy.int32) - accepted.astype(numpy.int32)
    max_delta = int(numpy.abs(delta).max())
    differing = int((delta != 0).sum())
    fraction = differing / delta.size
    signal_power = float((accepted.astype(numpy.float64) ** 2).mean())
    noise_power = float((delta.astype(numpy.float64) ** 2).mean())
    snr_db = float("inf") if noise_power == 0 else \
        10.0 * numpy.log10(signal_power / noise_power)

    ok = header_ok and max_delta <= 1 and fraction <= 0.01 and snr_db >= 60.0
    print(f"AUDIO_WAV_GATE {'PASS' if ok else 'FAIL'} "
          f"header_identical={header_ok} samples={delta.size} "
          f"max_int16_delta={max_delta} differing={differing} "
          f"differing_fraction={fraction:.6f} snr_db={snr_db:.2f} "
          f"bars={{header:identical,|d|<=1,frac<=1%,snr>=60dB}}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
