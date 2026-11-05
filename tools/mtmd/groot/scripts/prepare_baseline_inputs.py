#!/usr/bin/env python3
"""Inputs for the baseline runtime, built from the reference record so the
baseline's own tokenizer and preprocessing are not in the comparison path."""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def bf16_as_half(x):
    f = np.asarray(x, np.float32).copy()
    u = f.view(np.uint32)
    u += np.uint32(0x7FFF) + ((u >> 16) & 1)
    u &= np.uint32(0xFFFF0000)
    return f.astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--pixels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    contract = json.loads(a.contract.read_text())
    with np.load(a.npz) as z:
        tokens = z["input_ids"].reshape(-1).astype(np.int32)
        initial = z["initial_actions"].astype(np.float32).reshape(1, 40, 132)
        state = np.concatenate([
            np.asarray(z["raw_state_" + k], np.float32).reshape(-1) for k in KEYS
        ])

    lo, hi = [], []
    for k, d in zip(KEYS, [1, 1, 1, 1, 1, 1, 2]):
        pmin = contract["state_norm"][k]["min"]
        pmax = contract["state_norm"][k]["max"]
        lo.extend(np.asarray(pmin, np.float32).reshape(-1)[:d])
        hi.extend(np.asarray(pmax, np.float32).reshape(-1)[:d])
    state_norm = np.zeros((1, 1, 132), np.float32)
    state_norm[0, 0, :8] = np.clip(2 * (state - np.asarray(lo)) / (np.asarray(hi) - np.asarray(lo)) - 1,
                                   -1, 1)
    state_norm = bf16_as_half(state_norm)

    image_token = 151655  # <|image_pad|> in the GR00T tokenizer
    visual = (tokens == image_token).astype(np.uint8)
    positions = np.zeros((3, tokens.size), np.int32)
    offset = 0
    i = 0
    images = 0
    while i < tokens.size:
        if not visual[i]:
            positions[:, i] = offset
            i += 1
            offset += 1
            continue
        assert i + 64 <= tokens.size and visual[i:i + 64].all()
        positions[0, i:i + 64] = offset
        positions[1, i:i + 64] = offset + np.repeat(np.arange(8), 8)
        positions[2, i:i + 64] = offset + np.tile(np.arange(8), 8)
        offset += 8
        i += 64
        images += 1
    assert images == 2

    pixels = np.fromfile(a.pixels, np.float16).reshape(512, 1536)
    np.savez(a.out, pixel_values=pixels, input_ids=tokens.reshape(1, -1),
             positions=positions, visual_mask=visual,
             attention_mask=np.ones(tokens.size, np.uint8),
             state=state_norm, initial_actions=initial)
    print(f"wrote {a.out}: tokens={tokens.size} images={images}")


if __name__ == "__main__":
    main()
