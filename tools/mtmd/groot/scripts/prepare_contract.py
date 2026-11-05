#!/usr/bin/env python3
"""Pack the LIBERO normalization contract and the FP16 glue constants into the
flat binary the llama.cpp pipeline reads.

Layout (little endian, matches groot-vla.cpp):
  'GRTCTR01', int32 version=1
  float32 state_lo[8], state_hi[8], action_lo[7], action_hi[7]
  float16 vlln_weight[2048], vlln_bias[2048]
  float16 time_embedding[4*40*1536]
  float16 position_embedding[40*1536]
  float16 adaln[4*32*3072]
  float16 output_mod[4*3072]
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
DIMS = [1, 1, 1, 1, 1, 1, 2]


def flatten_norm(section):
    lo, hi = [], []
    for key, dim in zip(KEYS, DIMS):
        p = section[key]
        lo.extend(np.asarray(p["min"], np.float32).reshape(-1)[:dim])
        hi.extend(np.asarray(p["max"], np.float32).reshape(-1)[:dim])
    return np.asarray(lo, np.float32), np.asarray(hi, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--constants", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    contract = json.loads(a.contract.read_text())
    state_lo, state_hi = flatten_norm(contract["state_norm"])
    action_lo, action_hi = flatten_norm(contract["norm"])
    assert state_lo.size == 8 and action_lo.size == 7

    with np.load(a.constants) as z:
        vlln_w = z["vlln_weight"].astype(np.float16).reshape(-1)
        vlln_b = z["vlln_bias"].astype(np.float16).reshape(-1)
        time = z["action_time_embeddings"].astype(np.float16).reshape(4 * 40 * 1536)
        pos = z["position_embedding"].astype(np.float16).reshape(-1)[: 40 * 1536]
        adaln = z["adaln_modulations"].astype(np.float16).reshape(4 * 32 * 3072)
        outmod = z["output_modulations"].astype(np.float16).reshape(4 * 3072)

    with a.out.open("wb") as f:
        f.write(b"GRTCTR01")
        f.write(struct.pack("<i", 1))
        for array in (state_lo, state_hi, action_lo, action_hi):
            f.write(array.astype("<f4").tobytes())
        for array in (vlln_w, vlln_b, time, pos, adaln, outmod):
            f.write(array.astype("<f2").tobytes())
    print(f"wrote {a.out} ({a.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
