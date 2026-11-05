#!/usr/bin/env python3
"""Compare one pipeline run with the reference actions and, optionally, with a
baseline trace produced by the delivered groot-n17-k3 runtime."""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def metrics(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    cos = float(a @ b / np.sqrt(max((a @ a) * (b @ b), 1e-30)))
    return {"cos": cos, "rmse": float(np.sqrt(((a - b) ** 2).mean())),
            "max": float(np.abs(a - b).max())}


def decode(normalized, contract):
    out, start = [], 0
    for key in KEYS:
        p = contract["norm"][key]
        d = int(p["dim"])
        v = normalized[..., :16, start:start + d].astype(np.float32)
        lo = np.asarray(p["min"], np.float32)
        hi = np.asarray(p["max"], np.float32)
        out.append((np.clip(v, -1, 1) + 1) * 0.5 * (hi - lo) + lo)
        start += d
    return np.concatenate(out, axis=2).reshape(16, 7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="pipeline output directory")
    p.add_argument("--case", type=Path, required=True, help="prepared case directory")
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--baseline-trace", type=Path)
    a = p.parse_args()

    contract = json.loads(a.contract.read_text())
    expected = np.fromfile(a.case / "expected_physical.f32", np.float32).reshape(16, 7)
    ours = np.fromfile(a.out / "physical_actions.f32", np.float32).reshape(16, 7)
    print("pipeline", json.dumps(metrics(expected, ours)))

    if a.baseline_trace:
        trace = np.load(a.baseline_trace)
        base = decode(trace["normalized_actions"].reshape(1, 40, 132), contract)
        print("baseline", json.dumps(metrics(expected, base)))
        for stage, key in (("image_embeds", "image_embeds"), ("llm", "llm_output"),
                           ("vlln", "vlln_output"), ("vl", "vl_output")):
            f = a.out / f"{stage}.f16"
            if f.exists() and key in trace.files:
                print(stage, json.dumps(metrics(np.fromfile(f, np.float16), trace[key])))


if __name__ == "__main__":
    main()
