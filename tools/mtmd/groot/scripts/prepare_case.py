#!/usr/bin/env python3
"""Turn one replayed LIBERO observation into the case directory the llama.cpp
pipeline reads, and keep the baseline outputs next to it for comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True, help="reference/episodeXX_stepNNNN.npz")
    p.add_argument("--instruction", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.case) as z:
        front = z["raw_image_image"].reshape(256, 256, 3)
        wrist = z["raw_image_wrist_image"].reshape(256, 256, 3)
        state = np.concatenate([
            np.asarray(z["raw_state_" + k], np.float32).reshape(-1)
            for k in KEYS
        ])
        initial = z["initial_actions"].astype(np.float32).reshape(40, 132)
        expected = z["normalized_actions"].astype(np.float32).reshape(40, 132)
        physical = np.concatenate([
            np.asarray(z["action_" + k], np.float32).reshape(16, -1)[:, :1]
            for k in KEYS
        ], axis=1)

    (a.out / "front.u8").write_bytes(front.tobytes())
    (a.out / "wrist.u8").write_bytes(wrist.tobytes())
    (a.out / "state.f32").write_bytes(state.astype("<f4").tobytes())
    (a.out / "initial_actions.f32").write_bytes(initial.astype("<f4").tobytes())
    (a.out / "instruction.txt").write_text(a.instruction + "\n")
    (a.out / "expected_normalized.f32").write_bytes(expected.astype("<f4").tobytes())
    (a.out / "expected_physical.f32").write_bytes(physical.astype("<f4").tobytes())
    meta = {"case": a.case.name, "instruction": a.instruction,
            "state": state.tolist(), "physical_shape": list(physical.shape)}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"prepared {a.out}")


if __name__ == "__main__":
    main()
