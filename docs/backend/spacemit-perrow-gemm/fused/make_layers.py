#!/usr/bin/env python3
"""Synthetic layerNN.bin files in the llm_fused.inc layout (16 layers, 50,480,128 B each).

Order per layer: 7 INT8 weights (q,k,v,o,gate,up,down; n*k bytes each), 7 f32 row
scales (n each), smooth in/o/ffn/down (2048,2048,2048,6144), norms attn/q/k/ffn
(2048,128,128,2048). Values are small and finite so the run is numerically quiet.
"""
import os
import sys

import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
ns = [2048, 1024, 1024, 2048, 6144, 6144, 2048]
ks = [2048, 2048, 2048, 2048, 2048, 2048, 6144]
rng = np.random.default_rng(1)
for layer in range(16):
    parts = [rng.integers(-127, 128, size=n * k, dtype=np.int8).tobytes() for n, k in zip(ns, ks)]
    parts += [np.full(n, 0.05 / (127 * np.sqrt(k)), np.float32).tobytes() for n, k in zip(ns, ks)]
    parts += [np.ones(n, np.float32).tobytes() for n in (2048, 2048, 2048, 6144)]
    parts += [np.ones(n, np.float32).tobytes() for n in (2048, 128, 128, 2048)]
    blob = b"".join(parts)
    assert len(blob) == 50480128, len(blob)
    with open(os.path.join(out, f"layer{layer:02d}.bin"), "wb") as f:
        f.write(blob)
np.ones(2048, np.float32).tofile(os.path.join(out, "norm.f32"))
print("wrote", out)
