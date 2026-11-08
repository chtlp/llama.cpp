#!/usr/bin/env python3
"""Make a per-row-INT8 test GGUF from a Q8_0 Qwen3 GGUF.

Keeps the first --layers blocks. Every 2D Q8_0 weight inside a block is
dequantized and requantized to symmetric INT8 with one scale per output row,
written back as Q8_0 with that scale repeated in every 32-block (the layout the
spacemit per-row kernel detects). With --keep-q8 the block weights are copied
unchanged (standard Q8_0, same truncation) for an apples-to-apples baseline.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, sys.argv.pop(1))  # path to gguf-py
import gguf  # noqa: E402

Q8 = np.dtype([("d", "<f2"), ("qs", "i1", 32)])


def to_perrow(raw: np.ndarray, n: int, k: int) -> np.ndarray:
    blk = raw.reshape(-1).view(Q8).reshape(n, k // 32)
    w = blk["qs"].astype(np.float32) * blk["d"].astype(np.float32)[..., None]
    w = w.reshape(n, k)
    d = (np.abs(w).max(axis=1) / 127.0).astype(np.float16)
    df = d.astype(np.float32)
    inv = np.where(df > 0, 1.0 / np.where(df > 0, df, 1), 0)
    q = np.clip(np.rint(w * inv[:, None]), -127, 127).astype(np.int8)
    out = np.empty((n, k // 32), dtype=Q8)
    out["d"] = d[:, None]
    out["qs"] = q.reshape(n, k // 32, 32)
    return out.view(np.uint8).reshape(n, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--keep-q8", action="store_true")
    a = ap.parse_args()

    r = gguf.GGUFReader(a.src)
    arch = r.fields["general.architecture"]
    arch = bytes(arch.parts[arch.data[0]]).decode()
    w = gguf.GGUFWriter(a.dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if name == f"{arch}.block_count":
            w.add_uint32(name, a.layers)
            continue
        t = f.types[0]
        if t == gguf.GGUFValueType.ARRAY:
            st = f.types[1]
            if st == gguf.GGUFValueType.STRING:
                vals = [bytes(f.parts[i]).decode("utf-8", errors="surrogateescape") for i in f.data]
            else:
                vals = [f.parts[i].tolist()[0] for i in f.data]
            w.add_array(name, vals)
        elif t == gguf.GGUFValueType.STRING:
            w.add_string(name, bytes(f.parts[f.data[0]]).decode("utf-8", errors="surrogateescape"))
        else:
            w.add_key_value(name, f.parts[f.data[0]].tolist()[0], t)

    nconv = 0
    for t in r.tensors:
        if t.name.startswith("blk."):
            if int(t.name.split(".")[1]) >= a.layers:
                continue
        data = t.data
        shape = [int(x) for x in t.shape]  # ggml order: ne0 = k, ne1 = n
        if (not a.keep_q8 and t.name.startswith("blk.") and t.tensor_type == gguf.GGMLQuantizationType.Q8_0
                and len(shape) == 2):
            k, n = shape
            data = to_perrow(np.asarray(data), n, k)
            nconv += 1
        w.add_tensor(t.name, np.asarray(data), raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.dst}: {a.layers} layers, {nconv} tensors converted to per-row")


if __name__ == "__main__":
    main()
